import os
import sys
import wandb
import warnings
from dataclasses import dataclass
from pathlib import Path
import argparse
from types import SimpleNamespace

import argbind
import torch
import torchaudio
from torch.nn import functional as F
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter
from transformers import HubertModel, Wav2Vec2Processor

import dac

warnings.filterwarnings("ignore", category=UserWarning)

def namespace_to_dict(ns):
    if isinstance(ns, (SimpleNamespace, argparse.Namespace)):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
DiscoDAC = argbind.bind(dac.model.DiscoDAC)
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DiscoDAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss
    align_loss: losses.SemanticEmbeddingLoss

    train_data: AudioDataset
    val_data: AudioDataset

    step: int = 0


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac").exists():
            generator, g_extra = DAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = DiscoDAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    print(generator)
    print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    # if "tracker.pth" in g_extra:
    #     tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        train_data=train_data,
        val_data=val_data,
    )

@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }


@timer()
def train_loop(state, batch, accel, lambdas, hubert_model, hubert_processor):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
    
    with torch.no_grad():
        inputs = hubert_processor([x for x in out["audio"]], sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.cuda()
        outputs = hubert_model(input_values)
        hubert_emb = outputs.last_hidden_state  # Shape: [B, T, D]

    # ======== update discriminator  ======== 
    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    #  ======== update generator ========= 
    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        output["adv/gen_loss"], output["adv/feat_loss"] = state.gan_loss.generator_loss(recons, signal)
        output["vq/commit_loss_acs"] = out["vq/commit_loss_acs"]
        output["vq/commit_loss_sem"] = out["vq/commit_loss_sem"]
        output["vq/codebook_loss_acs"] = out["vq/codebook_loss_acs"]
        output["vq/codebook_loss_sem"] = out["vq/codebook_loss_sem"]
        output["align/loss"] = state.align_loss(out["e_sem"], hubert_emb)
        output["loss"] = sum([v * output[k] for k, v in lambdas.items() if k in output])

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path, tracker):
    metadata = {"logs": tracker.history}

    print(f"[Checkpoint] Saving to {str(Path('.').absolute())}")
    tags = ["latest"]
    if tracker.is_best("val","mel/loss"):
        print(f"[Checkpoint] Best generator so far")
        tags.append("best")
    if state.step in save_iters:
        tags.append(f"{state.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker-pth": tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


@torch.no_grad()
def save_samples(state, val_idx, save_path):
    print("--->Saving audio samples to disk--->")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            audio_sample = v[nb].cpu()

            filename = os.path.join(
                save_path,
                f"{k}_step{state.step}_sample{nb}.wav" if k == "recons" else f"{k}_sample{nb}.wav"
            )
            torchaudio.save(filename, audio_sample.audio_data[0], audio_sample.sample_rate)


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
        'align/loss': 1.0
    },
):

    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    
    state = load(args, accel, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    hubert_model = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960", 
        torch_dtype=torch.float16, 
        attn_implementation="flash_attention_2")
    hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
    

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    tracker = Tracker(rank=accel.local_rank)
    global val_loop, validate, save_samples, checkpoint
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    for step, batch in enumerate(train_dataloader, start=state.step):
        metrics = train_loop(state, batch, accel, lambdas, hubert_model)
        metrics = {f"train/{k}": (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
        if accel.local_rank == 0 and step % 50 ==0:
            wandb.log(metrics, step=state.step)

        last_iter = (
            step == num_iters - 1 if num_iters is not None else False
        )
        if step % sample_freq == 0 or last_iter:
            save_samples(state, val_idx, save_path)

        if step % valid_freq == 0 or last_iter:
            validate(state, val_dataloader, accel)
            if accel.local_rank == 0:
                val_metrics = {
                    f"val/{k}": float(v())  # returns a float, guaranteed
                    for k, v in tracker.metrics["val"]["mean"].items() if v is not None
                    }
                wandb.log(val_metrics, step=state.step)
            checkpoint(state, save_iters, save_path, tracker)
            print(f"Val done")

        if last_iter:
            break

        state.step +=1


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank == 0:
                wandb.login()
                os.environ["WANDB_MODE"] = "online"
                os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
                os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
                wandb.init(project="audio_codec", config=namespace_to_dict(args), name='baseline')

            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
