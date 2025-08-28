import os, random, shutil
import torchaudio

librispeech_src = "/vast/lg154/datasets/LibriSpeech/train-clean-100"
librispeech_out = "/vast/lg154/datasets/LibriSpeech/train_clean-10"
target_seconds = 10 * 3600  # 10 hours

all_files = []
# Collect all flac files
for root, _, files in os.walk(librispeech_src):
    for f in files:
        if f.endswith(".flac"):
            all_files.append(os.path.join(root, f))

# Compute durations
file_durations = []
for f in all_files:
    info = torchaudio.info(f)
    dur = info.num_frames / info.sample_rate
    file_durations.append((f, dur))

# Shuffle and pick files until ~10h
random.shuffle(file_durations)
print(f"---- total duration in train-clean-100 is {sum([d for _, d in file_durations])/3600:.2f}h ----")

subset, total = [], 0.0
for f, dur in file_durations:
    if total + dur > target_seconds:
        break
    subset.append(f)
    total += dur

print(f"Selected {len(subset)} files, total duration {total/3600:.2f}h")

# Copy files while preserving relative directory structure
for f in subset:
    rel_path = os.path.relpath(f, librispeech_src)
    out_f = os.path.join(librispeech_out, rel_path)
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    shutil.copyfile(f, out_f)
    print(f"Copied {rel_path}")