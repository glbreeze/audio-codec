python scripts/wave_eval.py \
  --org_dir ../datasets/LibriSpeech/test-clean \
  --cfg_yml conf/disco_base.yml \
  --ckpt runs/libri3_film0_align1_hbt9/best \
  --sr 16000 --cb 3


python scripts/wave_eval.py \
  --org_dir ../datasets/LibriSpeech/test-clean \
  --cfg_yml conf/disco_base.yml \
  --ckpt runs/libri3_film03_align1_hbt9/best \
  --sr 16000 --cb 3



python scripts/wave_eval.py \
  --org_dir ../datasets/LibriSpeech/test-clean \
  --cfg_yml conf/libri_base.yml \
  --ckpt runs/libri4_baseline/best \
  --model_type DAC \
  --sr 16000 --cb 4


# =============== eval decoded audio 

  python scripts/wave_eval_file.py \
    --args.load conf/disco_base.yml \
    --DiscoDAC.n_codebooks 3 \
    --DiscoDAC.film_layer_idx '0' \
    --save_path runs/libri3_film0_align1_hbt9/ \
    --val_batch_size 1 \
    --val/AudioDataset.duration 0


 python scripts/wave_eval_file.py \
    --args.load conf/disco_base.yml \
    --DiscoDAC.n_codebooks 3 \
    --DiscoDAC.film_layer_idx '03' \
    --save_path runs/libri3_film03_align1_hbt9/ \
    --val_batch_size 1 \
    --val/AudioDataset.duration 0


 python scripts/wave_eval_file.py \
    --args.load conf/disco_base.yml \
    --DiscoDAC.n_codebooks 3 \
    --DiscoDAC.film_layer_idx '03' \
    --save_path runs/libri3_film03_align1_hbt9/ \
    --val_batch_size 1 \
    --split train \
    --val/AudioDataset.duration 0 \
    --train/AudioDataset.duration 0


  python scripts/wave_eval_file.py \
    --args.load conf/dac_base.yml \
    --DAC.n_codebooks 4 \
    --save_path runs/libri4_baseline\
    --val_batch_size 1 \
    --val/AudioDataset.duration 0
  
# =============== eval latent code


  python scripts/wav_eval_code.py \
    --data_root runs/libri3_film03_align1_hbt9/ld/ \
    --input_type discrete 



