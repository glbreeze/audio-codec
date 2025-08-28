import os
import argparse
import shutil

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

def collect_stems(musdb_root: str, output_dir: str, subset: str, stem: str) -> int:

    subset_dir = os.path.join(musdb_root, subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    copied = 0
    for track_folder in sorted(os.listdir(subset_dir)):
        track_path = os.path.join(subset_dir, track_folder)
        if not os.path.isdir(track_path):
            continue

        for fname in os.listdir(track_path):
            src = os.path.join(track_path, fname)
            if not os.path.isfile(src):
                continue
            # match by filename and extension
            if stem.lower() in fname.lower() and fname.lower().endswith(AUDIO_EXTS):
                dst_dir = os.path.join(output_dir, subset, track_folder)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, fname)
                shutil.copy2(src, dst)  # copy metadata too
                copied += 1
                print(f"Copied {subset}/{track_folder}/{fname} -> {dst}")
    return copied

def main():
    parser = argparse.ArgumentParser(description="Extract specific stems from MUSDB.")
    parser.add_argument("--musdb_root", type=str, default='/vast/lg154/datasets/musdb')
    parser.add_argument("--output_dir", type=str, default='/vast/lg154/datasets/')
    parser.add_argument("--stem", type=str, default="vocals", help="Substring to match in filenames (e.g., 'vocals', 'bass').")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, f'musdb_{args.stem}')

    total = 0
    for subset in ['train', 'test']:
        n = collect_stems(args.musdb_root, args.output_dir, subset, args.stem)
        print(f"[{subset}] copied {n} files matching '{args.stem}'.")
        total += n
    print(f"Done. Total copied: {total}")

if __name__ == "__main__":
    main()