import musdb
import numpy as np
import soundfile as sf
import os


def main(musdb_root, output_dir, subset='train'):
    # Load MUSDB tracks
    mus = musdb.DB(root=musdb_root, is_wav=True, subsets=subset)  # use 'is_wav=True' if using musdb18hq

    for track in mus:
        print(f"Processing {track.name}...")

        # Construct accompaniment: bass + drums + other
        acc = (
            track.targets['bass'].audio +
            track.targets['drums'].audio +
            track.targets['other'].audio
        )

        track_dir = os.path.join(output_dir, subset, track.name)
        os.makedirs(track_dir, exist_ok=True)

        # Save the accompaniment audio
        output_path = os.path.join(track_dir, "accompaniment.wav")
        sf.write(output_path, acc, track.rate)

    print(f"✅ {subset} data done. All accompaniment-only tracks saved.")


if __name__ == '__main__':
    musdb_root = '/vast/lg154/datasets/musdb'
    output_dir = '/vast/lg154/datasets/musdb_instrumental'
    os.makedirs(output_dir, exist_ok=True)

    main(musdb_root, output_dir, 'train')
    main(musdb_root, output_dir, 'test')
