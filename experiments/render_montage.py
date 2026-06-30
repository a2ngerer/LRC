"""Tile per-model rollout videos into one side-by-side comparison grid mp4.

Streams frames from each source mp4 (so memory stays bounded) and writes a
grid video where every cell is one model's cheetah running in lockstep -- the
single clearest artifact for "how do the neuron types compare at driving the
body". Cells are ordered classical-first, then the continuous-time / liquid
thesis cells.

    uv run python experiments/render_montage.py \
        --video-dir results/mujoco_local/videos --out results/mujoco_local/_montage.mp4
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np

ORDER = ["mlp", "lstm", "gru", "ctrnn", "ltc", "lrc", "cfc"]


def main():
    p = argparse.ArgumentParser(description="Tile model videos into a comparison grid")
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed-tag", default="s0", help="seed suffix of the source files")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    files = []
    for m in ORDER:
        f = os.path.join(args.video_dir, f"{m}_{args.seed_tag}.mp4")
        if os.path.exists(f):
            files.append((m, f))
    if not files:
        raise SystemExit(f"no model videos ({'/'.join(ORDER)}) in {args.video_dir}")
    print(f"montage of {len(files)} models: {[m for m, _ in files]}")

    readers = [imageio.get_reader(f) for _, f in files]
    # frame geometry from the first reader
    first = readers[0].get_data(0)
    H, W = first.shape[:2]
    cols = args.cols
    rows = (len(files) + cols - 1) // cols
    n_cells = rows * cols

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)

    iters = [iter(r) for r in readers]
    t = 0
    while True:
        frames = []
        done = False
        for it in iters:
            try:
                frames.append(next(it))
            except StopIteration:
                done = True
                break
        if done or len(frames) < len(iters):
            break
        # pad to a full grid with black cells
        while len(frames) < n_cells:
            frames.append(np.zeros((H, W, 3), np.uint8))
        grid_rows = []
        for r in range(rows):
            row = np.concatenate(frames[r * cols:(r + 1) * cols], axis=1)
            grid_rows.append(row)
        grid = np.concatenate(grid_rows, axis=0)
        writer.append_data(grid)
        t += 1

    for r in readers:
        r.close()
    writer.close()
    print(f"wrote {args.out} ({t} frames, {rows}x{cols} grid)")


if __name__ == "__main__":
    main()
