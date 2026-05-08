# ch_data_process

Standalone data processing tools for `ch_milestones` datasets.

Merge four per-stage LeRobot datasets into complete-motion episodes:

```bash
pixi run ch-data-process \
  --input-root ~/aic_results/ch_milestones_lerobot \
  --output-root ~/aic_results/ch_milestones_complete
```

Or with a regular virtual environment:

```bash
python ch_data_process/phase_merge.py \
  --input-root ~/aic_results/ch_milestones_lerobot \
  --output-root ~/aic_results/ch_milestones_complete \
  --overwrite
```

Smoke-test one episode first:

```bash
python ch_data_process/phase_merge.py \
  --input-root ~/aic_results/ch_milestones_lerobot \
  --output-root /tmp/ch_milestones_complete_one \
  --episodes 0:1 \
  --overwrite
```

The input root must contain `approach/`, `coarse_align/`, `fine_align/`, and
`insert/`. Episode `i` from each phase becomes one complete episode in the
output dataset.

You can also pass the four phase datasets explicitly:

```bash
pixi run ch-data-process \
  --approach ~/data/approach \
  --coarse-align ~/data/coarse_align \
  --fine-align ~/data/fine_align \
  --insert ~/data/insert \
  --output-root ~/data/complete
```

By default, the output task prompt strips the phase suffix. Use
`--task-mode keep` to preserve per-frame phase prompts, or `--task-prompt` to
set one fixed prompt for every output frame.

The merger shows a progress bar and skips corrupted episodes by default. An
episode is skipped if `task.message_json` changes inside the appended rollout or
if TCP position jumps by more than `--max-position-jump` between adjacent
frames. The default jump threshold is `0.10` meters.

The output forces separate LeRobot data/video files per merged episode. Videos
are still organized by camera stream:

```text
videos/observation.images.left_camera/chunk-000/file-000.mp4
videos/observation.images.left_camera/chunk-000/file-001.mp4
videos/observation.images.center_camera/chunk-000/file-000.mp4
videos/observation.images.center_camera/chunk-000/file-001.mp4
```

Check inter-frame image similarity for a merged or phase dataset:

```bash
python ch_data_process/frame_similarity.py \
  --root ~/aic_results/ch_milestones_complete \
  --threshold 0.20
```

or:

```bash
pixi run ch-frame-similarity \
  --root ~/aic_results/ch_milestones_lerobot/approach \
  --threshold 0.20
```

The score is normalized mean absolute pixel difference in `[0, 1]`, computed
between adjacent frames within the same episode. Use `--episodes 0:10`,
`--cameras observation.images.left_camera`, or `--report report.csv` to narrow
or save the check.
