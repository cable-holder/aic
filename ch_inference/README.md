# ch_inference

ROS policy package for running a trained LeRobot VLA checkpoint in AIC.

The policy expects the postprocessed scheme:

- `observation.state`: 25D TCP position, TCP RPY, TCP velocity, TCP error, joints
- `action`: 6D absolute Cartesian TCP target `[x, y, z, roll, pitch, yaw]`
- images: left, center, right camera keys used by `ch_data_process/postprocess_scheme.py`

## Train

Use the VLA dataset produced by `ch_data_process/postprocess_scheme.py`:

```bash
/home/yl/aic_results/ch_milestones_vla
```

Install the local LeRobot checkout in your training environment:

```bash
cd /home/yl/Dev/lerobot
python -m pip install -e ".[training,smolvla]"
```

SmolVLA needs the camera rename map because the base checkpoint expects
`camera1`, `camera2`, and `camera3`:

```bash
cd /home/yl/Dev/lerobot

PYTHONPATH=/home/yl/Dev/lerobot/src lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr=3e-5 \
  --dataset.repo_id=local/ch_milestones_vla \
  --dataset.root=/home/yl/aic_results/ch_milestones_vla \
  --rename_map='{"observation.images.left_camera":"observation.images.camera1","observation.images.center_camera":"observation.images.camera2","observation.images.right_camera":"observation.images.camera3"}' \
  --output_dir=/home/yl/aic_results/train/aic_smolvla \
  --job_name=aic_smolvla \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=50 \
  --num_workers=4 \
  --wandb.enable=false
```

Pi05 should use the dataset camera names directly, so do not pass
`--rename_map`:

```bash
cd /home/yl/Dev/lerobot
python -m pip install -e ".[training,pi]"

PYTHONPATH=/home/yl/Dev/lerobot/src lerobot-train \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --dataset.repo_id=local/ch_milestones_vla \
  --dataset.root=/home/yl/aic_results/ch_milestones_vla \
  --output_dir=/home/yl/aic_results/train/aic_pi05 \
  --job_name=aic_pi05 \
  --policy.device=cuda \
  --batch_size=1 \
  --steps=60000 \
  --save_freq=1000 \
  --log_freq=50 \
  --num_workers=0 \
  --wandb.enable=false
```

The inference checkpoint path is:

```bash
/home/yl/aic_results/train/<run>/checkpoints/<step>/pretrained_model
```

## Run

Run it with:

```bash
CKPT=$(find /home/yl/aic_results/train/aic_pi05_small/checkpoints -maxdepth 3 -name policy_preprocessor.json -printf '%h\n' | sort | tail -1)

pixi run ros2 launch ch_inference ch_inference.launch.py \
  vla_policy_path:=$CKPT
```

Pi05 loads slowly, so inference waits across configure and activate by default:
`vla_configure_wait_seconds:=45` and `vla_activate_wait_seconds:=55`.

To use the policy directly from `aic_model`, set:

```bash
policy:=ch_inference.policies.VLAPolicy
vla_policy_path:=/home/yl/aic_results/train/<run>/checkpoints/<step>/pretrained_model
```
