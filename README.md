# DY-LUT

Paper: https://arxiv.org/abs/2607.22801

DY-LUT is an underwater image enhancement model based on depth-aware lookup tables
in YCbCr space.

<video src="demo/videos/original_vs_dylut.mp4" controls></video>

The released checkpoint is `checkpoints/paper_release/best_model.pth`.

## Installation

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

Run inference with the released checkpoint:

```bash
python3 inference.py \
  --checkpoint checkpoints/paper_release/best_model.pth \
  --input /path/to/input_images \
  --depth /path/to/depth_maps \
  --output results/paper_release \
  --no_auto_scale
```

Input images and depth maps are matched by filename stem. Remove `--no_auto_scale`
to enable adaptive downsampling for high-resolution images.

## Data preparation

Prepare paired images, depth maps, and gradient maps for training:

```bash
python3 prepare_uieb_finetune.py \
  --raw_dir /path/to/raw_images \
  --reference_dir /path/to/reference_images \
  --output_dir /path/to/prepared_data \
  --depth_anything_root /path/to/Depth-Anything-V2 \
  --depth_checkpoint /path/to/depth_anything_v2_vitl.pth
```

## Training

Train with the default configuration:

```bash
python3 train_lut.py \
  --config uieb_finetune \
  --data_root /path/to/prepared_data \
  --out_dir runs/paper_release
```

The data, checkpoint, and log roots can also be set with `DYLUT_DATA_ROOT`,
`DYLUT_CHECKPOINT_ROOT`, and `DYLUT_LOG_ROOT`.
