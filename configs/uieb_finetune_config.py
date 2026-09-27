#!/usr/bin/env python3
"""Portable UIEB configuration for the paper-release model."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("DYLUT_DATA_ROOT", PROJECT_ROOT / "datasets" / "UIEB")
).expanduser()
CHECKPOINT_ROOT = Path(
    os.environ.get("DYLUT_CHECKPOINT_ROOT", PROJECT_ROOT / "runs")
).expanduser()
LOG_ROOT = Path(os.environ.get("DYLUT_LOG_ROOT", CHECKPOINT_ROOT / "logs")).expanduser()


class Config:
    # Data
    data_root = str(DATA_ROOT)
    depth_source = "mono_dav2"
    dataset_1600 = {
        "input_rgb": str(DATA_ROOT / "input"),
        "input_depth": str(DATA_ROOT / "depth"),
        "input_grad": str(DATA_ROOT / "grad"),
        "target": str(DATA_ROOT / "target"),
        "name": "uieb_train",
        "weight": 1.0,
    }
    dataset_890 = dict(dataset_1600, name="uieb_unused", weight=0.0)
    train_list_file = str(DATA_ROOT / "train_list.txt")
    test_list_file = str(DATA_ROOT / "test_list.txt")
    train_split = 0.0
    use_file_split = True
    use_dataset_1600 = True
    use_dataset_890 = False

    # Model
    lut_mode = "full"
    num_luts = 3
    lut_bins = 25
    lut_output_channels = 3
    use_dual_encoder = True
    context_hidden_dim = 32
    context_num_layers = 3
    use_local_refine = True
    refine_type = "local"
    refine_hidden_dim = 16
    refine_num_layers = 3

    # Optimization
    batch_size = 4
    num_workers = 4
    num_epochs = 800
    lr_lut = 2e-4
    lr_context = 5e-4
    weight_decay = 1e-5
    grad_clip = 1.0
    warmup_epochs = 20
    lr_milestones = [400, 600]
    lr_gamma = 0.3

    # Loss
    loss_l1_weight = 1.0
    loss_l2_weight = 0.0
    loss_ssim_weight = 1.0
    loss_chroma_weight = 0.0
    loss_tv_weight = 5e-5
    loss_mn_weight = 2.0
    loss_contrast_weight = 0.0
    loss_vgg_weight = 0.1
    loss_fft_weight = 0.0
    loss_grad_weight = 0.05
    loss_cb_weight = 1.5
    loss_cr_weight = 1.5

    # Augmentation
    train_crop_size = 256
    train_resize = 512
    horizontal_flip_prob = 0.5
    vertical_flip_prob = 0.0
    rotation_prob = 0.0
    color_jitter_prob = 0.0
    val_resize = 512

    # Runtime
    pretrained_model = None
    pretrained_context_gen = None
    checkpoint_dir = str(CHECKPOINT_ROOT / "paper_release")
    log_dir = str(LOG_ROOT)
    save_interval = 10
    early_stopping_patience = 250
    min_delta = 1e-5
    device = "cuda"
    mixed_precision = False
    seed = 42
    debug = False
    debug_samples = 20
