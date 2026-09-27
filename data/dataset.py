#!/usr/bin/env python3
"""
RA-HUPE蒸馏数据集加载器
支持混合1600+890数据，加权采样
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class RAHUPEDistillDataset(Dataset):
    """
    RA-HUPE蒸馏数据集
    每个样本包含：input_rgb, input_grad, input_depth, target_enhanced
    """
    def __init__(self, config, dataset_config, split='train', transform=None):
        """
        Args:
            config: 总配置对象
            dataset_config: 数据集特定配置 (1600或890)
            split: 'train' or 'val'
            transform: 数据增强
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # 数据路径
        self.rgb_dir = Path(dataset_config['input_rgb'])
        self.grad_dir = Path(dataset_config['input_grad'])
        self.depth_dir = Path(dataset_config['input_depth'])
        self.target_dir = Path(dataset_config['target'])
        
        # 获取所有文件名
        all_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])

        # 划分训练/验证集：支持文件列表模式或比例模式
        use_file_split = getattr(config, 'use_file_split', False)
        if use_file_split:
            train_list_file = getattr(config, 'train_list_file', None)
            test_list_file  = getattr(config, 'test_list_file',  None)
            if train_list_file and test_list_file:
                with open(train_list_file) as f:
                    train_set = set(f.read().splitlines())
                with open(test_list_file) as f:
                    test_set = set(f.read().splitlines())
                if split == 'train':
                    self.rgb_files = [fn for fn in all_files if fn in train_set]
                else:
                    self.rgb_files = [fn for fn in all_files if fn in test_set]
            else:
                raise ValueError("use_file_split=True 但未提供 train_list_file / test_list_file")
        else:
            random.seed(config.seed)
            shuffled = list(all_files)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * config.train_split)
            if split == 'train':
                self.rgb_files = shuffled[:split_idx]
            else:
                self.rgb_files = shuffled[split_idx:]

        # Debug模式
        if config.debug:
            self.rgb_files = self.rgb_files[:config.debug_samples]
        
        print(f"  {dataset_config['name']} - {split}: {len(self.rgb_files)} samples")
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb: [3, H, W] ∈ [0, 1]
            grad: [1, H, W] ∈ [0, 1]
            depth: [1, H, W] ∈ [0, 1]
            target: [3, H, W] ∈ [0, 1]
        """
        filename = self.rgb_files[idx]
        
        # 加载图像
        rgb_path = self.rgb_dir / filename
        grad_path = self.grad_dir / filename
        depth_path = self.depth_dir / filename
        target_path = self.target_dir / filename
        
        # RGB (处理RGBA格式)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)  # 读取所有通道
        if rgb is None:
            raise ValueError(f"Failed to load RGB: {rgb_path}")
        
        # 处理RGBA -> RGB
        if rgb.shape[2] == 4:  # RGBA
            # 使用Alpha通道做混合（如果有背景的话）
            alpha = rgb[:, :, 3:4] / 255.0
            rgb = rgb[:, :, :3]
            # 可选：如果需要白色背景
            # rgb = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # BGR->RGB
        
        # 梯度（灰度）
        grad = cv2.imread(str(grad_path), cv2.IMREAD_GRAYSCALE)
        if grad is None:
            raise ValueError(f"Failed to load Grad: {grad_path}")
        grad = grad[..., np.newaxis]  # [H, W] -> [H, W, 1]
        
        # 深度（灰度）
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Failed to load Depth: {depth_path}")
        depth = depth[..., np.newaxis]
        
        # 目标增强图
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if target is None:
            raise ValueError(f"Failed to load Target: {target_path}")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # 全局统计量：在增强/裁剪之前，用已加载的完整图计算（避免重复读盘 3 次）
        rgb_full = rgb.astype(np.float32) / 255.0           # [H,W,3] RGB uint8 → float
        depth_full = depth[..., 0].astype(np.float32) / 255.0
        grad_full = grad[..., 0].astype(np.float32) / 255.0
        r, g_ch, b = rgb_full[:, :, 0], rgb_full[:, :, 1], rgb_full[:, :, 2]
        y_full  = 0.299 * r + 0.587 * g_ch + 0.114 * b
        cb_full = 0.5 - 0.168736 * r - 0.331264 * g_ch + 0.5 * b
        cr_full = 0.5 + 0.5 * r - 0.418688 * g_ch - 0.081312 * b
        global_stats = torch.tensor([
            y_full.mean(), cb_full.mean(), cr_full.mean(),
            depth_full.mean(), grad_full.mean()
        ], dtype=torch.float32)  # [5]

        # 数据增强/预处理
        if self.transform:
            rgb, grad, depth, target = self.transform(rgb, grad, depth, target)

        # 确保grad和depth是3维 [H, W, 1]
        if grad.ndim == 2:
            grad = grad[..., np.newaxis]
        if depth.ndim == 2:
            depth = depth[..., np.newaxis]

        # 转Tensor并归一化
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        grad = torch.from_numpy(grad.transpose(2, 0, 1)).float() / 255.0
        depth = torch.from_numpy(depth.transpose(2, 0, 1)).float() / 255.0
        target = torch.from_numpy(target.transpose(2, 0, 1)).float() / 255.0
        
        return rgb, grad, depth, target, global_stats


class Transform:
    """数据增强"""
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
    
    def __call__(self, rgb, grad, depth, target):
        if self.split == 'train':
            return self._train_transform(rgb, grad, depth, target)
        else:
            return self._val_transform(rgb, grad, depth, target)
    
    def _train_transform(self, rgb, grad, depth, target):
        """
        训练时数据增强
        增强策略：参考LUT-Fuse + 添加更多变换
        """
        config = self.config
        
        # 确保grad和depth是2维用于resize
        if grad.ndim == 3:
            grad = grad.squeeze(-1)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        
        # 1. Resize到统一尺寸
        h, w = rgb.shape[:2]
        if h != config.train_resize or w != config.train_resize:
            rgb = cv2.resize(rgb, (config.train_resize, config.train_resize))
            grad = cv2.resize(grad, (config.train_resize, config.train_resize))
            depth = cv2.resize(depth, (config.train_resize, config.train_resize))
            target = cv2.resize(target, (config.train_resize, config.train_resize))
        
        # 2. 随机裁剪
        # Optional random-scale crop, resized to the configured training crop size.
        crop_size = config.train_crop_size
        rand_scale = getattr(config, 'random_scale_crop', False)
        if rand_scale:
            lo, hi = getattr(config, 'scale_crop_range', (crop_size, config.train_resize))
            h, w = rgb.shape[:2]
            hi = min(hi, h, w)
            lo = min(lo, hi)
            actual_crop = random.randint(lo, hi) if hi > lo else hi
        else:
            actual_crop = crop_size

        h, w = rgb.shape[:2]
        if h > actual_crop and w > actual_crop:
            top = random.randint(0, h - actual_crop)
            left = random.randint(0, w - actual_crop)

            rgb = rgb[top:top+actual_crop, left:left+actual_crop]
            grad = grad[top:top+actual_crop, left:left+actual_crop]
            depth = depth[top:top+actual_crop, left:left+actual_crop]
            target = target[top:top+actual_crop, left:left+actual_crop]

        if rand_scale and actual_crop != crop_size:
            rgb = cv2.resize(rgb, (crop_size, crop_size))
            grad = cv2.resize(grad, (crop_size, crop_size))
            depth = cv2.resize(depth, (crop_size, crop_size))
            target = cv2.resize(target, (crop_size, crop_size))
        
        # 3. 随机水平翻转
        if random.random() < config.horizontal_flip_prob:
            rgb = cv2.flip(rgb, 1)
            grad = cv2.flip(grad, 1)
            depth = cv2.flip(depth, 1)
            target = cv2.flip(target, 1)
        
        # 4. 随机垂直翻转
        if random.random() < config.vertical_flip_prob:
            rgb = cv2.flip(rgb, 0)
            grad = cv2.flip(grad, 0)
            depth = cv2.flip(depth, 0)
            target = cv2.flip(target, 0)
        
        # 5. 随机旋转（小角度）
        if random.random() < config.rotation_prob:
            angle = random.uniform(-10, 10)
            h, w = rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            grad = cv2.warpAffine(grad, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            depth = cv2.warpAffine(depth, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            target = cv2.warpAffine(target, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Optional depth corruption for reliability-gate training.
        corrupt_prob = getattr(self.config, 'depth_corrupt_prob', 0.0)
        if corrupt_prob > 0 and random.random() < corrupt_prob:
            depth = self._corrupt_depth(depth)

        # 恢复单通道维度 [H, W] -> [H, W, 1]
        grad = grad[..., np.newaxis]
        depth = depth[..., np.newaxis]
        
        return rgb, grad, depth, target

    @staticmethod
    def _corrupt_depth(depth):
        """depth: [H, W] uint8 → 腐蚀后的 [H, W] uint8"""
        kind = random.choice(['noise', 'blur', 'constant', 'gamma'])
        d = depth.astype(np.float32)
        if kind == 'noise':
            sigma = random.uniform(5, 30)
            d = d + np.random.randn(*d.shape) * sigma
        elif kind == 'blur':
            k = random.choice([9, 17, 33])
            d = cv2.GaussianBlur(d, (k, k), 0)
        elif kind == 'constant':
            d = np.full_like(d, random.uniform(60, 200))
        elif kind == 'gamma':
            gamma = random.uniform(0.4, 2.5)
            d = ((d / 255.0) ** gamma) * 255.0
        return np.clip(d, 0, 255).astype(np.uint8)

    def _val_transform(self, rgb, grad, depth, target):
        """验证时只resize"""
        config = self.config
        
        # 确保grad和depth是2维用于resize
        if grad.ndim == 3:
            grad = grad.squeeze(-1)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        
        rgb = cv2.resize(rgb, (config.val_resize, config.val_resize))
        grad = cv2.resize(grad, (config.val_resize, config.val_resize))
        depth = cv2.resize(depth, (config.val_resize, config.val_resize))
        target = cv2.resize(target, (config.val_resize, config.val_resize))
        
        # 恢复单通道维度 [H, W] -> [H, W, 1]
        grad = grad[..., np.newaxis]
        depth = depth[..., np.newaxis]
        
        return rgb, grad, depth, target


def create_mixed_dataloader(config, split='train'):
    """
    创建数据加载器（支持选择性使用数据集）
    可以使用: 1600 + 890, 只用1600, 或只用890
    """
    # 创建数据集列表
    transform = Transform(config, split)
    datasets = []
    dataset_weights = []
    
    # 根据配置选择数据集
    if config.use_dataset_1600:
        dataset_1600 = RAHUPEDistillDataset(
            config, config.dataset_1600, split, transform
        )
        datasets.append(dataset_1600)
        dataset_weights.append(config.dataset_1600['weight'])
    
    if config.use_dataset_890:
        dataset_890 = RAHUPEDistillDataset(
            config, config.dataset_890, split, transform
        )
        datasets.append(dataset_890)
        dataset_weights.append(config.dataset_890['weight'])
    
    # 检查至少有一个数据集
    if len(datasets) == 0:
        raise ValueError("至少需要选择一个数据集！")
    
    # 混合数据集
    from torch.utils.data import ConcatDataset
    if len(datasets) == 1:
        mixed_dataset = datasets[0]
        print(f"  使用单个数据集: {len(mixed_dataset)} samples")
    else:
        mixed_dataset = ConcatDataset(datasets)
        print(f"  使用混合数据集: {len(mixed_dataset)} samples")
    
    # 加权采样器（训练时使用，仅当有多个数据集时）
    if split == 'train' and len(datasets) > 1:
        # 计算每个样本的权重
        weights = []
        for dataset, weight in zip(datasets, dataset_weights):
            weights += [weight] * len(dataset)
        
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights, 
            num_samples=len(mixed_dataset),
            replacement=True
        )
        
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
    elif split == 'train' and len(datasets) == 1:
        # 单个数据集，直接shuffle
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        # 验证时顺序采样
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=1,  # 验证时batch=1
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据加载
    import sys
    sys.path.append('..')
    from configs.config import Config
    
    config = Config()
    config.debug = True
    
    print("创建训练集...")
    train_loader = create_mixed_dataloader(config, split='train')
    print(f"训练批次数: {len(train_loader)}")
    
    print("\n创建验证集...")
    val_loader = create_mixed_dataloader(config, split='val')
    print(f"验证批次数: {len(val_loader)}")
    
    print("\n加载一个batch测试...")
    for rgb, grad, depth, target in train_loader:
        print(f"  RGB: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"  Grad: {grad.shape}, range: [{grad.min():.3f}, {grad.max():.3f}]")
        print(f"  Depth: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"  Target: {target.shape}, range: [{target.min():.3f}, {target.max():.3f}]")
        break
