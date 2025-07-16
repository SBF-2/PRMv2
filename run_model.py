#!/usr/bin/env python3
"""
PRM (Predictive Representation Model) 训练脚本
支持多个H5文件、分布式训练、简洁日志

使用方法:
1. 单GPU训练:
   python train_prm.py --data_dir /path/to/h5/files --config standard

2. 多GPU分布式训练:
   torchrun --nproc_per_node=4 train_prm.py --data_dir /path/to/h5/files --config standard --distributed

3. 指定GPU数量:
   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_prm.py --data_dir /path/to/h5/files --config large --distributed

输出格式:
- 训练/验证进度条显示实时指标
- 每epoch结束后显示格式化的总结信息
- *BEST* 标记表示当前最佳模型
"""

import os
import sys
import argparse
import logging
import json
import time
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import h5py
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# 导入用户本地的编码器（需要用户根据实际路径调整）
from model import ImgEncoder, ActionEncoder  # 用户需要根据实际情况调整导入路径
# 导入PRM模型
from model.PRM import (
    OptimizedEndToEndModel, create_optimized_model, 
    create_training_manager, TrainingManager
)

class H5SequenceDataset(Dataset):
    """从 H5 文件加载序列数据的数据集"""
    
    def __init__(self, h5_filepath: str, normalize_obs: bool = True):
        """
        Args:
            h5_filepath: H5 文件路径
            normalize_obs: 是否将观察归一化到 [0,1] (uint8 -> float32 / 255)
        """
        self.h5_filepath = h5_filepath
        self.normalize_obs = normalize_obs
        self.sequence_keys = []
        
        # 获取所有序列的键名
        with h5py.File(h5_filepath, 'r') as f:
            self.sequence_keys = [key for key in f.keys() if key.startswith('sequence_')]
            # 按数字顺序排序
            self.sequence_keys.sort(key=lambda x: int(x.split('_')[1]))
        
        print(f"从 {h5_filepath} 加载了 {len(self.sequence_keys)} 个序列")
    
    def __len__(self) -> int:
        return len(self.sequence_keys)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个序列"""
        sequence_key = self.sequence_keys[idx]
        
        with h5py.File(self.h5_filepath, 'r') as f:
            seq_group = f[sequence_key]
            
            # 读取数据
            observations = seq_group['observations'][:]  # (33, 4, 84, 84)
            actions = seq_group['actions'][:]            # (32,)
            rewards = seq_group['rewards'][:]            # (32,)
            terminations = seq_group['terminations'][:]   # (32,)
            truncations = seq_group['truncations'][:]     # (32,)
        
        # 转换为 torch tensors
        sample = {
            'observations': torch.from_numpy(observations),
            'actions': torch.from_numpy(actions).long(),
            'rewards': torch.from_numpy(rewards).float(),
            'terminations': torch.from_numpy(terminations),
            'truncations': torch.from_numpy(truncations),
            'sequence_idx': torch.tensor(idx)
        }
        
        # 归一化观察 (uint8 -> float32 / 255)
        if self.normalize_obs:
            sample['observations'] = sample['observations'].float() / 255.0
        else:
            sample['observations'] = sample['observations'].float()
        
        return sample

class MultiFileH5Dataset(Dataset):
    """处理文件夹内多个H5文件的数据集包装器"""
    
    def __init__(self, data_dir: str, h5_pattern: str = "*.h5", normalize_obs: bool = True):
        """
        Args:
            data_dir: 包含H5文件的文件夹路径
            h5_pattern: H5文件匹配模式 (例如: "*.h5", "train_*.h5")
            normalize_obs: 是否归一化观察
        """
        self.data_dir = data_dir
        self.h5_pattern = h5_pattern
        self.normalize_obs = normalize_obs
        self.datasets = []
        self.cumulative_sizes = []
        self.total_size = 0
        
        # 检查文件夹是否存在
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"数据文件夹不存在: {data_dir}")
        
        # 查找所有匹配的H5文件
        search_pattern = os.path.join(data_dir, h5_pattern)
        h5_filepaths = sorted(glob.glob(search_pattern))
        
        if not h5_filepaths:
            raise FileNotFoundError(f"在文件夹 {data_dir} 中没有找到匹配 '{h5_pattern}' 的H5文件")
        
        print(f"在文件夹 {data_dir} 中找到 {len(h5_filepaths)} 个H5文件")
        
        # 加载每个H5文件
        for i, filepath in enumerate(h5_filepaths):
            try:
                dataset = H5SequenceDataset(filepath, normalize_obs)
                self.datasets.append(dataset)
                self.total_size += len(dataset)
                self.cumulative_sizes.append(self.total_size)
                
                # 显示文件名而不是完整路径
                filename = os.path.basename(filepath)
                print(f"  文件 {i+1:2d}: {filename:30s} -> {len(dataset):6d} 个序列")
                
            except Exception as e:
                filename = os.path.basename(filepath)
                print(f"错误: 无法加载 {filename}: {e}")
                continue
        
        if not self.datasets:
            raise ValueError("没有成功加载任何数据集!")
        
        print(f"总共加载了 {self.total_size:,} 个序列")
        print(f"平均每个文件: {self.total_size / len(self.datasets):.1f} 个序列")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx >= self.total_size:
            raise IndexError("Index out of range")
        
        # 找到对应的数据集
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                dataset_idx = i
                break
        
        # 计算在该数据集中的索引
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]

def setup_logging(log_file: str = None, level: str = 'INFO', rank: int = 0):
    """设置简洁的日志"""
    # 只在主进程输出日志
    if rank != 0:
        return
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def setup_distributed():
    """初始化分布式训练"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

def print_main(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PRM模型训练')
    
    # === 必需参数 ===
    parser.add_argument('--data_dir', type=str, required=True,
                       help='H5文件目录')
    
    # === 数据参数 ===
    parser.add_argument('--h5_pattern', type=str, default='*.h5',
                       help='H5文件匹配模式')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--normalize_obs', action='store_true', default=True,
                       help='归一化观察')
    
    # === 模型参数（互斥组：要么选config，要么自定义） ===
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--config', type=str, 
                           choices=['lightweight', 'standard', 'large'],
                           help='使用预定义模型配置')
    model_group.add_argument('--custom_model', action='store_true',
                           help='使用自定义模型参数')
    
    # === 自定义模型参数（仅在--custom_model时生效） ===
    parser.add_argument('--d_model', type=int, default=512,
                       help='模型维度 (仅自定义模式)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数 (仅自定义模式)')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Transformer层数 (仅自定义模式)')
    parser.add_argument('--d_ff', type=int, default=2048,
                       help='FFN维度 (仅自定义模式)')
    
    # === 通用模型参数 ===
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    parser.add_argument('--seq_length', type=int, default=32,
                       help='序列长度')
    parser.add_argument('--pos_encoding', type=str, default='learnable',
                       choices=['learnable', 'sinusoidal'],
                       help='位置编码类型')
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['mse', 'cosine', 'huber', 'combined'],
                       help='损失函数')
    
    # === 训练参数 ===
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='学习率调度器')
    
    # === 评估和保存 ===
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='验证间隔(epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='保存间隔(epochs)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='恢复训练检查点')
    
    # === 分布式训练参数 ===
    parser.add_argument('--distributed', action='store_true',
                       help='使用分布式训练')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='使用的GPU数量 (None表示使用所有可用GPU)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='本地GPU rank (由torchrun自动设置)')
    
    # === 系统参数 ===
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--log_interval', type=int, default=50,
                       help='日志间隔(batches)')
    
    return parser.parse_args()

def load_datasets(args) -> Tuple[Dataset, Dataset]:
    """加载数据集"""
    # 检查数据目录是否存在
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")
    
    # 查找H5文件
    search_pattern = os.path.join(args.data_dir, args.h5_pattern)
    h5_files = sorted(glob.glob(search_pattern))
    
    if not h5_files:
        raise FileNotFoundError(f"在 {args.data_dir} 中没有找到匹配 '{args.h5_pattern}' 的H5文件")
    
    print(f"在目录 {args.data_dir} 中找到 {len(h5_files)} 个H5文件")
    
    # 根据train_ratio分割文件
    if args.train_ratio >= 1.0:
        # 如果train_ratio >= 1.0，使用所有文件作为训练集，最后一个文件同时作为验证集
        train_files = h5_files
        val_files = [h5_files[-1]]  # 使用最后一个文件作为验证集
        print(f"使用所有 {len(train_files)} 个文件作为训练集")
        print(f"使用最后 1 个文件作为验证集")
    else:
        # 正常分割
        num_train = max(1, int(len(h5_files) * args.train_ratio))
        train_files = h5_files[:num_train]
        val_files = h5_files[num_train:] if num_train < len(h5_files) else [h5_files[-1]]
        print(f"训练文件: {len(train_files)} 个")
        print(f"验证文件: {len(val_files)} 个")
    
    # 创建数据集
    print("\n=== 加载训练数据集 ===")
    train_dataset = create_dataset_from_files(train_files, args.normalize_obs)
    
    print("\n=== 加载验证数据集 ===") 
    val_dataset = create_dataset_from_files(val_files, args.normalize_obs)
    
    return train_dataset, val_dataset

def create_dataset_from_files(h5_files: List[str], normalize_obs: bool) -> Dataset:
    """从文件列表创建数据集的辅助函数"""
    datasets = []
    total_sequences = 0
    
    for i, filepath in enumerate(h5_files):
        if not os.path.exists(filepath):
            print(f"警告: 文件不存在，跳过 {filepath}")
            continue
        
        try:
            dataset = H5SequenceDataset(filepath, normalize_obs)
            datasets.append(dataset)
            total_sequences += len(dataset)
            
            filename = os.path.basename(filepath)
            print(f"  文件 {i+1:2d}: {filename:30s} -> {len(dataset):6d} 个序列")
            
        except Exception as e:
            filename = os.path.basename(filepath)
            print(f"错误: 无法加载 {filename}: {e}")
            continue
    
    if not datasets:
        raise ValueError("没有成功加载任何数据集!")
    
    print(f"总共加载了 {total_sequences:,} 个序列")
    
    # 使用ConcatDataset合并多个数据集
    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)

def create_model(args):
    """创建模型"""
    print_main("创建模型...")
    
    # 创建编码器（从用户本地导入）
    if args.custom_model:
        # 自定义模式：使用用户指定的参数
        img_encoder = ImageEncoder(d_model=args.d_model, dropout=args.dropout)
        action_encoder = ActionEncoder(d_model=args.d_model, dropout=args.dropout)
        
        model = OptimizedEndToEndModel(
            img_encoder=img_encoder,
            action_encoder=action_encoder,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seq_length=args.seq_length,
            loss_type=args.loss_type,
            pos_encoding_type=args.pos_encoding
        )
        print_main(f"自定义模型: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    else:
        # 预定义配置模式：使用create_optimized_model
        configs = {
            "lightweight": {"d_model": 256, "num_heads": 8, "d_ff": 1024, "num_layers": 4},
            "standard": {"d_model": 512, "num_heads": 8, "d_ff": 2048, "num_layers": 6},
            "large": {"d_model": 768, "num_heads": 12, "d_ff": 3072, "num_layers": 8}
        }
        
        config_params = configs[args.config]
        d_model = config_params["d_model"]
        
        img_encoder = ImageEncoder(d_model=d_model, dropout=args.dropout)
        action_encoder = ActionEncoder(d_model=d_model, dropout=args.dropout)
        
        model = create_optimized_model(
            img_encoder=img_encoder,
            action_encoder=action_encoder,
            config=args.config,
            seq_length=args.seq_length,
            pos_encoding_type=args.pos_encoding,
            loss_type=args.loss_type
        )
        print_main(f"预定义配置: {args.config}")
        print_main(f"  参数: d_model={d_model}, layers={config_params['num_layers']}, heads={config_params['num_heads']}")
    
    # 统计参数
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    return model

def setup_distributed_training(model, args):
    """设置分布式训练"""
    if args.distributed:
        # 获取当前进程的rank和world_size
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 将模型移动到对应的GPU
        device = torch.device(f'cuda:{local_rank}')
        model = model.to(device)
        
        # 使用DistributedDataParallel包装模型
        model = DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        
        print_main(f"使用分布式训练: {world_size} 个GPU")
        return model, device, rank, world_size, local_rank
    else:
        # 单GPU或CPU训练
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        model = model.to(device)
        print_main(f"使用设备: {device}")
        return model, device, 0, 1, 0

def train_epoch_with_progress(trainer, train_loader, train_sampler, epoch, args, rank=0):
    """带进度条的训练epoch"""
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    # 创建进度条（只在主进程显示）
    if is_main_process():
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch:3d}/{args.epochs}",
            unit="batch",
            ncols=100,
            leave=False
        )
    else:
        pbar = train_loader
    
    trainer.model.train()
    epoch_metrics = {
        'loss': 0.0,
        'mse_loss': 0.0,
        'cos_similarity': 0.0,
        'num_batches': 0
    }
    
    for batch_idx, batch in enumerate(pbar):
        # 处理不同的batch格式
        if isinstance(batch, dict):
            observations = batch['observations']
            actions = batch['actions']
        else:
            observations, actions = batch
            
        # 移动数据到设备
        observations = observations.to(trainer.device)
        actions = actions.to(trainer.device)
        
        # 训练一个batch
        metrics = trainer.model.train_on_batch(
            observations=observations,
            actions=actions,
            optimizer=trainer.optimizer,
            max_grad_norm=trainer.max_grad_norm,
            return_attention=False
        )
        
        # 累积指标
        for key in epoch_metrics:
            if key in metrics and metrics[key] is not None:
                epoch_metrics[key] += metrics[key]
        epoch_metrics['num_batches'] += 1
        trainer.step_count += 1
        
        # 学习率调度
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        # 更新进度条
        if is_main_process():
            current_lr = trainer.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.4f}",
                'CosSim': f"{metrics['cos_similarity']:.3f}",
                'LR': f"{current_lr:.2e}"
            })
    
    # 计算平均指标
    for key in epoch_metrics:
        if key != 'num_batches':
            epoch_metrics[key] /= epoch_metrics['num_batches']
    
    return epoch_metrics

def evaluate_with_progress(trainer, val_loader, epoch, args):
    """带进度条的验证"""
    if is_main_process():
        pbar = tqdm(
            val_loader,
            desc=f"Eval  {epoch:3d}/{args.epochs}",
            unit="batch", 
            ncols=100,
            leave=False
        )
    else:
        pbar = val_loader
    
    trainer.model.eval()
    eval_metrics = {
        'loss': 0.0,
        'mse_loss': 0.0,
        'cos_similarity_mean': 0.0,
        'accuracy_05': 0.0,
        'num_batches': 0
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 处理不同的batch格式
            if isinstance(batch, dict):
                observations = batch['observations']
                actions = batch['actions']
            else:
                observations, actions = batch
                
            # 移动数据到设备
            observations = observations.to(trainer.device)
            actions = actions.to(trainer.device)
            
            # 评估一个batch
            metrics = trainer.model.eval_on_batch(
                observations=observations,
                actions=actions,
                return_attention=False,
                compute_detailed_metrics=True
            )
            
            # 累积指标
            for key in eval_metrics:
                if key in metrics and metrics[key] is not None:
                    eval_metrics[key] += metrics[key]
            eval_metrics['num_batches'] += 1
            
            # 更新进度条
            if is_main_process():
                pbar.set_postfix({
                    'Loss': f"{metrics['loss']:.4f}",
                    'CosSim': f"{metrics['cos_similarity_mean']:.3f}"
                })
    
    # 计算平均指标
    for key in eval_metrics:
        if key != 'num_batches':
            eval_metrics[key] /= eval_metrics['num_batches']
    
    return eval_metrics

def print_epoch_summary(epoch, args, train_metrics, val_metrics, epoch_time, best_val_loss):
    """打印epoch总结信息"""
    if not is_main_process():
        return
    
    # 格式化时间
    time_str = f"{epoch_time:5.1f}s"
    
    # 训练指标
    train_loss = train_metrics['loss']
    train_cos = train_metrics['cos_similarity']
    
    if val_metrics is not None:
        # 有验证结果
        val_loss = val_metrics['loss']
        val_cos = val_metrics['cos_similarity_mean']
        val_acc = val_metrics['accuracy_05']
        
        # 检查是否为最佳模型
        is_best = val_loss < best_val_loss
        best_marker = " *BEST*" if is_best else ""
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: L={train_loss:.4f} C={train_cos:.3f} | "
              f"Val: L={val_loss:.4f} C={val_cos:.3f} A={val_acc:.3f} | "
              f"Time: {time_str}{best_marker}")
    else:
        # 仅训练结果
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: L={train_loss:.4f} C={train_cos:.3f} | "
              f"Time: {time_str}")

def save_config(args, output_dir):
    """保存配置"""
    if not is_main_process():
        return
        
    config_path = os.path.join(output_dir, 'config.json')
    
    config_dict = vars(args).copy()
    config_dict['timestamp'] = datetime.now().isoformat()
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"配置已保存: {config_path}")

def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    
    # 初始化分布式训练
    rank, world_size, local_rank = 0, 1, 0
    if args.distributed:
        local_rank = setup_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    # 验证参数
    if args.custom_model:
        print_main("使用自定义模型参数")
    else:
        print_main(f"使用预定义配置: {args.config}")
        if any([args.d_model != 512, args.num_heads != 8, args.num_layers != 6, args.d_ff != 2048]):
            print_main("注意: 在预定义配置模式下，自定义模型参数将被忽略")
    
    # 创建输出目录
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name = args.config if not args.custom_model else 'custom'
        args.experiment_name = f"prm_{config_name}_{timestamp}"
    
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    # 同步所有进程
    if args.distributed:
        dist.barrier()
    
    # 设置日志
    log_file = os.path.join(output_dir, 'train.log') if is_main_process() else None
    setup_logging(log_file, rank=rank)
    
    # 打印训练信息
    if is_main_process():
        print("=" * 80)
        print(f"PRM 模型训练")
        print("=" * 80)
        print(f"实验名称: {args.experiment_name}")
        print(f"输出目录: {output_dir}")
        if args.distributed:
            print(f"分布式训练: {world_size} GPU(s)")
        else:
            print(f"单机训练")
        print("=" * 80)
    
    # 保存配置
    save_config(args, output_dir)
    
    try:
        # 加载数据
        if is_main_process():
            print("📁 加载数据集...")
        train_dataset, val_dataset = load_datasets(args)
        
        # 创建数据加载器
        train_loader, val_loader, train_sampler, val_sampler = create_data_loaders(
            train_dataset, val_dataset, args, rank, world_size
        )
        
        print_main(f"📊 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
        
        # 创建模型
        model = create_model(args)
        model, device, rank, world_size, local_rank = setup_distributed_training(model, args)
        
        # 创建训练器
        scheduler_kwargs = {}
        if args.scheduler == 'cosine':
            scheduler_kwargs['T_max'] = args.epochs * len(train_loader)
        elif args.scheduler == 'step':
            scheduler_kwargs['step_size'] = args.epochs // 3
            scheduler_kwargs['gamma'] = 0.1
        
        trainer = create_training_manager(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler if args.scheduler != 'none' else None,
            max_grad_norm=args.max_grad_norm,
            device=device,
            **scheduler_kwargs
        )
        
        # 恢复训练
        start_epoch = 0
        best_val_loss = float('inf')
        
        if args.resume_from:
            if is_main_process():
                print(f"📤 从检查点恢复: {args.resume_from}")
            checkpoint = trainer.load_checkpoint(args.resume_from)
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_metric', float('inf'))
        
        # 训练循环
        print_main("🚀 开始训练...")
        print_main("=" * 80)
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = train_epoch_with_progress(
                trainer, train_loader, train_sampler, epoch, args, rank
            )
            
            # 验证
            val_metrics = None
            if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
                val_metrics = evaluate_with_progress(trainer, val_loader, epoch, args)
                
                # 保存最佳模型
                if is_main_process() and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_path = os.path.join(output_dir, 'best_model.pth')
                    trainer.save_checkpoint(best_path, epoch, best_val_loss)
            
            # 定期保存
            if is_main_process() and (epoch % args.save_interval == 0 or epoch == args.epochs - 1):
                save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
                trainer.save_checkpoint(save_path, epoch, best_val_loss)
            
            # 打印epoch总结
            epoch_time = time.time() - epoch_start
            print_epoch_summary(epoch, args, train_metrics, val_metrics, epoch_time, best_val_loss)
        
        # 训练完成
        if is_main_process():
            print("=" * 80)
            print(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}")
            print(f"📁 输出目录: {output_dir}")
            print("=" * 80)
        
    except KeyboardInterrupt:
        print_main("⚠️  训练被中断")
        # 保存中断检查点
        if is_main_process():
            interrupt_path = os.path.join(output_dir, 'checkpoint_interrupted.pth')
            if 'trainer' in locals():
                trainer.save_checkpoint(interrupt_path, epoch, best_val_loss)
                print(f"💾 中断检查点已保存: {interrupt_path}")
    
    except Exception as e:
        print_main(f"❌ 训练出错: {e}")
        raise
    
    finally:
        # 清理分布式训练
        if args.distributed:
            cleanup_distributed()

if __name__ == '__main__':
    # 使用示例和说明
    if len(sys.argv) == 1:
        print("""
PRM 模型训练脚本

基础使用:
    python train_prm.py --data_dir /path/to/h5/files --config standard

分布式训练 (推荐):
    torchrun --nproc_per_node=4 train_prm.py \\
        --data_dir /path/to/h5/files \\
        --config large \\
        --distributed \\
        --batch_size 64 \\
        --epochs 200

恢复训练:
    torchrun --nproc_per_node=4 train_prm.py \\
        --data_dir /path/to/h5/files \\
        --config standard \\
        --distributed \\
        --resume_from outputs/experiment/best_model.pth

输出说明:
- Epoch 进度条显示实时训练指标
- L: Loss, C: Cosine Similarity, A: Accuracy
- *BEST* 标记当前最佳验证结果
- 检查点自动保存到 outputs/ 目录

参数说明:
- --config: 预定义配置 (lightweight/standard/large)
- --custom_model: 使用自定义模型参数
- --distributed: 启用分布式训练
- --resume_from: 从检查点恢复训练
        """)
        sys.exit(0)
    
    main()