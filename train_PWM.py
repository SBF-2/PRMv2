# train_pwm_advanced.py
# -----------------------------------------------------------------------------
# 使用DDP进行训练，并集成高级训练策略：
# - AdamW 优化器
# - 梯度累计
# - 学习率预热与余弦调度
#
# 运行示例 (2卡, 梯度累计4步, 模拟64*2=128的batch size):
# python train_pwm.py --gradient_accumulation_steps 2  --gpu_ids 0,2,4 --batch_size 4 --epochs 60 --data_dir testData --save_prefix --config 'light'
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import torch.multiprocessing as mp
import h5py
import os
import argparse
import logging
import time
import math
from typing import Dict, Any
from model.PWM import PWM
import csv
# ==============================================================================
# 1. 导入与配置
# ==============================================================================
def get_config(config_name: str = "normal") -> Dict[str, Any]:
    """
    提供统一的配置。
    支持 'light', 'normal', 'large' 三种预设。
    """
    if config_name not in ["light", "normal", "large"]:
        config_name = "normal"
    d_models = {"light": 256, "normal": 512, "large": 768}
    d_model = d_models.get(config_name, 512)
    configs = {
        "light": {
            "d_model": d_model, "dropout": 0.1, "seq_length": 32, "pos_encoding_type": "learnable", "loss_type": "combined",
            "img_encoder_blocks": 2, "img_encoder_groups": 4, "action_encoder_hidden_dim": d_model,
            "transformer_layers": 3, "transformer_heads": 4, "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0 # 方差正则化权重
        },
        "normal": {
            "d_model": d_model, "dropout": 0.1, "seq_length": 32, "pos_encoding_type": "learnable", "loss_type": "combined",
            "img_encoder_blocks": 4, "img_encoder_groups": 8, "action_encoder_hidden_dim": d_model,
            "transformer_layers": 6, "transformer_heads": 8, "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0
        },
        "large": {
            "d_model": d_model, "dropout": 0.1, "seq_length": 32, "pos_encoding_type": "learnable", "loss_type": "combined",
            "img_encoder_blocks": 5, "img_encoder_groups": 16, "action_encoder_hidden_dim": d_model,
            "transformer_layers": 8, "transformer_heads": 12, "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0
        },
    }
    return configs.get(config_name)

# ==============================================================================
# 2. 日志与数据加载 (与之前版本相同)
# ==============================================================================
def setup_logger(args):
    # 输出文件夹路径
    output_path = os.path.join(args.output_dir, args.save_prefix)
    os.makedirs(output_path, exist_ok=True)
    # log文件路径
    log_file = os.path.join(output_path, args.log_file)
    logger = logging.getLogger('PWM_Training'); logger.setLevel(logging.INFO)
    if logger.hasHandlers(): return logger
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

class H5SequenceDataset(Dataset):
    def __init__(self, h5_filepath: str, normalize_obs: bool = True):
        """
        构造函数
        Args:
            h5_filepath (str): H5 文件的路径。
            normalize_obs (bool, optional): 是否归一化观测值。默认为 True。
        """
        # 1. 初始化基本属性
        self.h5_filepath = h5_filepath
        self.normalize_obs = normalize_obs
        self.sequence_keys = []

        # 2. 使用 'with' 语句安全地打开H5文件
        try:
            with h5py.File(h5_filepath, 'r') as f:
                # 3. 获取所有符合条件的键
                keys = [key for key in f.keys() if key.startswith('sequence_')]
                
                # 4. 按数字顺序对键进行排序
                # 例如，将 'sequence_10' 排在 'sequence_2' 之后
                keys.sort(key=lambda x: int(x.split('_')[1]))
                
                # 5. 将排序后的键列表赋值给实例属性
                self.sequence_keys = keys
                
        except Exception as e:
            print(f"错误：无法读取H5文件 {h5_filepath}。 错误信息: {e}")
            # 如果文件无法读取，sequence_keys 将保持为空列表
    def __len__(self):
        return len(self.sequence_keys)
    def __getitem__(self,idx):
        with h5py.File(self.h5_filepath,'r')as f:
            g=f[self.sequence_keys[idx]]
            o,a,r,t,tr=g['observations'][:],g['actions'][:],g['rewards'][:],g['terminations'][:],g['truncations'][:]
        s={'observations':torch.from_numpy(o).float(),'actions':torch.from_numpy(a).long(),'rewards':torch.from_numpy(r).float(),'terminations':torch.from_numpy(t).bool(),'truncations':torch.from_numpy(tr).bool()}
        if self.normalize_obs:
            s['observations']/=255.0
        return s
def create_multih5_dataset(data_dir,normalize_obs=True):
    h5_files=[os.path.join(data_dir,f)for f in os.listdir(data_dir)if f.endswith('.h5')]
    if not h5_files:
        raise FileNotFoundError(f"Directory '{data_dir}' contains no .h5 files.")
    return ConcatDataset([H5SequenceDataset(fp,normalize_obs)for fp in h5_files])

# ==============================================================================
# 3. 学习率调度器
# ==============================================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """创建带预热的余弦学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# ==============================================================================
# 4. DDP 工作函数 
# ==============================================================================
def setup(rank, world_size): 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup(): 
    dist.destroy_process_group()

def validate_epoch(rank, ddp_model, val_loader):
    ddp_model.eval()
    total_metrics = {}
    for batch in val_loader:
        batch = {k: v.to(rank) for k, v in batch.items()}
        metrics = ddp_model.module.eval_on_batch(batch)
        for k, v in metrics.items(): total_metrics[k] = total_metrics.get(k, 0.0) + v
    for k in total_metrics:
        metric_tensor = torch.tensor(total_metrics[k], device=rank)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        total_metrics[k] = metric_tensor.item()
    if rank == 0: 
        return {k: v / (len(val_loader)*dist.get_world_size()) for k, v in total_metrics.items()}
    return {}

def worker_fn(rank, world_size, args):
    if rank == 0: 
        logger = setup_logger(args)
    setup(rank, world_size)
    
    # --- 初始化 (日志、配置、数据) ---
    if rank == 0:
        logger.info("="*50+"\n           训练开始\n"+"="*50)
        logger.info("Hyperparameters & Arguments:")
        [logger.info(f"  - {k}: {v}") for k, v in vars(args).items()]
    config = get_config(args.config)
    
    full_dataset = create_multih5_dataset(args.data_dir)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    # info计入数据量信息
    logger.info(f"{"*"*20} dataset:{len(full_dataset)} sequences {"*"*20}")
    logger.info(f"{"*"*20} Train-dataset:{train_size} sequences {"*"*20}")
    logger.info(f"{"*"*20} val-dataset:{len(val_size)} sequences {"*"*20}")
    
    # 分割数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=val_sampler, pin_memory=True)

    # --- 模型, 优化器, 调度器 ---
    model = PWM(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    
    # --- 【新增】初始化CSV日志和最佳模型跟踪器 ---
    if rank == 0:
        logger.info(f"Model Initialized. Total Trainable Params: {sum(p.numel() for p in ddp_model.parameters() if p.requires_grad):,}")
        best_eval_loss = float('inf')
        # ...
        # 1. 定义输出目录的完整路径
        output_path = os.path.join(args.output_dir, args.save_prefix)

        # 2. 【核心】创建目录，如果它不存在的话
        # os.makedirs 会自动创建所有必需的父目录
        # exist_ok=True 表示如果目录已经存在，则不要抛出错误
        os.makedirs(output_path, exist_ok=True)
        
        # 3. 文件地址
        train_csv_path = os.path.join(output_path, 'train_log.csv')
        eval_csv_path = os.path.join(output_path, 'eval_log.csv')
        # 打开CSV文件并创建写入器
        train_csv_file = open(train_csv_path, 'w', newline='')
        eval_csv_file = open(eval_csv_path, 'w', newline='')

        train_csv_writer = csv.writer(train_csv_file)
        eval_csv_writer = csv.writer(eval_csv_file)
        train_header_written, eval_header_written = False, False

    # --- 训练与验证循环 ---
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        ddp_model.train(), train_sampler.set_epoch(epoch)
        total_train_metrics = {}
        
        if rank == 0: 
            logger.info(f"\n--- EPOCH {epoch+1}/{args.epochs} ---")

        for batch_idx, batch in enumerate(train_loader):
            batch = {k:v.to(rank) for k,v in batch.items()}
            outputs = ddp_model(observations=batch['observations'], actions=batch['actions'])
            loss_dict = ddp_model.module._compute_loss(outputs['predicted'], outputs['target'])
            scaled_loss = loss_dict['total_loss'] / args.gradient_accumulation_steps
            scaled_loss.backward()
            for k,v in loss_dict.items(): total_train_metrics[k] = total_train_metrics.get(k, 0.0) + v.item()
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad()

        # --- Epoch 结束，记录训练日志 ---
        if rank == 0:
            avg_train_metrics = {k: v / len(train_loader) for k, v in total_train_metrics.items()}
            avg_train_metrics['epoch'] = epoch + 1
            
            # 写入训练CSV日志
            if not train_header_written:
                train_csv_writer.writerow(avg_train_metrics.keys())
                train_header_written = True
            train_csv_writer.writerow(avg_train_metrics.values())
            
            logger.info("  Training Summary:")
            for k,v in avg_train_metrics.items(): logger.info(f"    - Avg Train {k}: {v:.6f}")
            
        # --- 【新增】按间隔执行验证 ---
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            val_metrics = validate_epoch(rank, ddp_model, val_loader)

            if rank == 0:
                val_metrics['epoch'] = epoch + 1
                
                # 写入验证CSV日志
                if not eval_header_written:
                    eval_csv_writer.writerow(val_metrics.keys())
                    eval_header_written = True
                eval_csv_writer.writerow(val_metrics.values())

                logger.info("  Validation Summary:")
                for k,v in val_metrics.items(): logger.info(f"    - Avg Val {k}: {v:.6f}")

                # --- 【新增】检查并保存最佳模型 ---
                current_eval_loss = val_metrics['total_loss']
                if current_eval_loss < best_eval_loss:
                    best_eval_loss = current_eval_loss
                    best_model_path = f"{args.output_dir}/{args.save_prefix}/ckp/EvalBest.pth"
                    torch.save(ddp_model.module.state_dict(), best_model_path)
                    logger.info(f"  ** New best model found! ---> Loss: {best_eval_loss:.6f}.  Saved to {best_model_path} **")
                
                elif (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                    saved_model_path = f"{args.output_dir}/{args.save_prefix}/ckp/epoch_{epoch+1}.pth"
                    torch.save(ddp_model.module.state_dict(), saved_model_path)
                    logger.info(f"  ** checkpoint saved in epoch_{epoch+1}! ---> Loss: {current_eval_loss:.6f}.  Saved to {saved_model_path} **")
        
        if rank == 0:
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"  Epoch Duration: {epoch_duration:.2f} seconds\n" + "-"*50)
    
    if rank == 0:
        train_csv_file.close()
        eval_csv_file.close()
        logger.info("CSV log files closed.")

    cleanup()

# ==============================================================================
# 5. 主启动器 (已更新)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Final Advanced PWM Training with DDP.")
    parser.add_argument('--data_dir',type=str,required=True,help='Directory with .h5 files.')
    parser.add_argument('--gpu_ids',type=str,default="0,2,4,6",help='Comma-separated GPU IDs.')
    parser.add_argument('--config',type=str,default='normal',choices=['light','normal','large'])
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--batch_size',type=int,default=16,help='Batch size PER GPU.')
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--num_workers',type=int,default=0)
    parser.add_argument('--val_split',type=float,default=0.1,help='Fraction of data for validation.')
    parser.add_argument('--output_dir',type=str,default='Output',help='Directory to save logs,csv and checkpoints.')
    parser.add_argument('--log_file',type=str,default='training.log')
    parser.add_argument('--save_prefix',type=str,default='episode_0', help='different training episode dir name.')
    
    # 高级训练参数
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--warmup_steps',type=int,default=500)
    
    # 新增控制参数
    parser.add_argument('--eval_interval',type=int,default=2,help='Evaluate every N epochs.')
    parser.add_argument('--save_interval',type=int,default=4,help='Save model every N epochs.')
    parser.add_argument('--log_interval',type=int,default=50,help='Log training status every N optimizer steps.')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    world_size = len(args.gpu_ids.split(','))

    if world_size > 1:
        mp.spawn(worker_fn, args=(world_size, args), nprocs=world_size, join=True)
    else:
        worker_fn(rank=0, world_size=1, args=args)

if __name__ == "__main__":
    main()