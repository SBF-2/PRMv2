# view_saved_dataset.py
from Dataset_minari import D4RLSequentialDataset,load_h5_dataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import minari

# dataset = minari.load_dataset('atari/alien/expert-v0')
# print(f'TOTAL EPISODES ORIGINAL DATASET: {dataset.total_episodes}')
import os

file_path = 'Data/worker_13.h5'
print(f"检查文件路径: {file_path}")
print(f"文件是否存在: {os.path.exists(file_path)}")
print(f"当前工作目录: {os.getcwd()}")
# 加载已保存的数据集
print("加载数据集...")

dataset = load_h5_dataset('Data/worker_13.h5')
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=0  # 使用 0 避免多进程问题
)

print(f"\n测试 DataLoader:")
for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    for key, value in batch.items():
        print(f"  {key}: shape={value.shape}")
    if batch_idx == 0:  # 只显示第一个 batch
        break