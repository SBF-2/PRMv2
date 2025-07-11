# view_saved_dataset.py
from Dataset_minari import D4RLSequentialDataset
import matplotlib.pyplot as plt
import torch

import minari

dataset = minari.load_dataset('atari/alien/expert-v0')
print(f'TOTAL EPISODES ORIGINAL DATASET: {dataset.total_episodes}')

# # 加载已保存的数据集
# print("加载数据集...")
# dataset = D4RLSequentialDataset(
#     game_list=[],
#     load_from_file='/Users/feisong/Desktop/self-experience/code/PRM_v2/Data/d4rl_sequential_dataset.h5'
# )
# print(f'TOTAL EPISODES ORIGINAL DATASET: {dataset.total_episodes}')
# print(f"数据集加载完成！共有 {len(dataset)} 条序列")

# # 查看统计信息
# stats = dataset.get_statistics()
# print("\n=== 数据集统计信息 ===")
# for key, value in stats.items():
#     print(f"{key}: {value}")

# # 查看第一条数据
# sample = dataset[0]
# print("\n=== 第一条数据 ===")
# for key, value in sample.items():
#     print(f"{key}: {value.shape}, dtype: {value.dtype}")

# # 查看数值范围
# print(f"\n观察值范围: [{sample['observations'].min():.1f}, {sample['observations'].max():.1f}]")
# print(f"动作范围: [{sample['actions'].min():.1f}, {sample['actions'].max():.1f}]")
# print(f"奖励总和: {sample['rewards'].sum():.2f}")

# print("\n数据加载成功！你现在可以使用 dataset[i] 查看任意一条数据")

# sample = dataset[60]
# obs_t0 = sample['observations'][9]  # shape: [4, 84, 84]

# # 显示4帧
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# for i in range(4):
#     axes[i].imshow(obs_t0[i], cmap='gray')
#     axes[i].set_title(f'Frame {i}')
#     axes[i].axis('off')

# plt.suptitle('第0条数据，时间步0的4帧堆叠')
# plt.tight_layout()
# plt.show()