# all_game_list = [
#        "atari/jamesbond/expert-v0", "atari/pitfall/expert-v0", 
#         "atari/robotank/expert-v0", "atari/alien/expert-v0", 
#         "atari/phoenix/expert-v0", "atari/choppercommand/expert-v0", 
#         "atari/centipede/expert-v0", "atari/krull/expert-v0",
#         "atari/frostbite/expert-v0", "atari/breakout/expert-v0", 
#         "atari/kungfumaster/expert-v0", "atari/demonattack/expert-v0", 
#         "atari/fishingderby/expert-v0", "atari/boxing/expert-v0", 
#         "atari/riverraid/expert-v0", "atari/kangaroo/expert-v0",
#         "atari/atlantis/expert-v0", "atari/gopher/expert-v0", 
#         "atari/amidar/expert-v0", "atari/bankheist/expert-v0", 
#         "atari/asteroids/expert-v0", "atari/videopinball/expert-v0", 
#         "atari/asterix/expert-v0", "atari/wizardofwor/expert-v0", 
#         "atari/timepilot/expert-v0", "atari/crazyclimber/expert-v0", 
#         "atari/mspacman/expert-v0", "atari/tutankham/expert-v0", 
#         "atari/skiing/expert-v0", "atari/enduro/expert-v0", 
#         "atari/zaxxon/expert-v0", "atari/pong/expert-v0", 
#         "atari/venture/expert-v0", "atari/roadrunner/expert-v0", 
#         "atari/freeway/expert-v0", "atari/battlezone/expert-v0", 
#         "atari/solaris/expert-v0", "atari/icehockey/expert-v0", 
#         "atari/yarsrevenge/expert-v0", "atari/doubledunk/expert-v0", 
#         "atari/spaceinvaders/expert-v0", "atari/beamrider/expert-v0", 
#         "atari/namethisgame/expert-v0", "atari/upndown/expert-v0", 
#         "atari/tennis/expert-v0", "atari/hero/expert-v0", 
#         "atari/qbert/expert-v0", "atari/surround/expert-v0", 
#         "atari/berzerk/expert-v0", "atari/assault/expert-v0", 
#         "atari/defender/expert-v0", "atari/bowling/expert-v0", 
#         "atari/montezumarevenge/expert-v0", "atari/stargunner/expert-v0", 
#         "atari/privateeye/expert-v0", "atari/seaquest/expert-v0", "atari/gravitar/expert-v0"]
# game = all_game_list[:20]
# print(game)
# view_saved_dataset.py
from Dataset_minari import D4RLSequentialDataset
import matplotlib.pyplot as plt
import torch
import minari

# dataset = D4RLSequentialDataset(
#     game_list=[],
#     load_from_file='../Data/worker_13.h5'
# )

# # Print statistics
# stats = dataset.get_statistics()
# print("Dataset Statistics:")
# for key, value in stats.items():
#     print(f"  {key}: {value}")

# # Create PyTorch DataLoader directly
# from torch.utils.data import DataLoader
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# # Test loading a batch
# for batch_idx, batch in enumerate(dataloader):
#     print(f"\nBatch {batch_idx}:")
#     print(f"  Observations shape: {batch['observations'].shape}")
#     print(f"  Actions shape: {batch['actions'].shape}")
#     print(f"  Rewards shape: {batch['rewards'].shape}")
#     print(f"  Terminations shape: {batch['terminations'].shape}")
#     print(f"  Truncations shape: {batch['truncations'].shape}")
#     print(f"  Game indices shape: {batch['game_idx'].shape}")
    
#     if batch_idx == 0:  # Only show first batch
#         break

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# def inspect_h5_file(filepath):
#     """检查 h5 文件的实际结构"""
#     print(f"检查文件: {filepath}")
#     print("=" * 50)
    
#     with h5py.File(filepath, 'r') as f:
#         print("文件属性 (Attributes):")
#         if len(f.attrs) > 0:
#             for key, value in f.attrs.items():
#                 print(f"  {key}: {value}")
#         else:
#             print("  无属性")
        
#         print(f"\n数据集和组:")
#         def print_item(name, obj):
#             if isinstance(obj, h5py.Dataset):
#                 print(f"  📊 {name}: shape={obj.shape}, dtype={obj.dtype}")
#                 # 如果是小数组，显示一些内容
#                 if obj.size < 20:
#                     print(f"      内容: {obj[:]}")
#             elif isinstance(obj, h5py.Group):
#                 print(f"  📁 {name}: (group)")
        
#         f.visititems(print_item)
        
#         # 特别检查顶层的 keys
#         print(f"\n顶层 keys: {list(f.keys())}")
        
#         return list(f.keys())

# class SimpleH5Dataset(Dataset):
#     """简单的 H5 文件数据集读取器"""
    
#     def __init__(self, filepath, normalize_obs=True):
#         self.filepath = filepath
#         self.normalize_obs = normalize_obs
#         self.data = {}
#         self.length = 0
        
#         self._load_data()
    
#     def _load_data(self):
#         """直接从 h5 文件读取所有数据到内存"""
#         print(f"从 {self.filepath} 读取数据...")
        
#         with h5py.File(self.filepath, 'r') as f:
#             # 方法1: 如果数据是按序列组织的 (sequence_0, sequence_1, ...)
#             sequence_keys = [key for key in f.keys() if key.startswith('sequence_')]
            
#             if sequence_keys:
#                 print(f"发现 {len(sequence_keys)} 个序列")
#                 self._load_sequences(f, sequence_keys)
            
#             # 方法2: 如果数据是直接存储的数组
#             elif 'observations' in f.keys() or 'states' in f.keys() or 'data' in f.keys():
#                 print("发现直接存储的数据数组")
#                 self._load_arrays(f)
            
#             # 方法3: 如果有其他组织方式
#             else:
#                 print("尝试读取所有可用数据...")
#                 self._load_all_available(f)
    
#     def _load_sequences(self, f, sequence_keys):
#         """加载序列格式的数据"""
#         all_obs = []
#         all_actions = []
#         all_rewards = []
#         all_dones = []
        
#         # 按数字顺序排序
#         sequence_keys.sort(key=lambda x: int(x.split('_')[1]))
        
#         for seq_key in sequence_keys:
#             seq_group = f[seq_key]
            
#             if 'observations' in seq_group:
#                 obs = seq_group['observations'][:]
#                 all_obs.append(obs)
            
#             if 'actions' in seq_group:
#                 actions = seq_group['actions'][:]
#                 all_actions.append(actions)
            
#             if 'rewards' in seq_group:
#                 rewards = seq_group['rewards'][:]
#                 all_rewards.append(rewards)
            
#             # 检查 terminations 或 dones
#             if 'terminations' in seq_group:
#                 dones = seq_group['terminations'][:]
#                 all_dones.append(dones)
#             elif 'dones' in seq_group:
#                 dones = seq_group['dones'][:]
#                 all_dones.append(dones)
        
#         # 合并所有序列
#         if all_obs:
#             self.data['observations'] = np.concatenate(all_obs, axis=0)
#         if all_actions:
#             self.data['actions'] = np.concatenate(all_actions, axis=0)
#         if all_rewards:
#             self.data['rewards'] = np.concatenate(all_rewards, axis=0)
#         if all_dones:
#             self.data['dones'] = np.concatenate(all_dones, axis=0)
        
#         self.length = len(self.data['actions']) if 'actions' in self.data else len(self.data['observations'])
#         print(f"加载了 {self.length} 个数据点")
    
#     def _load_arrays(self, f):
#         """加载直接存储的数组数据"""
#         for key in f.keys():
#             if isinstance(f[key], h5py.Dataset):
#                 self.data[key] = f[key][:]
#                 print(f"  加载 {key}: shape={f[key].shape}")
        
#         # 确定数据长度
#         if 'observations' in self.data:
#             self.length = len(self.data['observations'])
#         elif 'states' in self.data:
#             self.length = len(self.data['states'])
#         else:
#             self.length = len(list(self.data.values())[0])
        
#         print(f"数据长度: {self.length}")
    
#     def _load_all_available(self, f):
#         """尝试加载所有可用数据"""
#         def load_recursive(group, prefix=""):
#             for key, item in group.items():
#                 full_key = f"{prefix}{key}" if prefix else key
                
#                 if isinstance(item, h5py.Dataset):
#                     self.data[full_key] = item[:]
#                     print(f"  加载 {full_key}: shape={item.shape}")
#                 elif isinstance(item, h5py.Group):
#                     load_recursive(item, f"{full_key}/")
        
#         load_recursive(f)
        
#         if self.data:
#             self.length = len(list(self.data.values())[0])
#         else:
#             self.length = 0
    
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, idx):
#         """获取单个数据点"""
#         sample = {}
        
#         for key, data in self.data.items():
#             value = data[idx]
            
#             # 转换为 tensor
#             if key == 'observations' and self.normalize_obs:
#                 # 如果是图像数据，归一化到 [0, 1]
#                 if value.dtype == np.uint8:
#                     value = value.astype(np.float32) / 255.0
#                 sample[key] = torch.FloatTensor(value)
#             elif 'action' in key.lower():
#                 sample[key] = torch.FloatTensor(value)
#             elif 'reward' in key.lower():
#                 sample[key] = torch.FloatTensor([value])
#             elif 'done' in key.lower() or 'termination' in key.lower():
#                 sample[key] = torch.BoolTensor([value])
#             else:
#                 # 其他数据尝试转换为 tensor
#                 try:
#                     sample[key] = torch.FloatTensor(value)
#                 except:
#                     sample[key] = value  # 保持原格式
        
#         return sample
    
#     def get_info(self):
#         """获取数据集信息"""
#         info = {
#             'length': self.length,
#             'data_keys': list(self.data.keys()),
#             'shapes': {key: data.shape for key, data in self.data.items()}
#         }
#         return info

# # 使用示例
# if __name__ == "__main__":
    # filepath = 'Data/worker_13.h5'
    
    # # 1. 先检查文件结构
    # print("步骤1: 检查文件结构")
    # inspect_h5_file(filepath)
    
    # print("\n" + "="*50)
    # print("步骤2: 创建数据集")
    
    # # 2. 创建简单的数据集
    # try:
    #     dataset = SimpleH5Dataset(filepath)
        
    #     print(f"\n数据集创建成功!")
    #     print(f"数据集长度: {len(dataset)}")
    #     print(f"数据集信息:")
    #     info = dataset.get_info()
    #     for key, value in info.items():
    #         print(f"  {key}: {value}")
        
    #     # 3. 测试获取一个样本
    #     if len(dataset) > 0:
    #         print(f"\n样本测试:")
    #         sample = dataset[0]
    #         for key, value in sample.items():
    #             if isinstance(value, torch.Tensor):
    #                 print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    #             else:
    #                 print(f"  {key}: {value} (type: {type(value)})")
        
    # except Exception as e:
    #     print(f"错误: {e}")
    #     import traceback
    #     traceback.print_exc()

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any

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
    
    def get_sequence_info(self, idx: int = 0) -> Dict[str, Any]:
        """获取序列信息（用第一个序列作为示例）"""
        if idx >= len(self.sequence_keys):
            idx = 0
            
        sample = self[idx]
        
        info = {
            'total_sequences': len(self.sequence_keys),
            'sequence_length': sample['actions'].shape[0],  # 32
            'observation_shape': sample['observations'].shape,  # (33, 4, 84, 84)
            'action_shape': sample['actions'].shape,  # (32,)
            'reward_shape': sample['rewards'].shape,  # (32,)
            'data_types': {
                'observations': sample['observations'].dtype,
                'actions': sample['actions'].dtype,
                'rewards': sample['rewards'].dtype,
                'terminations': sample['terminations'].dtype,
                'truncations': sample['truncations'].dtype,
            }
        }
        
        return info

def load_h5_dataset(h5_filepath: str, normalize_obs: bool = True) -> H5SequenceDataset:
    """
    简单的函数来加载 H5 序列数据集
    
    Args:
        h5_filepath: H5 文件路径
        normalize_obs: 是否归一化观察到 [0,1]
        
    Returns:
        H5SequenceDataset 实例
    """
    return H5SequenceDataset(h5_filepath, normalize_obs)

# 使用示例
if __name__ == "__main__":
    # 使用你的文件
    h5_file = 'Data/worker_13.h5'
    
    # 创建数据集
    dataset = load_h5_dataset(h5_file)
    
    # 查看数据集信息
    print("数据集信息:")
    info = dataset.get_sequence_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试获取一个序列
    print(f"\n测试加载第一个序列:")
    sample = dataset[0]
    for key, value in sample.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 创建 DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
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