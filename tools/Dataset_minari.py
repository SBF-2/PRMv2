import torch
import torch.utils.data as data
import numpy as np
import h5py
import pickle
import random
import os
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import minari
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class D4RLSequentialDataset(data.Dataset):
    """
    Minari Sequential Dataset for training models with continuous sequences
    Applies D4RL-Atari style preprocessing: RGB -> Grayscale -> 84x84 -> 4-frame stacking
    """
    
    def __init__(self, 
                 game_list: List[str], 
                 num_steps: int = 33,
                 overlap: int = 16,
                 frame_stack: int = 4,  # Number of frames to stack (4 for Atari)
                 normalize_obs: bool = True,  # 添加归一化选项
                 num_seq_per_game: Optional[int] = None,
                 save_path: Optional[str] = None,
                 load_from_file: Optional[str] = None):
        """
        Args:
            game_list: List of Minari dataset names
            num_steps: Number of consecutive steps in each sequence
            overlap: Step size for overlapping sequences (smaller = more overlap)
            frame_stack: Number of frames to stack for observation (4 for Atari)
            num_seq_per_game: Number of sequences to extract per game (None for all)
            save_path: Path to save processed dataset
            load_from_file: Path to load pre-processed dataset
        """
        self.game_list = game_list
        self.num_steps = num_steps
        self.frame_stack = frame_stack
        self.overlap = overlap
        self.num_seq_per_game = num_seq_per_game
        self.save_path = save_path
        self.normalize_obs = normalize_obs
        
        if load_from_file and os.path.exists(load_from_file):
            self.load_dataset(load_from_file)
        else:
            self.sequences = []
            self.game_indices = []
            self._build_dataset()
            if save_path:
                self.save_dataset(save_path)
    
    def _build_dataset(self):
        """Build dataset by extracting sequences from Minari games"""
        print("Building sequential dataset from Minari games...")
        
        all_sequences = []
        game_sequence_counts = []
        
        for game_idx, game_name in enumerate(self.game_list):
            print(f"Processing {game_name}...")
            sequences = self._extract_sequences_from_game(game_name)
            
            if self.num_seq_per_game:
                sequences = sequences[:self.num_seq_per_game]
            
            all_sequences.extend(sequences)
            game_sequence_counts.append(len(sequences))
            
            # Store game index for each sequence
            self.game_indices.extend([game_idx] * len(sequences))
        
        self.sequences = all_sequences
        print(f"Total sequences extracted: {len(self.sequences)}")
        print(f"Sequences per game: {dict(zip(self.game_list, game_sequence_counts))}")
        
        # Shuffle sequences to ensure random distribution
        combined = list(zip(self.sequences, self.game_indices))
        random.shuffle(combined)
        self.sequences, self.game_indices = zip(*combined)
        self.sequences = list(self.sequences)
        self.game_indices = list(self.game_indices)
    
    def _extract_sequences_from_game(self, game_name: str) -> List[Dict]:
        """Extract sequences from a single Minari dataset"""
        # Load Minari dataset
        dataset = minari.load_dataset(game_name, download=True)
        
        sequences = []
        episode_count = 0
        
        # Iterate through episodes in the dataset
        for episode_data in dataset.iterate_episodes():
            observations = episode_data.observations
            actions = episode_data.actions
            rewards = episode_data.rewards
            terminations = episode_data.terminations
            truncations = episode_data.truncations
            
            episode_length = len(actions)  # actions has same length as steps
            print(f"Episode {episode_count}: length {episode_length}")
            
            if episode_length < self.num_steps:
                episode_count += 1
                continue  # Skip short episodes
            
            # Extract overlapping sequences from this episode with specified overlap
            for seq_start in range(0, episode_length - self.num_steps + 1, self.overlap):
                sequence = self._build_sequence(
                    observations, actions, rewards, terminations, truncations,
                    seq_start, self.num_steps
                )
                sequences.append(sequence)
            
            episode_count += 1
            
            # Optional: limit episodes processed for testing
            if self.num_seq_per_game and len(sequences) >= self.num_seq_per_game:
                break
        
        print(f"游戏{game_name}处理了{episode_count}个episodes, 提取了{len(sequences)}个序列")
        return sequences
    
    def _preprocess_observation(self, obs):
        """Preprocess observation: RGB -> Grayscale -> 84x84"""
        # Convert RGB to grayscale (210, 160, 3) -> (210, 160)
        if len(obs.shape) == 3:
            # Use luminance formula: 0.299*R + 0.587*G + 0.114*B
            gray = np.dot(obs[...,:3], [0.299, 0.587, 0.114])
        else:
            gray = obs
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized.astype(np.uint8)
    
    def _stack_frames(self, observations, start_idx, num_steps):
        """Stack frames with backward-looking approach"""
        # Preprocess all observations first
        processed_obs = []
        for obs in observations:
            processed_obs.append(self._preprocess_observation(obs))
        
        stacked_sequences = []
        
        for step in range(num_steps + 1):  # +1 because observations has extra initial obs
            current_idx = start_idx + step
            
            # Backward-looking frame stacking
            frame_stack = []
            for i in range(self.frame_stack):
                frame_idx = current_idx - self.frame_stack + 1 + i
                # If frame_idx is before episode start, use the first frame
                frame_idx = max(start_idx, frame_idx)
                frame_stack.append(processed_obs[frame_idx])
            
            # Stack frames: (4, 84, 84)
            stacked_frame = np.stack(frame_stack, axis=0)
            stacked_sequences.append(stacked_frame)
        
        return np.array(stacked_sequences)
    
    def _build_sequence(self, observations, actions, rewards, terminations, truncations,
                       start_idx: int, num_steps: int) -> Dict:
        """Build a sequence with frame stacking from Minari episode data"""
        # Stack frames for observations
        stacked_observations = self._stack_frames(observations, start_idx, num_steps)
        
        sequence_data = {
            'observations': stacked_observations,  # Shape: (num_steps + 1, 4, 84, 84)
            'actions': actions[start_idx:start_idx + num_steps],
            'rewards': rewards[start_idx:start_idx + num_steps],
            'terminations': terminations[start_idx:start_idx + num_steps],
            'truncations': truncations[start_idx:start_idx + num_steps]
        }
        
        return {key: np.array(value) for key, value in sequence_data.items()}
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        game_idx = self.game_indices[idx]
        
        # 转换observations
        obs = torch.FloatTensor(sequence['observations'])
        if self.normalize_obs:
            obs = obs / 255.0  # 归一化到[0, 1]
        
        sample = {
            'observations': obs,
            'actions': torch.FloatTensor(sequence['actions']),
            'rewards': torch.FloatTensor(sequence['rewards']),
            'terminations': torch.BoolTensor(sequence['terminations']),
            'truncations': torch.BoolTensor(sequence['truncations']),
            'game_idx': torch.LongTensor([game_idx])
        }
        
        return sample
    
    def save_dataset(self, filepath: str):
        """Save processed dataset to file"""
        print(f"Saving dataset to {filepath}...")
        
        # Use HDF5 for efficient storage and future sequence splitting
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            f.attrs['num_sequences'] = len(self.sequences)
            f.attrs['num_steps'] = self.num_steps
            f.attrs['overlap'] = self.overlap
            f.attrs['frame_stack'] = self.frame_stack
            f.attrs['game_list'] = [game.encode('utf-8') for game in self.game_list]
            
            # Save game indices
            f.create_dataset('game_indices', data=np.array(self.game_indices))
            
            # Save sequences
            for seq_idx, sequence in enumerate(self.sequences):
                grp = f.create_group(f'sequence_{seq_idx}')
                for key, value in sequence.items():
                    grp.create_dataset(key, data=value, compression='gzip')
        
        print(f"Dataset saved successfully!")
    
    def load_dataset(self, filepath: str):
        """Load processed dataset from file"""
        print(f"Loading dataset from {filepath}...")
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            self.num_steps = f.attrs['num_steps']
            self.overlap = f.attrs.get('overlap', 1)  # Default to 1 if not saved
            self.frame_stack = f.attrs.get('frame_stack', 4)  # Default to 4 if not saved
            self.game_list = [game for game in f.attrs['game_list']]
            
            # Load game indices
            self.game_indices = list(f['game_indices'][:])
            
            # Load sequences
            self.sequences = []
            num_sequences = f.attrs['num_sequences']
            
            for seq_idx in range(num_sequences):
                grp = f[f'sequence_{seq_idx}']
                sequence = {}
                for key in grp.keys():
                    sequence[key] = grp[key][:]
                self.sequences.append(sequence)
        
        print(f"Dataset loaded successfully! Total sequences: {len(self.sequences)}")
    
    def split_sequences(self, new_num_steps: int, save_path: Optional[str] = None) -> 'D4RLSequentialDataset':
        """Split existing sequences into shorter sequences"""
        if new_num_steps >= self.num_steps:
            raise ValueError("New sequence length must be shorter than current length")
        
        print(f"Splitting sequences from {self.num_steps} to {new_num_steps} steps...")
        
        new_sequences = []
        new_game_indices = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            game_idx = self.game_indices[seq_idx]
            
            # Calculate how many new sequences we can create
            num_new_sequences = self.num_steps - new_num_steps + 1
            
            for start_offset in range(num_new_sequences):
                new_sequence = {}
                for key, value in sequence.items():
                    if key == 'observations':
                        # observations has one extra element (initial obs)
                        new_sequence[key] = value[start_offset:start_offset + new_num_steps + 1]
                    else:
                        # actions, rewards, terminations, truncations
                        new_sequence[key] = value[start_offset:start_offset + new_num_steps]
                
                new_sequences.append(new_sequence)
                new_game_indices.append(game_idx)
        
        # Create new dataset instance
        new_dataset = D4RLSequentialDataset.__new__(D4RLSequentialDataset)
        new_dataset.game_list = self.game_list
        new_dataset.num_steps = new_num_steps
        new_dataset.overlap = self.overlap
        new_dataset.frame_stack = self.frame_stack
        new_dataset.sequences = new_sequences
        new_dataset.game_indices = new_game_indices
        new_dataset.num_seq_per_game = None
        new_dataset.save_path = save_path
        
        if save_path:
            new_dataset.save_dataset(save_path)
        
        print(f"Split complete! New dataset has {len(new_sequences)} sequences")
        return new_dataset
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        game_counts = defaultdict(int)
        for game_idx in self.game_indices:
            game_counts[self.game_list[game_idx]] += 1
        
        # Calculate observation and action shapes
        sample_seq = self.sequences[0]
        obs_shape = sample_seq['observations'].shape
        action_shape = sample_seq['actions'].shape
        
        stats = {
            'total_sequences': len(self.sequences),
            'sequence_length': self.num_steps,
            'overlap': self.overlap,
            'frame_stack': self.frame_stack,
            'games': dict(game_counts),
            'observation_shape': obs_shape,
            'action_shape': action_shape
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Define games to use
    game_list = [
        'atari/alien/expert-v0'
    ] 
    # Create dataset
    dataset = D4RLSequentialDataset(
        game_list=game_list,
        num_steps=32,
        overlap=16,
        num_seq_per_game=100,      # Create dataset, extract 1000 sequences per game
        save_path='/Users/feisong/Desktop/self-experience/code/PRM_v2/Data/d4rl_sequential_dataset.h5'
    )
    
    # Print statistics
    stats = dataset.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create PyTorch DataLoader directly
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Test loading a batch
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Observations shape: {batch['observations'].shape}")
        print(f"  Actions shape: {batch['actions'].shape}")
        print(f"  Rewards shape: {batch['rewards'].shape}")
        print(f"  Terminations shape: {batch['terminations'].shape}")
        print(f"  Truncations shape: {batch['truncations'].shape}")
        print(f"  Game indices shape: {batch['game_idx'].shape}")
        
        if batch_idx == 0:  # Only show first batch
            break