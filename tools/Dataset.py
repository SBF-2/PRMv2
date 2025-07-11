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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class D4RLSequentialDataset(data.Dataset):
    """
    Minari Sequential Dataset for training models with continuous sequences
    Minari datasets provide preprocessed offline RL data
    """
    
    def __init__(self, 
                 game_list: List[str], 
                 num_steps: int = 33,
                 overlap: int = 16,
                 num_seq_per_game: Optional[int] = None,
                 save_path: Optional[str] = None,
                 load_from_file: Optional[str] = None):
        """
        Args:
            game_list: List of Minari dataset names
            num_steps: Number of consecutive steps in each sequence
            overlap: Step size for overlapping sequences (smaller = more overlap)
            num_seq_per_game: Number of sequences to extract per game (None for all)
            save_path: Path to save processed dataset
            load_from_file: Path to load pre-processed dataset
        """
        self.game_list = game_list
        self.num_steps = num_steps
        self.overlap = overlap
        self.num_seq_per_game = num_seq_per_game
        self.save_path = save_path
        
        if load_from_file and os.path.exists(load_from_file):
            self.load_dataset(load_from_file)
        else:
            self.sequences = []
            self.game_indices = []
            self._build_dataset()
            if save_path:
                self.save_dataset(save_path)
    
    def _build_dataset(self):
        """Build dataset by extracting sequences from D4RL games"""
        print("Building sequential dataset from D4RL games...")
        
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
        # print(f"Dataset metadata: {dataset.metadata}")
        
        observations = dataset.observations
        actions = dataset.actions
        rewards = dataset.rewards
        terminals = dataset.terminals
        
        print(f"游戏{game_name}的数据集大小: {len(observations)}")
        
        # Handle timeouts - check if available in minari dataset
        timeouts = getattr(dataset, 'timeouts', np.zeros_like(terminals))
        
        sequences = []
        episode_starts = [0]
        
        # Find episode boundaries
        for i in range(len(terminals)):
            if terminals[i] or (hasattr(timeouts, '__len__') and timeouts[i]):
                episode_starts.append(i + 1)
        
        # Extract sequences from each episode
        for start_idx in episode_starts[:-1]:
            end_idx = episode_starts[episode_starts.index(start_idx) + 1]
            episode_length = end_idx - start_idx
            
            if episode_length < self.num_steps:
                continue  # Skip short episodes
            
            # Extract overlapping sequences from this episode with specified overlap
            for seq_start in range(start_idx, end_idx - self.num_steps + 1, self.overlap):
                sequence = self._build_sequence(
                    observations, actions, rewards, terminals,
                    seq_start, self.num_steps
                )
                sequences.append(sequence)
        
        return sequences
    
    def _build_sequence(self, observations, actions, rewards, terminals, 
                       start_idx: int, num_steps: int) -> Dict:
        """Build a simple sequence from Minari data"""
        sequence_data = {
            'observations': observations[start_idx:start_idx + num_steps],
            'actions': actions[start_idx:start_idx + num_steps],
            'rewards': rewards[start_idx:start_idx + num_steps],
            'terminals': terminals[start_idx:start_idx + num_steps]
        }
        
        return {key: np.array(value) for key, value in sequence_data.items()}
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample"""
        sequence = self.sequences[idx]
        game_idx = self.game_indices[idx]
        
        # Convert to tensors
        sample = {
            'observations': torch.FloatTensor(sequence['observations']),
            'actions': torch.FloatTensor(sequence['actions']),
            'rewards': torch.FloatTensor(sequence['rewards']),
            'terminals': torch.BoolTensor(sequence['terminals']),
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
            self.game_list = [game.decode('utf-8') for game in f.attrs['game_list']]
            
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
                    new_sequence[key] = value[start_offset:start_offset + new_num_steps]
                
                new_sequences.append(new_sequence)
                new_game_indices.append(game_idx)
        
        # Create new dataset instance
        new_dataset = D4RLSequentialDataset.__new__(D4RLSequentialDataset)
        new_dataset.game_list = self.game_list
        new_dataset.num_steps = new_num_steps
        new_dataset.overlap = self.overlap
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
        num_steps=10,
        overlap=16,
        num_seq_per_game=1000,      # Create dataset, extract 1000 sequences per game
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
        print(f"  Terminals shape: {batch['terminals'].shape}")
        print(f"  Game indices shape: {batch['game_idx'].shape}")
        
        if batch_idx == 0:  # Only show first batch
            break