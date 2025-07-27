# PWM_validation.py
# Predictive World Model Validation Framework (Final)
# Updated to match training data format and model interface
# -----------------------------------------------------------------------------
# Data format from training:
# - observations: (batch_size, sequence_length, 84, 84) - normalized [0,1]
# - actions: (batch_size, sequence_length) - long tensor
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
from datetime import datetime
import json
from tqdm import tqdm
from scipy.spatial.distance import squareform, pdist
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class PWMValidator:
    """
    Comprehensive validation framework for PWM models.
    Updated to match actual training data format and model interface.
    """
    
    def __init__(self, model, device='cuda', save_dir='validation_results'):
        """
        Initialize the validator.
        
        Args:
            model: Trained PWM model
            device: Device to run validation on
            save_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.model.eval()
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Storage for results
        self.all_predictions = []
        self.all_ground_truths = []
        self.step_by_step_outputs = []
        self.batch_image_similarities = []
        self.batch_output_similarities = []
        
    def prepare_data_for_stepwise_prediction(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for stepwise prediction.
        
        Args:
            observations: (B, seq_length, 84, 84) - from training format
            actions: (B, seq_length) - from training format
            
        Returns:
            first_observation: (B, 1, 84, 84) - First frame for stepwise prediction
            actions: (B, seq_length) - Actions unchanged
        """
        batch_size, seq_length, height, width = observations.shape
        
        # Get first observation for stepwise prediction
        first_observation = observations[:, 0:1, :, :]  # (B, 1, 84, 84)
        
        return first_observation, actions
    
    def stepwise_prediction(self, first_observation: torch.Tensor, actions: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Perform step-by-step prediction using only the first observation and action sequence.
        
        Args:
            first_observation: (B, 1, 84, 84) - First observation
            actions: (B, seq_length) - Complete action sequence
            
        Returns:
            predictions: List of predicted features for each step
            final_prediction_sequence: (B, seq_length, d_model) - All predictions concatenated
        """
        batch_size, seq_length = actions.shape
        predictions = []
        
        with torch.no_grad():
            # Convert single channel to 4-channel format expected by PWM model
            # Add fake first frame by duplicating the first observation
            first_obs_4ch = first_observation.unsqueeze(2).repeat(1, 1, 4, 1, 1)  # (B, 1, 4, 84, 84)
            
            # Encode the first observation to get initial feature
            current_img_feature = self.model.img_encoder(first_obs_4ch).squeeze(1)  # (B, d_model)
            
            for step in range(seq_length):
                # Get current action
                current_action = actions[:, step:step+1]  # (B, 1)
                
                # Encode current action
                action_feature = self.model.action_encoder(current_action)  # (B, 1, d_model)
                
                # Prepare features for transformer
                img_feature_seq = current_img_feature.unsqueeze(1)  # (B, 1, d_model)
                
                # Pass through transformer
                transformer_output = self.model.transformer(img_feature_seq, action_feature)  # (B, 1, d_model)
                
                # Apply output projection
                predicted_feature = self.model.output_proj(transformer_output).squeeze(1)  # (B, d_model)
                
                predictions.append(predicted_feature.clone())
                
                # Update current image feature for next step
                current_img_feature = predicted_feature
        
        # Stack all predictions
        final_prediction_sequence = torch.stack(predictions, dim=1)  # (B, seq_length, d_model)
        
        return predictions, final_prediction_sequence
    
    def validate_batch(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate a single batch and compute comprehensive metrics.
        
        Args:
            batch_data: Dictionary containing 'observations' and 'actions' in training format
            
        Returns:
            Dictionary containing all computed metrics
        """
        observations = batch_data['observations'].to(self.device)  # (B, seq_length, 84, 84)
        actions = batch_data['actions'].to(self.device)  # (B, seq_length)
        
        batch_size, seq_length, height, width = observations.shape
        
        # Prepare observations for model (convert to 4-channel format expected by PWM)
        # Create fake sequence by duplicating first frame
        first_obs = observations[:, 0:1, :, :].unsqueeze(2).repeat(1, 1, 4, 1, 1)  # (B, 1, 4, 84, 84)
        rest_obs = observations.unsqueeze(2).repeat(1, 1, 4, 1, 1)  # (B, seq_length, 4, 84, 84)
        model_observations = torch.cat([first_obs, rest_obs], dim=1)  # (B, seq_length+1, 4, 84, 84)
        
        # Get ground truth features using model's forward pass
        with torch.no_grad():
            # Use model's forward method to get ground truth
            model_output = self.model(observations=model_observations, actions=actions)
            ground_truth_features = model_output['target']  # (B, seq_length, d_model)
        
        # Perform step-by-step prediction
        first_observation, _ = self.prepare_data_for_stepwise_prediction(observations, actions)
        step_predictions, predicted_sequence = self.stepwise_prediction(first_observation, actions)
        
        # Store results for later analysis
        self.all_predictions.append(predicted_sequence.cpu())
        self.all_ground_truths.append(ground_truth_features.cpu())
        self.step_by_step_outputs.extend([pred.cpu() for pred in step_predictions])
        
        # Compute metrics
        metrics = {}
        
        # 1. Image-to-image similarity with heatmap data
        img_similarity_metrics, img_similarity_matrix = self.compute_image_similarity(observations)
        metrics.update(img_similarity_metrics)
        self.batch_image_similarities.append(img_similarity_matrix)
        
        # 2. Output-to-output similarity with heatmap data
        output_similarity_metrics, output_similarity_matrix = self.compute_output_similarity(predicted_sequence)
        metrics.update(output_similarity_metrics)
        self.batch_output_similarities.append(output_similarity_matrix)
        
        # 3. Prediction vs Ground Truth metrics (per-step)
        pred_gt_metrics = self.compute_prediction_accuracy(predicted_sequence, ground_truth_features)
        metrics.update(pred_gt_metrics)
        
        return metrics
    
    def compute_image_similarity(self, observations: torch.Tensor) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Compute MSE similarity between all pairs of normalized observations in a batch.
        
        Args:
            observations: (B, seq_length, 84, 84) - normalized observations
            
        Returns:
            Dictionary with image similarity metrics
            Similarity matrix for heatmap visualization
        """
        batch_size, seq_len, height, width = observations.shape
        
        # Observations are already normalized in training (divided by 255.0)
        # Reshape for easier computation
        obs_flat = observations.view(batch_size, seq_len, -1)  # (B, seq_len, 84*84)
        
        # Compute similarity matrix averaged across batch
        similarity_matrices = []
        all_similarities = []
        
        for b in range(batch_size):
            # Compute pairwise MSE distances for this batch item
            batch_obs = obs_flat[b]  # (seq_len, 84*84)
            distances = pdist(batch_obs.cpu().numpy(), metric='euclidean')
            similarity_matrix = squareform(distances)
            similarity_matrices.append(similarity_matrix)
            
            # Collect all pairwise similarities (upper triangle)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    mse_sim = similarity_matrix[i, j]
                    all_similarities.append(mse_sim)
        
        # Average similarity matrix across batch
        avg_similarity_matrix = np.mean(similarity_matrices, axis=0)
        
        return {
            'image_mse_similarity_mean': np.mean(all_similarities),
            'image_mse_similarity_std': np.std(all_similarities),
            'image_mse_similarity_median': np.median(all_similarities)
        }, avg_similarity_matrix
    
    def compute_output_similarity(self, predictions: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute MSE and cosine similarity between all pairs of outputs in a batch.
        
        Args:
            predictions: (B, seq_length, d_model)
            
        Returns:
            Dictionary with output similarity metrics
            Similarity matrices for heatmap visualization
        """
        batch_size, seq_len, d_model = predictions.shape
        
        mse_matrices = []
        cosine_matrices = []
        all_mse_similarities = []
        all_cosine_similarities = []
        
        for b in range(batch_size):
            # Initialize matrices
            mse_matrix = np.zeros((seq_len, seq_len))
            cosine_matrix = np.zeros((seq_len, seq_len))
            
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        # MSE similarity
                        mse_sim = F.mse_loss(predictions[b, i], predictions[b, j], reduction='mean')
                        mse_matrix[i, j] = mse_sim.item()
                        if i < j:  # Only collect upper triangle once
                            all_mse_similarities.append(mse_sim.item())
                        
                        # Cosine similarity
                        cos_sim = F.cosine_similarity(
                            predictions[b, i].unsqueeze(0), 
                            predictions[b, j].unsqueeze(0), 
                            dim=1
                        )
                        cosine_matrix[i, j] = cos_sim.item()
                        if i < j:  # Only collect upper triangle once
                            all_cosine_similarities.append(cos_sim.item())
                    else:
                        # Diagonal elements
                        mse_matrix[i, j] = 0.0
                        cosine_matrix[i, j] = 1.0
            
            mse_matrices.append(mse_matrix)
            cosine_matrices.append(cosine_matrix)
        
        # Average matrices across batch
        avg_mse_matrix = np.mean(mse_matrices, axis=0)
        avg_cosine_matrix = np.mean(cosine_matrices, axis=0)
        
        return {
            'output_mse_similarity_mean': np.mean(all_mse_similarities),
            'output_mse_similarity_std': np.std(all_mse_similarities),
            'output_cosine_similarity_mean': np.mean(all_cosine_similarities),
            'output_cosine_similarity_std': np.std(all_cosine_similarities)
        }, {'mse': avg_mse_matrix, 'cosine': avg_cosine_matrix}
    
    def compute_prediction_accuracy(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, Any]:
        """
        Compute MSE and (1-cosine similarity) between predictions and ground truth per step.
        
        Args:
            predictions: (B, seq_length, d_model)
            ground_truth: (B, seq_length, d_model)
            
        Returns:
            Dictionary with prediction accuracy metrics
        """
        batch_size, seq_length, d_model = predictions.shape
        
        # Per-step metrics
        step_mse = []
        step_cosine_distance = []  # 1 - cosine_similarity
        
        for step in range(seq_length):
            step_mse_val = F.mse_loss(predictions[:, step], ground_truth[:, step], reduction='mean')
            step_cos_val = F.cosine_similarity(predictions[:, step], ground_truth[:, step], dim=1).mean()
            step_cosine_distance_val = 1 - step_cos_val  # Convert to distance
            
            step_mse.append(step_mse_val.item())
            step_cosine_distance.append(step_cosine_distance_val.item())
        
        # Overall metrics
        overall_mse = F.mse_loss(predictions, ground_truth, reduction='mean')
        pred_flat = predictions.view(-1, predictions.size(-1))
        gt_flat = ground_truth.view(-1, ground_truth.size(-1))
        overall_cosine = F.cosine_similarity(pred_flat, gt_flat, dim=1).mean()
        overall_cosine_distance = 1 - overall_cosine
        
        return {
            'pred_gt_mse_overall': overall_mse.item(),
            'pred_gt_cosine_distance_overall': overall_cosine_distance.item(),
            'pred_gt_mse_per_step': step_mse,
            'pred_gt_cosine_distance_per_step': step_cosine_distance,
            'pred_gt_mse_step_std': np.std(step_mse),
            'pred_gt_cosine_distance_step_std': np.std(step_cosine_distance)
        }
    
    def run_validation(self, dataloader, num_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Run validation on multiple batches.
        
        Args:
            dataloader: DataLoader providing validation data in training format
            num_batches: Number of batches to validate (None for all)
            
        Returns:
            Aggregated validation results
        """
        print("Starting PWM model validation...")
        
        all_metrics = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Validating")):
                if num_batches is not None and batch_count >= num_batches:
                    break
                
                metrics = self.validate_batch(batch_data)
                all_metrics.append(metrics)
                batch_count += 1
        
        # Aggregate metrics
        aggregated_metrics = self.aggregate_metrics(all_metrics)
        
        # Save results and generate visualizations
        self.save_results(aggregated_metrics)
        
        return aggregated_metrics
    
    def aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across all batches."""
        aggregated = {}
        
        # Simple metrics (means and stds)
        simple_keys = [
            'image_mse_similarity_mean', 'image_mse_similarity_std', 'image_mse_similarity_median',
            'output_mse_similarity_mean', 'output_mse_similarity_std',
            'output_cosine_similarity_mean', 'output_cosine_similarity_std',
            'pred_gt_mse_overall', 'pred_gt_cosine_distance_overall',
            'pred_gt_mse_step_std', 'pred_gt_cosine_distance_step_std'
        ]
        
        for key in simple_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[f'{key}_batch_mean'] = np.mean(values)
                aggregated[f'{key}_batch_std'] = np.std(values)
        
        # Per-step metrics
        per_step_mse = []
        per_step_cosine_distance = []
        
        for metrics in all_metrics:
            if 'pred_gt_mse_per_step' in metrics:
                per_step_mse.append(metrics['pred_gt_mse_per_step'])
            if 'pred_gt_cosine_distance_per_step' in metrics:
                per_step_cosine_distance.append(metrics['pred_gt_cosine_distance_per_step'])
        
        if per_step_mse:
            per_step_mse_array = np.array(per_step_mse)
            aggregated['per_step_mse_mean'] = per_step_mse_array.mean(axis=0).tolist()
            aggregated['per_step_mse_std'] = per_step_mse_array.std(axis=0).tolist()
        
        if per_step_cosine_distance:
            per_step_cosine_distance_array = np.array(per_step_cosine_distance)
            aggregated['per_step_cosine_distance_mean'] = per_step_cosine_distance_array.mean(axis=0).tolist()
            aggregated['per_step_cosine_distance_std'] = per_step_cosine_distance_array.std(axis=0).tolist()
        
        return aggregated
    
    def save_results(self, metrics: Dict[str, Any]):
        """Save validation results to files and generate visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics as JSON
        metrics_file = os.path.join(self.save_dir, f'validation_metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed similarities to CSV
        self.save_similarities_csv(timestamp)
        
        # Save step-by-step outputs
        self.save_step_outputs(timestamp)
        
        # Generate all visualizations
        self.generate_visualizations(metrics, timestamp)
        
        print(f"Results saved to {self.save_dir}")
    
    def save_similarities_csv(self, timestamp: str):
        """Save detailed similarity data to CSV."""
        similarity_data = []
        
        for batch_idx, (predictions, ground_truths) in enumerate(zip(self.all_predictions, self.all_ground_truths)):
            batch_size, seq_len, d_model = predictions.shape
            
            for b in range(batch_size):
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        # Output similarities
                        mse_sim = F.mse_loss(predictions[b, i], predictions[b, j], reduction='mean').item()
                        cos_sim = F.cosine_similarity(
                            predictions[b, i].unsqueeze(0), 
                            predictions[b, j].unsqueeze(0), 
                            dim=1
                        ).item()
                        
                        similarity_data.append({
                            'batch_idx': batch_idx,
                            'sample_idx': b,
                            'step_i': i,
                            'step_j': j,
                            'output_mse_similarity': mse_sim,
                            'output_cosine_similarity': cos_sim
                        })
        
        df = pd.DataFrame(similarity_data)
        csv_file = os.path.join(self.save_dir, f'similarity_details_{timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"Detailed similarity data saved to {csv_file}")
    
    def save_step_outputs(self, timestamp: str):
        """Save step-by-step outputs for further analysis."""
        outputs_file = os.path.join(self.save_dir, f'step_outputs_{timestamp}.pt')
        torch.save({
            'step_by_step_outputs': self.step_by_step_outputs,
            'all_predictions': self.all_predictions,
            'all_ground_truths': self.all_ground_truths
        }, outputs_file)
        print(f"Step-by-step outputs saved to {outputs_file}")
    
    def generate_visualizations(self, metrics: Dict[str, Any], timestamp: str):
        """Generate all required visualizations."""
        plt.style.use('default')
        
        # 1. Generate heatmaps for image and output similarities
        self.plot_similarity_heatmaps(timestamp)
        
        # 2. Generate Hinton diagram for outputs
        self.plot_hinton_diagram(timestamp)
        
        # 3. Generate performance curves
        self.plot_performance_curves(metrics, timestamp)
    
    def plot_similarity_heatmaps(self, timestamp: str):
        """Plot heatmaps for image and output similarities."""
        # Average similarity matrices across all batches
        if self.batch_image_similarities:
            avg_img_similarity = np.mean(self.batch_image_similarities, axis=0)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Image similarity heatmap
            sns.heatmap(avg_img_similarity, annot=True, fmt='.3f', cmap='viridis', 
                       ax=axes[0, 0], cbar_kws={'label': 'Euclidean Distance'})
            axes[0, 0].set_title('Observation-to-Observation Similarity Heatmap')
            axes[0, 0].set_xlabel('Frame Index')
            axes[0, 0].set_ylabel('Frame Index')
            
            # Output similarities
            if self.batch_output_similarities:
                avg_output_mse = np.mean([sim['mse'] for sim in self.batch_output_similarities], axis=0)
                avg_output_cosine = np.mean([sim['cosine'] for sim in self.batch_output_similarities], axis=0)
                
                # Output MSE similarity heatmap
                sns.heatmap(avg_output_mse, annot=True, fmt='.3f', cmap='plasma',
                           ax=axes[0, 1], cbar_kws={'label': 'MSE Distance'})
                axes[0, 1].set_title('Output-to-Output MSE Similarity Heatmap')
                axes[0, 1].set_xlabel('Output Index')
                axes[0, 1].set_ylabel('Output Index')
                
                # Output cosine similarity heatmap
                sns.heatmap(avg_output_cosine, annot=True, fmt='.3f', cmap='coolwarm',
                           ax=axes[1, 0], cbar_kws={'label': 'Cosine Similarity'})
                axes[1, 0].set_title('Output-to-Output Cosine Similarity Heatmap')
                axes[1, 0].set_xlabel('Output Index')
                axes[1, 0].set_ylabel('Output Index')
                
                # Combined visualization
                combined_sim = avg_output_mse / np.max(avg_output_mse) + (1 - avg_output_cosine)
                sns.heatmap(combined_sim, annot=True, fmt='.3f', cmap='RdYlBu_r',
                           ax=axes[1, 1], cbar_kws={'label': 'Combined Distance'})
                axes[1, 1].set_title('Combined Output Similarity (MSE + Cosine Distance)')
                axes[1, 1].set_xlabel('Output Index')
                axes[1, 1].set_ylabel('Output Index')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'similarity_heatmaps_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_hinton_diagram(self, timestamp: str):
        """Plot Hinton diagram for all outputs in a batch."""
        if not self.all_predictions:
            return
            
        # Take the first batch for visualization
        first_batch_predictions = self.all_predictions[0]  # (B, seq_length, d_model)
        batch_size, seq_length, d_model = first_batch_predictions.shape
        
        # Reshape to (batch_size * seq_length, d_model) for Hinton diagram
        all_outputs = first_batch_predictions.view(-1, d_model).numpy()
        
        # Create Hinton diagram
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Normalize values for better visualization
        max_val = np.abs(all_outputs).max()
        normalized_outputs = all_outputs / max_val if max_val > 0 else all_outputs
        
        # Plot squares with sizes proportional to values
        max_dims = min(50, normalized_outputs.shape[1])  # Limit dimensions for readability
        for i in range(normalized_outputs.shape[0]):
            for j in range(max_dims):
                val = normalized_outputs[i, j]
                color = 'red' if val > 0 else 'blue'
                size = abs(val) * 1000  # Scale for visibility
                
                ax.scatter(j, i, s=size, c=color, alpha=0.7, marker='s')
        
        ax.set_xlim(-0.5, max_dims - 0.5)
        ax.set_ylim(-0.5, all_outputs.shape[0] - 0.5)
        ax.set_xlabel('Feature Dimension')
        ax.set_ylabel('Output Index (Batch × Sequence)')
        ax.set_title(f'Hinton Diagram: All Outputs in First Batch\n'
                    f'({batch_size} samples × {seq_length} steps, showing first {max_dims} dimensions)')
        
        # Add legend
        ax.scatter([], [], c='red', s=100, label='Positive values', marker='s', alpha=0.7)
        ax.scatter([], [], c='blue', s=100, label='Negative values', marker='s', alpha=0.7)
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert y-axis for better readability
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'hinton_diagram_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_curves(self, metrics: Dict[str, Any], timestamp: str):
        """Plot performance curves: MSE and (1-cosine similarity) vs sequence length."""
        if 'per_step_mse_mean' not in metrics or 'per_step_cosine_distance_mean' not in metrics:
            return
            
        steps = range(len(metrics['per_step_mse_mean']))
        mse_mean = metrics['per_step_mse_mean']
        mse_std = metrics.get('per_step_mse_std', [0] * len(mse_mean))
        cosine_dist_mean = metrics['per_step_cosine_distance_mean']
        cosine_dist_std = metrics.get('per_step_cosine_distance_std', [0] * len(cosine_dist_mean))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # MSE curve
        ax1.plot(steps, mse_mean, 'b-', linewidth=2, label='MSE Loss', marker='o')
        ax1.fill_between(steps, 
                        np.array(mse_mean) - np.array(mse_std),
                        np.array(mse_mean) + np.array(mse_std),
                        alpha=0.3, color='blue', label='±1 std')
        ax1.set_xlabel('Sequence Step')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('MSE Loss vs Sequence Length\n(Average across Batches)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Cosine distance curve
        ax2.plot(steps, cosine_dist_mean, 'r-', linewidth=2, label='1 - Cosine Similarity', marker='s')
        ax2.fill_between(steps,
                        np.array(cosine_dist_mean) - np.array(cosine_dist_std),
                        np.array(cosine_dist_mean) + np.array(cosine_dist_std),
                        alpha=0.3, color='red', label='±1 std')
        ax2.set_xlabel('Sequence Step')
        ax2.set_ylabel('1 - Cosine Similarity')
        ax2.set_title('Cosine Distance vs Sequence Length\n(Average across Batches)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'performance_curves_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


# Dataset classes to match training format
class H5SequenceDataset(Dataset):
    """Dataset class matching the training format."""
    def __init__(self, h5_filepath: str, normalize_obs: bool = True):
        self.h5_filepath = h5_filepath
        self.normalize_obs = normalize_obs
        self.sequence_keys = []

        try:
            with h5py.File(h5_filepath, 'r') as f:
                keys = [key for key in f.keys() if key.startswith('sequence_')]
                keys.sort(key=lambda x: int(x.split('_')[1]))
                self.sequence_keys = keys
        except Exception as e:
            print(f"Error reading H5 file {h5_filepath}: {e}")
    
    def __len__(self):
        return len(self.sequence_keys)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_filepath, 'r') as f:
            g = f[self.sequence_keys[idx]]
            obs = g['observations'][:]
            actions = g['actions'][:]
            
        sample = {
            'observations': torch.from_numpy(obs).float(),
            'actions': torch.from_numpy(actions).long()
        }
        
        if self.normalize_obs:
            sample['observations'] /= 255.0
            
        return sample

def create_validation_dataloader(data_dir: str, batch_size: int = 8, num_workers: int = 0, 
                               shuffle: bool = False, normalize_obs: bool = True):
    """Create validation dataloader matching training format."""
    h5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
    
    if not h5_files:
        raise FileNotFoundError(f"Directory '{data_dir}' contains no .h5 files.")
    
    # Create dataset
    datasets = [H5SequenceDataset(fp, normalize_obs) for fp in h5_files]
    full_dataset = ConcatDataset(datasets)
    
    # Create dataloader
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader

def create_mock_dataloader(batch_size=4, seq_length=32, num_batches=5, normalize=True):
    """Create a mock dataloader matching training format for testing."""
    class MockDataset:
        def __init__(self, num_batches, batch_size, seq_length):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.seq_length = seq_length
        
        def __len__(self):
            return self.num_batches
        
        def __getitem__(self, idx):
            # Match training format exactly
            observations = torch.randn(self.batch_size, self.seq_length, 84, 84)
            actions = torch.randint(0, 18, (self.batch_size, self.seq_length))
            
            if normalize:
                observations = torch.clamp(observations, 0, 1)  # Simulate normalized observations
            
            return {
                'observations': observations,
                'actions': actions
            }
    
    dataset = MockDataset(num_batches, batch_size, seq_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def run_validation_example(model, dataloader=None):
    """
    Example function to run the validation framework.
    
    Args:
        model: Trained PWM model
        dataloader: DataLoader with validation data (optional, will create mock data if None)
    """
    # Create validator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    validator = PWMValidator(model, device=device, save_dir='validation_results')
    
    # Use provided dataloader or create mock data
    if dataloader is None:
        print("Creating mock dataloader for testing...")
        dataloader = create_mock_dataloader(batch_size=4, seq_length=8, num_batches=3)
    
    # Run validation
    results = validator.run_validation(dataloader, num_batches=3)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        elif isinstance(value, list) and len(value) <= 10:
            print(f"{key}: {[f'{v:.4f}' for v in value]}")
    
    return results

if __name__ == "__main__":
    print("PWM Validation Framework (Final)")
    print("Data format: observations (B, seq_length, 84, 84), actions (B, seq_length)")
    print("Use run_validation_example(model, dataloader) to test with your trained model.")