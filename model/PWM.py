# PWM.py
# Predictive World Model
# -----------------------------------------------------------------------------
# This file integrates three components into a unified Predictive World Model (PWM):
# 1. ImgEncoder: A convolutional encoder for processing sequences of images.
# 2. ActionEncoder: An MLP-based encoder for processing sequences of actions.
# 3. PRM (Predictive Recurrent Model): A Transformer-based model that predicts
#    the next image representation based on past images and actions.
#
# The model is designed for flexibility with unified configuration presets
# and command-line customizability.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from typing import Optional, Tuple, Literal, Union, Dict, Any

# ==============================================================================
# 1. UNIFIED CONFIGURATION
# ==============================================================================

def get_config(config_name: str = "normal") -> Dict[str, Any]:
    """
    Provides unified configuration for the entire PWM model.

    Args:
        config_name: The name of the configuration preset.
            - "light": Lightweight model for fast prototyping.
            - "normal": Balanced model, recommended for general use.
            - "large": Larger model for potentially higher performance.
            - "user_setting": A placeholder for custom command-line overrides.

    Returns:
        A dictionary containing parameters for all sub-modules.
    """
    if config_name not in ["light", "normal", "large", "user_setting"]:
        print(f"Warning: Unknown config name '{config_name}'. Falling back to 'normal'.")
        config_name = "normal"

    # Base d_model for each config
    d_models = {"light": 256, "normal": 512, "large": 768}
    d_model = d_models.get(config_name, 512)

    # --- All configurations defined here ---
    configs = {
        "light": {
            # General
            "d_model": d_model,
            "dropout": 0.1,
            "seq_length": 32,
            "pos_encoding_type": "learnable",
            "loss_type": "combined",
            # Image Encoder
            "img_encoder_blocks": 2,
            "img_encoder_groups": 4,
            # Action Encoder (d_model is shared)
            "action_encoder_hidden_dim": d_model,
            # Transformer
            "transformer_layers": 4,
            "transformer_heads": 4,
            "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0 # 方差正则化权重
        },
        "normal": {
            # General
            "d_model": d_model,
            "dropout": 0.1,
            "seq_length": 32,
            "pos_encoding_type": "learnable",
            "loss_type": "combined",
            # Image Encoder
            "img_encoder_blocks": 4,
            "img_encoder_groups": 8,
            # Action Encoder
            "action_encoder_hidden_dim": d_model,
            # Transformer
            "transformer_layers": 6,
            "transformer_heads": 8,
            "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0 # 方差正则化权重
        },
        "large": {
            # General
            "d_model": d_model,
            "dropout": 0.1,
            "seq_length": 32,
            "pos_encoding_type": "learnable",
            "loss_type": "combined",
            # Image Encoder
            "img_encoder_blocks": 5,
            "img_encoder_groups": 16,
            # Action Encoder
            "action_encoder_hidden_dim": d_model,
            # Transformer
            "transformer_layers": 8,
            "transformer_heads": 12,
            "transformer_d_ff": d_model * 4,
            "variance_weight": 0.0 # 方差正则化权重
        },
    }
    
    # "user_setting" starts with "normal" and gets overridden by CLI args
    if config_name == "user_setting":
        return configs["normal"]
        
    return configs.get(config_name, configs["normal"])


# ==============================================================================
# 2. HELPER & ENCODER MODULES 
# ==============================================================================

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Convolutional Block with GroupNorm ---
class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm, ReLU, optional SE, and residual connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_se: bool = True, num_groups: int = 8):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # Ensure num_groups is valid
        num_groups = min(num_groups, out_channels)
        while num_groups > 0 and out_channels % num_groups != 0:
            num_groups -= 1
        if num_groups == 0: num_groups = 1 # Fallback
            
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else None
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_groups = min(num_groups, out_channels)
            while shortcut_groups > 0 and out_channels % shortcut_groups != 0:
                shortcut_groups -= 1
            if shortcut_groups == 0: shortcut_groups = 1 # Fallback

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(shortcut_groups, out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm(self.conv(x)))
        if self.se:
            out = self.se(out)
        out += identity
        return self.relu(out)

# --- Atari Image Encoder ---
class AtariImgEncoder(nn.Module):
    """Encodes a sequence of Atari frames into feature vectors."""
    def __init__(self, d_model: int = 512, num_blocks: int = 4, 
                 use_se: bool = True, num_groups: int = 8):
        super().__init__()
        self.d_model = d_model
        
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),
            nn.GroupNorm(min(num_groups, 32), 32),
            nn.ReLU(inplace=True)
        )
        
        channels = [32, 64, 128, 256, 512]
        strides = [2, 2, 2, 1]
        self.blocks = nn.ModuleList()
        in_channels = 32
        
        for i in range(num_blocks):
            out_channels = channels[min(i + 1, len(channels) - 1)]
            stride = strides[min(i, len(strides) - 1)]
            self.blocks.append(ConvBlock(in_channels, out_channels, stride=stride, use_se=use_se, num_groups=num_groups))
            in_channels = out_channels
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(in_channels, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.shape[:2]
        x = x.view(batch_size * seq_length, 4, 84, 84)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).view(batch_size * seq_length, -1)
        x = self.output_proj(x)
        return x.view(batch_size, seq_length, self.d_model)

# --- MLP Action Encoder ---
class MLPActionEncoder(nn.Module):
    """Encodes a sequence of discrete actions using an MLP."""
    def __init__(self, num_actions: int = 18, d_model: int = 512, 
                 hidden_dim: Optional[int] = None, dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        hidden_dim = hidden_dim or d_model
        embed_dim = min(d_model // 2, 256)
        
        self.action_embedding = nn.Embedding(num_actions, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        embedded = self.action_embedding(actions)
        output = self.mlp(embedded)
        if self.layer_norm:
            output = self.layer_norm(output)
        return self.dropout(output)

# ==============================================================================
# 3. TRANSFORMER & PREDICTOR MODULES
# ==============================================================================

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """Adds positional information to input embeddings."""
    def __init__(self, d_model: int, max_seq_length: int = 512, 
                 encoding_type: Literal['learnable', 'sinusoidal'] = 'learnable', dropout: float = 0.1):
        super().__init__()
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'learnable':
            self.positional_embedding = nn.Embedding(max_seq_length, d_model)
            nn.init.normal_(self.positional_embedding.weight, std=0.02)
        elif encoding_type == 'sinusoidal':
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        if self.encoding_type == 'learnable':
            positions = torch.arange(start_pos, start_pos + seq_length, device=x.device, dtype=torch.long)
            pos_encoding = self.positional_embedding(positions).unsqueeze(0)
        else: # sinusoidal
            pos_encoding = self.pe[:, start_pos:start_pos + seq_length, :]
        
        x = x + pos_encoding.expand(batch_size, -1, -1)
        return self.dropout(x)

# --- Custom Transformer Block ---
class CustomTransformerBlock(nn.Module):
    """A custom Transformer block with self-attention and cross-attention."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, img_features: torch.Tensor, action_features: torch.Tensor, 
                self_attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image self-attention
        img_attended, _ = self.self_attention(img_features, img_features, img_features, attn_mask=self_attn_mask, need_weights=False)
        img_features = self.norm1(img_features + self.dropout(img_attended))
        
        # Cross-attention: action queries image
        cross_attended, cross_weights = self.cross_attention(action_features, img_features, img_features, need_weights=True)
        action_features = self.norm2(action_features + self.dropout(cross_attended))
        
        # FFN on action features
        ffn_output = self.ffn(action_features)
        output = self.norm3(action_features + self.dropout(ffn_output))
        
        return output, cross_weights

# --- Decoder-Only Transformer ---
class OptimizedDecoderOnlyTransformer(nn.Module):
    """The core predictive Transformer module."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float, seq_length: int, pos_encoding_type: str):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.img_pos_encoding = PositionalEncoding(d_model, seq_length + 10, pos_encoding_type, dropout)
        self.action_pos_encoding = PositionalEncoding(d_model, seq_length + 10, pos_encoding_type, dropout)
        self.transformer_blocks = nn.ModuleList([CustomTransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.register_buffer('_causal_mask', torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool(), persistent=False)
    
    def get_causal_mask(self, length: int) -> torch.Tensor:
        if self._causal_mask.size(0) < length:
            mask = torch.triu(torch.ones(length, length, device=self._causal_mask.device), diagonal=1).bool()
            self.register_buffer('_causal_mask', mask, persistent=False)
        return self._causal_mask[:length, :length]

    def forward(self, img_features: torch.Tensor, action_features: torch.Tensor,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        action_seq_len = action_features.shape[1]
        # 去掉最后一帧图像，最后一帧作为预测值
        img_input = img_features[:, :action_seq_len, :]
        
        img_input = self.img_pos_encoding(img_input)
        action_features = self.action_pos_encoding(action_features)
        
        causal_mask = self.get_causal_mask(action_seq_len)
        attention_weights = [] if return_attention_weights else None
        
        output = action_features
        for layer in self.transformer_blocks:
            output, cross_weights = layer(img_input, output, self_attn_mask=causal_mask)
            if return_attention_weights:
                attention_weights.append(cross_weights)
                
        output = self.final_norm(output)
        return (output, attention_weights) if return_attention_weights else output

# ==============================================================================
# 4. PWM - THE END-TO-END PREDICTIVE WORLD MODEL
# ==============================================================================
class PWM(nn.Module):
    """
    Predictive World Model (PWM).

    This end-to-end model integrates image and action encoders with a
    Transformer-based predictive core to forecast future world states.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.loss_type = config['loss_type']
        self.variance_weight = config.get('variance_weight', 0.001)
        
        # 1. Image Encoder
        self.img_encoder = AtariImgEncoder(
            d_model=self.d_model,
            num_blocks=config['img_encoder_blocks'],
            num_groups=config['img_encoder_groups']
        )
        
        # 2. Action Encoder (MLP type)
        self.action_encoder = MLPActionEncoder(
            num_actions=config.get('num_actions', 18),
            d_model=self.d_model,
            hidden_dim=config['action_encoder_hidden_dim'],
            dropout=config['dropout']
        )
        
        # 3. Predictive Transformer
        self.transformer = OptimizedDecoderOnlyTransformer(
            d_model=self.d_model,
            num_heads=config['transformer_heads'],
            d_ff=config['transformer_d_ff'],
            num_layers=config['transformer_layers'],
            dropout=config['dropout'],
            seq_length=config['seq_length'],
            pos_encoding_type=config['pos_encoding_type']
        )
        
        # 4. Output Projection Head
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.d_model // 2, self.d_model)
        )

    def forward(self, 
                observations: torch.Tensor, 
                actions: torch.Tensor
               ) -> Dict[str, torch.Tensor]:
        """
        【已修改】只执行前向传播，返回预测结果和目标，不计算损失。

        Args:
            observations: (B, S+1, 4, 84, 84)
            actions: (B, S)

        Returns:
            一个包含'predicted'和'target'的字典。
        """
        # Encode inputs
        img_features = self.img_encoder(observations)
        action_features = self.action_encoder(actions)
        
        # Predict with Transformer
        transformer_output = self.transformer(img_features, action_features)
            
        # Project to final prediction
        predicted = self.output_proj(transformer_output)
        
        # Target is the next image feature, detached to act as a label
        target = img_features[:, 1:, :].detach()

        results = {'predicted': predicted, 'target': target}
        return results

    def _compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失，并返回一个包含各项损失的字典。"""
        metrics = {}
        if self.loss_type == 'mse':
            main_loss = F.mse_loss(predicted, target)
            metrics['main_loss'] = main_loss
        elif self.loss_type == 'cos':
            main_loss = (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()
            metrics['main_loss'] = main_loss
        elif self.loss_type == 'combined':
            mse_loss = F.mse_loss(predicted, target)
            cos_loss = (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()
            main_loss = mse_loss + 0.1 * cos_loss
            metrics['main_loss'] = main_loss
            metrics['mse_loss'] = mse_loss
            metrics['cosine_loss'] = cos_loss
        else:
            main_loss = F.mse_loss(predicted, target)
            metrics['main_loss'] = main_loss

        total_loss = main_loss
        if self.variance_weight > 0:
            # 计算分子：预测输出的batch内方差
            pred_batch_var = torch.var(predicted, dim=0).mean()
            
            # 计算分母：目标特征的总方差 + batch内方差
            target_total_var = torch.var(target)
            target_batch_var = torch.var(target, dim=0).mean()
            
            # 为了数值稳定性，给分母加上一个极小值
            denominator = target_total_var + target_batch_var + 1e-7
            
            # 计算惩罚项
            diversity_penalty = pred_batch_var / denominator
        
        # 记录到metrics中
            metrics['diversity_penalty'] = diversity_penalty
            total_loss += self.variance_weight * diversity_penalty
        metrics['total_loss'] = total_loss
        return metrics

    @torch.no_grad()
    def eval_on_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Performs a single evaluation step on a batch of data."""
        self.eval()
        observations, actions = batch
        
        results = self.forward(observations, actions)
        predicted, target = results['predicted'], results['target']

        loss_dict = self._compute_loss(predicted, target)
        # Calculate comprehensive evaluation metrics
        loss_dict['eval_mse'] = F.mse_loss(predicted, target).item()
        loss_dict['eval_l1'] = F.l1_loss(predicted, target).item()
        loss_dict['var_target'] = torch.var(target, dim=0).mean().item()
        loss_dict['var_predicted'] = torch.var(predicted, dim=0).mean().item()
        loss_dict['eval_cosine_similarity'] = F.cosine_similarity(predicted, target, dim=-1).mean().item()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

# ==============================================================================
# 5. MAIN EXECUTION BLOCK & CLI
# ==============================================================================

# def main():
#     """Main function to parse arguments, create the model, and run a test."""
#     parser = argparse.ArgumentParser(description="Predictive World Model (PWM) Training and Testing")
    
#     # --- Primary Config ---
#     parser.add_argument('--config', type=str, default='normal', choices=['light', 'normal', 'large'],
#                         help='Model configuration preset.')
    
#     # --- User Setting Overrides ---
#     # Use 'user_setting' config if any of these are provided
#     parser.add_argument('--d_model', type=int, help='Override: Master dimension for all model parts.')
#     parser.add_argument('--seq_length', type=int, help='Override: Sequence length for training.')
#     parser.add_argument('--img_encoder_blocks', type=int, help='Override: Number of blocks in the image encoder.')
#     parser.add_argument('--transformer_layers', type=int, help='Override: Number of layers in the Transformer.')
#     parser.add_argument('--transformer_heads', type=int, help='Override: Number of attention heads in the Transformer.')

#     args = parser.parse_args()
    
#     # Check if user is providing custom settings
#     user_overrides = {
#         key: val for key, val in vars(args).items() 
#         if key != 'config' and val is not None
#     }
    
#     if user_overrides:
#         config_name = "user_setting"
#         print("User settings detected. Overriding defaults from 'normal' config.")
#         config = get_config("normal")
#         config.update(user_overrides)
#     else:
#         config_name = args.config
#         config = get_config(config_name)

#     print(f"--- Initializing PWM with '{config_name}' configuration ---")
#     for key, val in config.items():
#         print(f"  {key}: {val}")
#     print("-" * 50)
    
#     # Create the PWM model
#     model = PWM(config)
    
#     # Print parameter count
#     param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Model created successfully.")
#     print(f"Total trainable parameters: {param_count:,}")
#     print("-" * 50)

#     # --- Test with mock data ---
#     print("--- Running a test with mock data ---")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     print(f"Using device: {device}")
    
#     batch_size = 4
#     seq_length = config['seq_length']
#     num_actions = config.get('num_actions', 18)
    
#     # Mock data
#     mock_observations = torch.randn(batch_size, seq_length + 1, 4, 84, 84, device=device)
#     mock_actions = torch.randint(0, num_actions, (batch_size, seq_length), device=device)
#     mock_batch = (mock_observations, mock_actions)
    
#     # Test forward pass
#     try:
#         results = model.forward(mock_observations, mock_actions)
#         print("Forward pass successful.")
#         print(f"  Input Observations: {mock_observations.shape}")
#         print(f"  Input Actions:      {mock_actions.shape}")
#         print(f"  Output Predicted:   {results['predicted'].shape}")
#         print(f"  Output Target:      {results['target'].shape}")
#         print(f"  Initial Loss:       {results['loss'].item():.4f}")
#     except Exception as e:
#         print(f"Error during forward pass: {e}")
#         return
        
#     print("-" * 50)
    
#     # --- Test train_on_batch and eval_on_batch ---
#     print("--- Testing train_on_batch and eval_on_batch ---")
    
#     # Test train_on_batch
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     try:
#         train_metrics = model.train_on_batch(mock_batch, optimizer)
#         print("train_on_batch successful.")
#         print(f"  Training Loss: {train_metrics['loss']:.4f}")
#         print(f"  Cosine Similarity: {train_metrics['cos_similarity']:.4f}")
#         print(f"  Grad Norm: {train_metrics['grad_norm']:.4f}")
#     except Exception as e:
#         print(f"Error during train_on_batch: {e}")

#     # Test eval_on_batch
#     try:
#         eval_metrics = model.eval_on_batch(mock_batch)
#         print("\neval_on_batch successful.")
#         print(f"  Evaluation Loss: {eval_metrics['loss']:.4f}")
#         print(f"  Variance Explained: {eval_metrics['variance_explained']:.4f}")
#         print(f"  Accuracy (>0.9): {eval_metrics['accuracy_09']:.4f}")
#     except Exception as e:
#         print(f"Error during eval_on_batch: {e}")
        
#     print("-" * 50)
#     print("PWM model integration and testing complete.")

# if __name__ == "__main__":
#     main()