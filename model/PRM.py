import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal, Union

class PositionalEncoding(nn.Module):
    """
    位置编码模块，支持可学习和固定两种模式
    """
    
    def __init__(self, 
                 d_model: int, 
                 max_seq_length: int = 512, 
                 encoding_type: Literal['learnable', 'sinusoidal'] = 'learnable',
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'learnable':
            # 可学习的位置编码
            self.positional_embedding = nn.Embedding(max_seq_length, d_model)
            # 初始化
            nn.init.normal_(self.positional_embedding.weight, std=0.02)
            
        elif encoding_type == 'sinusoidal':
            # 固定的正弦位置编码
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_length, d_model)
        
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_length, d_model) - 输入特征
            start_pos: 起始位置（用于处理不同长度的序列）
        
        Returns:
            (batch_size, seq_length, d_model) - 添加位置编码后的特征
        """
        batch_size, seq_length, d_model = x.shape
        
        if self.encoding_type == 'learnable':
            # 生成位置索引
            positions = torch.arange(start_pos, start_pos + seq_length, 
                                   device=x.device, dtype=torch.long)
            pos_encoding = self.positional_embedding(positions)  # (seq_length, d_model)
            pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            
        elif self.encoding_type == 'sinusoidal':
            pos_encoding = self.pe[:, start_pos:start_pos + seq_length, :]
            pos_encoding = pos_encoding.expand(batch_size, -1, -1)
        
        # 添加位置编码
        x = x + pos_encoding
        return metrics
    
    def eval_on_batch(self, 
                     observations: torch.Tensor,
                     actions: torch.Tensor,
                     return_attention: bool = False,
                     compute_detailed_metrics: bool = True) -> dict:
        """
        在单个batch上评估模型
        
        Args:
            observations: (batch_size, seq_length+1, 4, 84, 84) - 观察序列
            actions: (batch_size, seq_length) - 动作序列
            return_attention: 是否返回注意力权重
            compute_detailed_metrics: 是否计算详细指标
        
        Returns:
            dict: 包含评估指标的字典
        """
        self.eval()
        
        with torch.no_grad():
            # 前向传播
            results = self.forward(observations, actions, return_attention=return_attention)
            loss = results['loss']
            predicted = results['predicted']
            target = results['target']
            
            # 基础指标
            metrics = {
                'loss': loss.item(),
                'batch_size': observations.shape[0]
            }
            
            if compute_detailed_metrics:
                # MSE误差
                mse_loss = F.mse_loss(predicted, target)
                
                # L1误差
                l1_loss = F.l1_loss(predicted, target)
                
                # 余弦相似度
                cos_sim = F.cosine_similarity(predicted, target, dim=-1)
                cos_sim_mean = cos_sim.mean()
                cos_sim_std = cos_sim.std()
                
                # 预测准确性（基于余弦相似度阈值）
                accuracy_05 = (cos_sim > 0.5).float().mean()
                accuracy_07 = (cos_sim > 0.7).float().mean()
                accuracy_09 = (cos_sim > 0.9).float().mean()
                
                # 相对误差
                relative_error = ((predicted - target).abs() / (target.abs() + 1e-8)).mean()
                
                # 方差解释比例 (类似R²)
                target_var = target.var()
                residual_var = (predicted - target).var()
                variance_explained = 1 - (residual_var / (target_var + 1e-8))
                
                metrics.update({
                    'mse_loss': mse_loss.item(),
                    'l1_loss': l1_loss.item(),
                    'cos_similarity_mean': cos_sim_mean.item(),
                    'cos_similarity_std': cos_sim_std.item(),
                    'accuracy_05': accuracy_05.item(),
                    'accuracy_07': accuracy_07.item(),
                    'accuracy_09': accuracy_09.item(),
                    'relative_error': relative_error.item(),
                    'variance_explained': variance_explained.item()
                })
            
            if return_attention:
                metrics['attention_weights'] = results.get('attention_weights')
        
        return metrics
    
    def predict_next_features(self, 
                             observations: torch.Tensor,
                             actions: torch.Tensor,
                             return_attention: bool = False) -> dict:
        """
        预测下一个时间步的特征（推理模式）
        
        Args:
            observations: (batch_size, seq_length+1, 4, 84, 84)
            actions: (batch_size, seq_length)
            return_attention: 是否返回注意力权重
        
        Returns:
            dict: 包含预测结果的字典
        """
        self.eval()
        
        with torch.no_grad():
            results = self.forward(observations, actions, return_attention=return_attention)
            
            # 只返回预测结果，不计算损失
            output = {
                'predicted': results['predicted'],
                'batch_size': observations.shape[0]
            }
            
            if return_attention:
                output['attention_weights'] = results.get('attention_weights')
            
            return output self.dropout(x)

class CustomTransformerBlock(nn.Module):
    """
    自定义Transformer块，使用官方组件但支持灵活的attention配置
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 使用官方的MultiheadAttention - 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 重要：使用batch_first=True
        )
        
        # 使用官方的MultiheadAttention - 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 使用官方的LayerNorm和Linear
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 使用官方的FFN结构
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                img_features: torch.Tensor,
                action_features: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor] = None,
                cross_attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            img_features: (batch_size, seq_length, d_model) - 图像特征
            action_features: (batch_size, seq_length, d_model) - 动作特征
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
        
        Returns:
            (batch_size, seq_length, d_model) - 输出特征
        """
        # 1. 图像特征的自注意力（带因果掩码）
        img_attended, _ = self.self_attention(
            query=img_features,
            key=img_features,
            value=img_features,
            attn_mask=self_attn_mask,
            need_weights=False  # 提升性能
        )
        img_features = self.norm1(img_features + self.dropout(img_attended))
        
        # 2. 交叉注意力：动作查询图像
        cross_attended, cross_weights = self.cross_attention(
            query=action_features,      # action作为query
            key=img_features,          # img作为key
            value=img_features,        # img作为value
            attn_mask=cross_attn_mask,
            need_weights=True  # 可能需要分析注意力模式
        )
        action_features = self.norm2(action_features + self.dropout(cross_attended))
        
        # 3. FFN
        ffn_output = self.ffn(action_features)
        output = self.norm3(action_features + self.dropout(ffn_output))
        
        return output, cross_weights

class OptimizedDecoderOnlyTransformer(nn.Module):
    """
    优化的Decoder-Only Transformer，使用官方组件，包含位置编码
    """
    
    def __init__(self, 
                 d_model: int = 512,
                 num_heads: int = 8, 
                 d_ff: int = 2048,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 seq_length: int = 32,
                 pos_encoding_type: str = 'learnable'):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # 位置编码模块
        self.img_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=seq_length + 10,  # 留一些余量
            encoding_type=pos_encoding_type,
            dropout=dropout
        )
        
        self.action_pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=seq_length + 10,
            encoding_type=pos_encoding_type,
            dropout=dropout
        )
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 注册掩码缓冲区（支持动态修改）
        self.register_buffer('_causal_mask', torch.empty(0))
        self._init_causal_mask(seq_length)
    
    def _init_causal_mask(self, seq_length: int):
        """初始化因果掩码"""
        # 创建下三角掩码
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        self.register_buffer('_causal_mask', mask, persistent=False)
    
    def set_causal_mask(self, mask: torch.Tensor):
        """动态设置因果掩码"""
        self.register_buffer('_causal_mask', mask, persistent=False)
    
    def get_causal_mask(self, seq_length: int) -> torch.Tensor:
        """获取适当大小的因果掩码"""
        if self._causal_mask.size(0) < seq_length:
            self._init_causal_mask(seq_length)
        return self._causal_mask[:seq_length, :seq_length]
    
    def forward(self, 
                img_features: torch.Tensor, 
                action_features: torch.Tensor,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Args:
            img_features: (batch_size, seq_length+1, d_model) - 原始图像特征
            action_features: (batch_size, seq_length, d_model) - 原始动作特征
            return_attention_weights: 是否返回注意力权重
        
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: (可选) 每层的交叉注意力权重
        """
        batch_size, img_seq_len, _ = img_features.shape
        _, action_seq_len, _ = action_features.shape
        
        # 取前seq_length个图像特征
        img_input = img_features[:, :action_seq_len, :]  # (batch_size, seq_length, d_model)
        
        # 添加位置编码
        img_input = self.img_pos_encoding(img_input, start_pos=0)
        action_features = self.action_pos_encoding(action_features, start_pos=0)
        
        # 获取因果掩码
        causal_mask = self.get_causal_mask(action_seq_len)
        
        # 存储注意力权重
        attention_weights = [] if return_attention_weights else None
        
        # 通过所有Transformer块
        output = action_features
        for layer in self.transformer_blocks:
            output, cross_weights = layer(
                img_features=img_input,
                action_features=output,
                self_attn_mask=causal_mask,
                cross_attn_mask=None  # 交叉注意力通常不需要掩码
            )
            
            if return_attention_weights:
                attention_weights.append(cross_weights)
        
        # 最终归一化
        output = self.final_norm(output)
        
        if return_attention_weights:
            return output, attention_weights
        return output

class SimplePredictor(nn.Module):
    """
    简化的预测器，直接使用detach策略获取目标
    """
    
    def __init__(self, transformer: OptimizedDecoderOnlyTransformer):
        super().__init__()
        
        self.transformer = transformer
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(transformer.d_model, transformer.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(transformer.d_model // 2, transformer.d_model)
        )
    
    def forward(self, 
                img_features: torch.Tensor,
                action_features: torch.Tensor,
                return_attention: bool = False) -> dict:
        """
        Args:
            img_features: (batch_size, seq_length+1, d_model)
            action_features: (batch_size, seq_length, d_model)
            return_attention: 是否返回注意力权重
        
        Returns:
            dict with 'predicted', 'target', and optionally 'attention_weights'
        """
        # 前向预测
        if return_attention:
            predicted, attention_weights = self.transformer(
                img_features, action_features, return_attention_weights=True
            )
        else:
            predicted = self.transformer(img_features, action_features)
            attention_weights = None
        
        # 输出投影
        predicted = self.output_proj(predicted)
        
        # 获取目标特征（直接detach，无梯度）
        target = img_features[:, 1:, :].detach()  # 后seq_length个图像特征，无梯度
        
        result = {
            'predicted': predicted,
            'target': target
        }
        
        if return_attention:
            result['attention_weights'] = attention_weights
        
        return result

class OptimizedEndToEndModel(nn.Module):
    """
    优化的端到端模型，包含位置编码，使用简化的预测器
    """
    
    def __init__(self, 
                 img_encoder: nn.Module,
                 action_encoder: nn.Module,
                 d_model: int = 512,
                 num_heads: int = 8,
                 d_ff: int = 2048,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 seq_length: int = 32,
                 loss_type: str = 'combined',
                 pos_encoding_type: str = 'learnable'):
        super().__init__()
        
        self.img_encoder = img_encoder
        self.action_encoder = action_encoder
        self.loss_type = loss_type
        self.seq_length = seq_length
        
        # 创建优化的transformer（包含位置编码）
        transformer = OptimizedDecoderOnlyTransformer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=seq_length,
            pos_encoding_type=pos_encoding_type
        )
        
        # 创建简化的预测器
        self.predictor = SimplePredictor(transformer=transformer)
    
    def forward(self, 
                observations: torch.Tensor,
                actions: torch.Tensor,
                return_attention: bool = False) -> dict:
        """
        Args:
            observations: (batch_size, seq_length+1, 4, 84, 84)
            actions: (batch_size, seq_length)
            return_attention: 是否返回注意力权重
        
        Returns:
            dict with 'predicted', 'target', 'loss', and optionally 'attention_weights'
        """
        # 编码
        img_features = self.img_encoder(observations)  # (batch_size, seq_length+1, d_model)
        action_features = self.action_encoder(actions)  # (batch_size, seq_length, d_model)
        
        # 预测（内部会添加位置编码）
        results = self.predictor(
            img_features=img_features,
            action_features=action_features,
            return_attention=return_attention
        )
        
        # 计算损失
        loss = self._compute_loss(results['predicted'], results['target'])
        results['loss'] = loss
        
        return results
    
    def _compute_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        if self.loss_type == 'mse':
            return F.mse_loss(predicted, target)
        elif self.loss_type == 'cosine':
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            return (1 - cos_sim).mean()
        elif self.loss_type == 'huber':
            return F.huber_loss(predicted, target)
        elif self.loss_type == 'combined':
            mse_loss = F.mse_loss(predicted, target)
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            cos_loss = (1 - cos_sim).mean()
            return mse_loss + 0.1 * cos_loss
        else:
            return F.mse_loss(predicted, target)
    
    def set_causal_mask(self, mask: torch.Tensor):
        """设置自定义因果掩码"""
        self.predictor.transformer.set_causal_mask(mask)
    
    def train_on_batch(self, 
                      observations: torch.Tensor,
                      actions: torch.Tensor,
                      optimizer: torch.optim.Optimizer,
                      max_grad_norm: Optional[float] = 1.0,
                      return_attention: bool = False) -> dict:
        """
        在单个batch上训练模型
        
        Args:
            observations: (batch_size, seq_length+1, 4, 84, 84) - 观察序列
            actions: (batch_size, seq_length) - 动作序列
            optimizer: 优化器
            max_grad_norm: 梯度裁剪的最大范数，None表示不裁剪
            return_attention: 是否返回注意力权重
        
        Returns:
            dict: 包含训练指标的字典
        """
        self.train()
        
        # 前向传播
        results = self.forward(observations, actions, return_attention=return_attention)
        loss = results['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = None
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        
        # 更新参数
        optimizer.step()
        
        # 计算额外的指标
        with torch.no_grad():
            predicted = results['predicted']
            target = results['target']
            
            # MSE误差
            mse_loss = F.mse_loss(predicted, target)
            
            # 余弦相似度
            cos_sim = F.cosine_similarity(predicted, target, dim=-1).mean()
            
            # L1误差
            l1_loss = F.l1_loss(predicted, target)
            
            # 预测准确性（基于余弦相似度）
            accuracy = (cos_sim > 0.5).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'cos_similarity': cos_sim.item(),
            'accuracy': accuracy.item(),
            'grad_norm': grad_norm.item() if grad_norm is not None else None,
            'batch_size': observations.shape[0]
        }
        
        if return_attention:
            metrics['attention_weights'] = results.get('attention_weights')
        
        return

# 训练辅助工具函数
class TrainingManager:
    """
    训练管理器，提供统一的训练接口
    """
    
    def __init__(self, 
                 model: OptimizedEndToEndModel,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 max_grad_norm: float = 1.0,
                 device: str = 'cuda'):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # 训练历史
        self.train_history = []
        self.eval_history = []
        self.step_count = 0
    
    def train_epoch(self, 
                   dataloader,
                   log_interval: int = 100,
                   return_attention_freq: int = 0) -> dict:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            log_interval: 日志打印间隔
            return_attention_freq: 返回注意力权重的频率（0表示不返回）
        
        Returns:
            dict: epoch训练统计
        """
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'mse_loss': 0.0,
            'l1_loss': 0.0,
            'cos_similarity': 0.0,
            'accuracy': 0.0,
            'grad_norm': 0.0,
            'num_batches': 0
        }
        
        attention_samples = []
        
        for batch_idx, (observations, actions) in enumerate(dataloader):
            # 移动数据到设备
            observations = observations.to(self.device)
            actions = actions.to(self.device)
            
            # 是否返回注意力权重
            return_attention = (return_attention_freq > 0 and 
                              batch_idx % return_attention_freq == 0)
            
            # 训练一个batch
            metrics = self.model.train_on_batch(
                observations=observations,
                actions=actions,
                optimizer=self.optimizer,
                max_grad_norm=self.max_grad_norm,
                return_attention=return_attention
            )
            
            # 累积指标
            for key in epoch_metrics:
                if key in metrics and metrics[key] is not None:
                    epoch_metrics[key] += metrics[key]
            epoch_metrics['num_batches'] += 1
            self.step_count += 1
            
            # 保存注意力权重样本
            if return_attention and 'attention_weights' in metrics:
                attention_samples.append({
                    'batch_idx': batch_idx,
                    'step': self.step_count,
                    'attention_weights': metrics['attention_weights']
                })
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 打印日志
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={metrics['loss']:.6f}, "
                      f"CosSim={metrics['cos_similarity']:.4f}, "
                      f"LR={current_lr:.2e}")
        
        # 计算平均指标
        for key in epoch_metrics:
            if key != 'num_batches':
                epoch_metrics[key] /= epoch_metrics['num_batches']
        
        epoch_metrics['attention_samples'] = attention_samples
        self.train_history.append(epoch_metrics)
        
        return epoch_metrics
    
    def evaluate(self, 
                dataloader,
                return_attention_freq: int = 0) -> dict:
        """
        评估模型
        
        Args:
            dataloader: 验证数据加载器
            return_attention_freq: 返回注意力权重的频率
        
        Returns:
            dict: 评估统计
        """
        self.model.eval()
        eval_metrics = {
            'loss': 0.0,
            'mse_loss': 0.0,
            'l1_loss': 0.0,
            'cos_similarity_mean': 0.0,
            'cos_similarity_std': 0.0,
            'accuracy_05': 0.0,
            'accuracy_07': 0.0,
            'accuracy_09': 0.0,
            'relative_error': 0.0,
            'variance_explained': 0.0,
            'num_batches': 0
        }
        
        attention_samples = []
        
        with torch.no_grad():
            for batch_idx, (observations, actions) in enumerate(dataloader):
                # 移动数据到设备
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # 是否返回注意力权重
                return_attention = (return_attention_freq > 0 and 
                                  batch_idx % return_attention_freq == 0)
                
                # 评估一个batch
                metrics = self.model.eval_on_batch(
                    observations=observations,
                    actions=actions,
                    return_attention=return_attention,
                    compute_detailed_metrics=True
                )
                
                # 累积指标
                for key in eval_metrics:
                    if key in metrics and metrics[key] is not None:
                        eval_metrics[key] += metrics[key]
                eval_metrics['num_batches'] += 1
                
                # 保存注意力权重样本
                if return_attention and 'attention_weights' in metrics:
                    attention_samples.append({
                        'batch_idx': batch_idx,
                        'attention_weights': metrics['attention_weights']
                    })
        
        # 计算平均指标
        for key in eval_metrics:
            if key != 'num_batches':
                eval_metrics[key] /= eval_metrics['num_batches']
        
        eval_metrics['attention_samples'] = attention_samples
        self.eval_history.append(eval_metrics)
        
        return eval_metrics
    
    def save_checkpoint(self, filepath: str, epoch: int, best_metric: float = None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'train_history': self.train_history,
            'eval_history': self.eval_history,
            'best_metric': best_metric
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> dict:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        self.train_history = checkpoint.get('train_history', [])
        self.eval_history = checkpoint.get('eval_history', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"检查点已从 {filepath} 加载")
        return checkpoint

def create_training_manager(model: OptimizedEndToEndModel,
                           learning_rate: float = 1e-4,
                           weight_decay: float = 1e-5,
                           scheduler_type: str = 'cosine',
                           max_grad_norm: float = 1.0,
                           device: str = 'cuda',
                           **scheduler_kwargs) -> TrainingManager:
    """
    创建训练管理器的便捷函数
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        scheduler_type: 调度器类型 ('cosine', 'step', 'plateau', None)
        max_grad_norm: 梯度裁剪范数
        device: 设备
        **scheduler_kwargs: 调度器参数
    
    Returns:
        TrainingManager: 训练管理器
    """
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 创建学习率调度器
    scheduler = None
    if scheduler_type == 'cosine':
        T_max = scheduler_kwargs.get('T_max', 1000)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    elif scheduler_type == 'step':
        step_size = scheduler_kwargs.get('step_size', 100)
        gamma = scheduler_kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
    
    return TrainingManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=max_grad_norm,
        device=device
    )
def create_optimized_model(img_encoder: nn.Module,
                          action_encoder: nn.Module,
                          config: str = "standard",
                          pos_encoding_type: str = "learnable",
                          **kwargs) -> OptimizedEndToEndModel:
    """
    创建优化的模型
    
    Args:
        img_encoder: 图像编码器
        action_encoder: 动作编码器
        config: 配置名称 ("lightweight", "standard", "large")
        pos_encoding_type: 位置编码类型 ("learnable", "sinusoidal")
        **kwargs: 其他参数
    """
    configs = {
        "lightweight": {
            "d_model": 256, "num_heads": 8, "d_ff": 1024,
            "num_layers": 4, "dropout": 0.1
        },
        "standard": {
            "d_model": 512, "num_heads": 8, "d_ff": 2048,
            "num_layers": 6, "dropout": 0.1
        },
        "large": {
            "d_model": 768, "num_heads": 12, "d_ff": 3072,
            "num_layers": 8, "dropout": 0.1
        }
    }
    
    config_params = configs.get(config, configs["standard"])
    config_params.update(kwargs)
    config_params['pos_encoding_type'] = pos_encoding_type
    
    model = OptimizedEndToEndModel(
        img_encoder=img_encoder,
        action_encoder=action_encoder,
        **config_params
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"创建简化模型 ({config}):")
    print(f"  参数量: {total_params:,}")
    print(f"  位置编码: {pos_encoding_type}")
    print(f"  目标策略: detach (无梯度)")
    print(f"  使用官方MultiheadAttention: ✅")
    print(f"  支持自定义掩码: ✅")
    
    return model

# 使用示例
if __name__ == "__main__":
    print("=== 简化的Transformer模型测试（包含训练功能）===")
    
    # 创建模拟编码器
    class MockImgEncoder(nn.Module):
        def __init__(self, d_model=512):
            super().__init__()
            self.conv = nn.Conv2d(4, 64, 3, 2, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, d_model)
        
        def forward(self, x):
            b, s = x.shape[:2]
            x = x.view(b*s, 4, 84, 84)
            x = self.pool(self.conv(x)).view(b*s, 64)
            x = self.fc(x)
            return x.view(b, s, -1)
    
    class MockActionEncoder(nn.Module):
        def __init__(self, d_model=512):
            super().__init__()
            self.emb = nn.Embedding(18, d_model)
        
        def forward(self, x):
            return self.emb(x)
    
    # 创建模拟数据加载器
    class MockDataset:
        def __init__(self, num_samples=100, seq_length=32):
            self.num_samples = num_samples
            self.seq_length = seq_length
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            observations = torch.randn(self.seq_length + 1, 4, 84, 84)
            actions = torch.randint(0, 18, (self.seq_length,))
            return observations, actions
    
    def create_dataloader(dataset, batch_size=4, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    
    # 创建编码器和模型
    img_encoder = MockImgEncoder(512)
    action_encoder = MockActionEncoder(512)
    
    print(f"\n=== 测试基础功能 ===")
    
    # 创建简化模型
    model = create_optimized_model(
        img_encoder=img_encoder,
        action_encoder=action_encoder,
        config="standard",
        seq_length=32,
        pos_encoding_type="learnable"
    )
    
    # 测试数据
    batch_size = 4
    seq_length = 32
    observations = torch.randn(batch_size, seq_length + 1, 4, 84, 84)
    actions = torch.randint(0, 18, (batch_size, seq_length))
    
    print(f"输入数据:")
    print(f"  观察: {observations.shape}")
    print(f"  动作: {actions.shape}")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        results = model(observations, actions, return_attention=True)
    
    print(f"输出结果:")
    print(f"  预测: {results['predicted'].shape}")
    print(f"  目标: {results['target'].shape}")
    print(f"  目标梯度信息: {results['target'].requires_grad}")
    print(f"  损失: {results['loss'].item():.6f}")
    
    print(f"\n=== 测试训练功能 ===")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 测试单batch训练
    print("测试train_on_batch:")
    train_metrics = model.train_on_batch(
        observations=observations,
        actions=actions,
        optimizer=optimizer,
        max_grad_norm=1.0,
        return_attention=False
    )
    
    print(f"  训练损失: {train_metrics['loss']:.6f}")
    print(f"  MSE损失: {train_metrics['mse_loss']:.6f}")
    print(f"  余弦相似度: {train_metrics['cos_similarity']:.4f}")
    print(f"  梯度范数: {train_metrics['grad_norm']:.6f}")
    print(f"  准确率: {train_metrics['accuracy']:.4f}")
    
    # 测试单batch评估
    print("\n测试eval_on_batch:")
    eval_metrics = model.eval_on_batch(
        observations=observations,
        actions=actions,
        return_attention=False,
        compute_detailed_metrics=True
    )
    
    print(f"  评估损失: {eval_metrics['loss']:.6f}")
    print(f"  MSE损失: {eval_metrics['mse_loss']:.6f}")
    print(f"  余弦相似度(均值): {eval_metrics['cos_similarity_mean']:.4f}")
    print(f"  余弦相似度(标准差): {eval_metrics['cos_similarity_std']:.4f}")
    print(f"  准确率(>0.5): {eval_metrics['accuracy_05']:.4f}")
    print(f"  准确率(>0.7): {eval_metrics['accuracy_07']:.4f}")
    print(f"  准确率(>0.9): {eval_metrics['accuracy_09']:.4f}")
    print(f"  相对误差: {eval_metrics['relative_error']:.6f}")
    print(f"  方差解释比例: {eval_metrics['variance_explained']:.4f}")
    
    # 测试预测功能
    print("\n测试predict_next_features:")
    pred_results = model.predict_next_features(
        observations=observations,
        actions=actions,
        return_attention=True
    )
    
    print(f"  预测形状: {pred_results['predicted'].shape}")
    print(f"  注意力权重层数: {len(pred_results['attention_weights'])}")
    
    print(f"\n=== 测试训练管理器 ===")
    
    # 创建训练管理器
    trainer = create_training_manager(
        model=model,
        learning_rate=1e-4,
        weight_decay=1e-5,
        scheduler_type='cosine',
        max_grad_norm=1.0,
        device='cpu',  # 使用CPU进行测试
        T_max=100
    )
    
    # 创建模拟数据
    train_dataset = MockDataset(num_samples=20, seq_length=32)
    val_dataset = MockDataset(num_samples=10, seq_length=32)
    
    train_loader = create_dataloader(train_dataset, batch_size=4, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=4, shuffle=False)
    
    print("训练数据加载器创建成功")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 测试一个epoch的训练
    print("\n训练一个epoch:")
    epoch_metrics = trainer.train_epoch(
        dataloader=train_loader,
        log_interval=2,
        return_attention_freq=0
    )
    
    print(f"Epoch训练结果:")
    print(f"  平均损失: {epoch_metrics['loss']:.6f}")
    print(f"  平均余弦相似度: {epoch_metrics['cos_similarity']:.4f}")
    print(f"  平均梯度范数: {epoch_metrics['grad_norm']:.6f}")
    
    # 测试评估
    print("\n评估模型:")
    eval_results = trainer.evaluate(
        dataloader=val_loader,
        return_attention_freq=0
    )
    
    print(f"评估结果:")
    print(f"  评估损失: {eval_results['loss']:.6f}")
    print(f"  余弦相似度: {eval_results['cos_similarity_mean']:.4f}")
    print(f"  准确率(>0.5): {eval_results['accuracy_05']:.4f}")
    print(f"  方差解释比例: {eval_results['variance_explained']:.4f}")
    
    # 测试检查点保存和加载
    print("\n测试检查点功能:")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        
        # 保存检查点
        trainer.save_checkpoint(
            filepath=checkpoint_path,
            epoch=1,
            best_metric=eval_results['loss']
        )
        
        # 加载检查点
        checkpoint_info = trainer.load_checkpoint(checkpoint_path)
        print(f"  检查点包含epoch: {checkpoint_info['epoch']}")
        print(f"  最佳指标: {checkpoint_info['best_metric']:.6f}")
    
    print(f"\n=== 性能对比测试 ===")
    
    # 测试不同配置的性能
    configs = ['lightweight', 'standard']
    
    for config in configs:
        print(f"\n测试 {config} 配置:")
        
        test_model = create_optimized_model(
            img_encoder=img_encoder,
            action_encoder=action_encoder,
            config=config,
            seq_length=32,
            pos_encoding_type="learnable"
        )
        
        # 测试前向传播时间
        import time
        
        test_model.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                _ = test_model(observations, actions)
            end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"  平均前向传播时间: {avg_time*1000:.2f} ms")
    
    print(f"\n✅ 所有测试通过！")
    print(f"📈 训练功能完整：train_on_batch, eval_on_batch, TrainingManager")
    print(f"💾 支持检查点保存和加载")
    print(f"📊 提供详细的训练和评估指标")
    print(f"🔧 支持多种优化器和学习率调度器")