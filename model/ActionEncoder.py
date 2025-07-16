import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

class SimpleActionEncoder(nn.Module):
    """简单的动作编码器 - 基础Embedding方案"""
    
    def __init__(self, num_actions: int = 18, d_model: int = 512, dropout: float = 0.1):
        """
        Args:
            num_actions: 动作空间大小 (Atari通常是18)
            d_model: 输出特征维度
            dropout: dropout概率
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_actions = num_actions
        
        # 动作嵌入
        self.action_embedding = nn.Embedding(num_actions, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.normal_(self.action_embedding.weight, std=0.02)
    
    def forward(self, actions):
        """
        Args:
            actions: (batch_size, seq_length) - 动作ID序列
        Returns:
            (batch_size, seq_length, d_model)
        """
        # 嵌入动作
        embedded = self.action_embedding(actions)  # (batch_size, seq_length, d_model)
        embedded = self.dropout(embedded)
        
        return embedded

class MLPActionEncoder(nn.Module):
    """MLP增强的动作编码器 - 推荐方案"""
    
    def __init__(self, num_actions: int = 18, d_model: int = 512, 
                 hidden_dim: Optional[int] = None, dropout: float = 0.1,
                 use_layer_norm: bool = True):
        """
        Args:
            num_actions: 动作空间大小
            d_model: 输出特征维度
            hidden_dim: 隐藏层维度 (默认为d_model)
            dropout: dropout概率
            use_layer_norm: 是否使用LayerNorm
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_actions = num_actions
        hidden_dim = hidden_dim or d_model
        
        # 动作嵌入到较小维度
        embed_dim = min(d_model // 2, 256)
        self.action_embedding = nn.Embedding(num_actions, embed_dim)
        
        # MLP投影到目标维度
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        
        # 可选的LayerNorm
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 动作嵌入初始化
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        
        # MLP初始化
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, actions):
        """
        Args:
            actions: (batch_size, seq_length) - 动作ID序列
        Returns:
            (batch_size, seq_length, d_model)
        """
        # 嵌入
        embedded = self.action_embedding(actions)  # (batch_size, seq_length, embed_dim)
        
        # MLP投影
        output = self.mlp(embedded)  # (batch_size, seq_length, d_model)
        
        # LayerNorm
        if self.layer_norm:
            output = self.layer_norm(output)
        
        # Dropout
        output = self.dropout(output)
        
        return output

class PositionalActionEncoder(nn.Module):
    """带位置编码的动作编码器 - 考虑时序信息"""
    
    def __init__(self, num_actions: int = 18, d_model: int = 512, 
                 max_seq_length: int = 512, dropout: float = 0.1):
        """
        Args:
            num_actions: 动作空间大小
            d_model: 输出特征维度
            max_seq_length: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_actions = num_actions
        
        # 动作嵌入
        self.action_embedding = nn.Embedding(num_actions, d_model)
        
        # 位置编码
        self.register_buffer('positional_encoding', 
                           self._create_positional_encoding(max_seq_length, d_model))
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def _initialize_weights(self):
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        
        for module in self.output_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, actions):
        """
        Args:
            actions: (batch_size, seq_length) - 动作ID序列
        Returns:
            (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = actions.shape
        
        # 动作嵌入
        embedded = self.action_embedding(actions)  # (batch_size, seq_length, d_model)
        
        # 添加位置编码
        pos_encoding = self.positional_encoding[:, :seq_length, :]
        embedded = embedded + pos_encoding
        
        # 投影和归一化
        output = self.output_proj(embedded)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output

class LightweightTransformerActionEncoder(nn.Module):
    """轻量级Transformer动作编码器 - 最强表达能力"""
    
    def __init__(self, num_actions: int = 18, d_model: int = 512, 
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            num_actions: 动作空间大小
            d_model: 输出特征维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: dropout概率
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_actions = num_actions
        
        # 动作嵌入
        self.action_embedding = nn.Embedding(num_actions, d_model)
        
        # 位置编码
        self.register_buffer('positional_encoding', 
                           self._create_positional_encoding(512, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _initialize_weights(self):
        nn.init.normal_(self.action_embedding.weight, std=0.02)
    
    def forward(self, actions):
        """
        Args:
            actions: (batch_size, seq_length) - 动作ID序列
        Returns:
            (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = actions.shape
        
        # 动作嵌入
        embedded = self.action_embedding(actions) * math.sqrt(self.d_model)
        
        # 添加位置编码
        pos_encoding = self.positional_encoding[:, :seq_length, :]
        embedded = embedded + pos_encoding
        embedded = self.dropout(embedded)
        
        # Transformer编码
        output = self.transformer(embedded)
        
        return output

def create_action_encoder(encoder_type: str = "mlp", num_actions: int = 18, 
                         d_model: int = 512, **kwargs):
    """
    便捷的动作编码器创建函数
    
    Args:
        encoder_type: 编码器类型
            - "simple": 简单嵌入
            - "mlp": MLP增强 (推荐)
            - "positional": 带位置编码
            - "transformer": 轻量级Transformer
        num_actions: 动作空间大小
        d_model: 输出维度
        **kwargs: 其他参数
    """
    
    encoders = {
        "simple": SimpleActionEncoder,
        "mlp": MLPActionEncoder,
        "positional": PositionalActionEncoder,
        "transformer": LightweightTransformerActionEncoder
    }
    
    if encoder_type not in encoders:
        print(f"未知类型 '{encoder_type}'，使用默认 'mlp'")
        encoder_type = "mlp"
    
    encoder_class = encoders[encoder_type]
    encoder = encoder_class(num_actions=num_actions, d_model=d_model, **kwargs)
    
    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"创建 {encoder_type} 动作编码器:")
    print(f"  动作空间: {num_actions}")
    print(f"  输出维度: {d_model}")
    print(f"  参数量: {param_count:,}")
    
    return encoder

# 使用示例和测试
if __name__ == "__main__":
    print("=== ActionEncoder设计方案测试 ===")
    
    # 模拟数据
    batch_size = 4
    seq_length = 32  # 你的序列长度
    num_actions = 18  # Atari动作空间
    d_model = 512
    
    # 创建模拟动作序列
    actions = torch.randint(0, num_actions, (batch_size, seq_length))
    print(f"输入动作形状: {actions.shape}")
    print(f"动作范围: [{actions.min()}, {actions.max()}]")
    print()
    
    # 测试不同编码器
    encoder_types = ["simple", "mlp", "positional", "transformer"]
    
    for encoder_type in encoder_types:
        print(f"=== 测试 {encoder_type.upper()} 编码器 ===")
        
        try:
            encoder = create_action_encoder(
                encoder_type=encoder_type,
                num_actions=num_actions,
                d_model=d_model
            )
            
            # 前向传播
            with torch.no_grad():
                output = encoder(actions)
            
            print(f"  输出形状: {output.shape}")
            print(f"  输出统计: mean={output.mean():.4f}, std={output.std():.4f}")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
        
        print()
    
    print("=== 推荐使用方案 ===")
    recommendations = {
        "快速原型": "simple",
        "日常使用": "mlp", 
        "需要时序": "positional",
        "最强性能": "transformer"
    }
    
    for use_case, rec_type in recommendations.items():
        print(f"{use_case}: create_action_encoder('{rec_type}')")
    
    print("\n=== 与ImgEncoder配合使用示例 ===")
    print("""
# 在训练循环中
img_encoder = create_encoder("balanced")      # (batch, 33, 4, 84, 84) -> (batch, 33, 512)
action_encoder = create_action_encoder("mlp") # (batch, 32) -> (batch, 32, 512)

for batch in dataloader:
    observations = batch['observations']  # (batch, 33, 4, 84, 84)
    actions = batch['actions']           # (batch, 32)
    
    # 编码
    obs_features = img_encoder(observations)    # (batch, 33, 512)
    action_features = action_encoder(actions)   # (batch, 32, 512)
    
    # 对齐：取观察的前32步与动作对应
    state_features = obs_features[:, :-1, :]    # (batch, 32, 512)
    next_state_features = obs_features[:, 1:, :] # (batch, 32, 512)
    
    # 现在三者维度一致，可以进行联合处理
    # combined = torch.cat([state_features, action_features], dim=-1)
    """)