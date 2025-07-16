import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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

class ConvBlock(nn.Module):
    """使用GroupNorm的卷积块 - 最适合序列图像数据"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_se: bool = True, num_groups: int = 8):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # 使用GroupNorm - 最适合序列数据
        # 确保group数能整除通道数
        num_groups = min(num_groups, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels) if use_se else None
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut_groups = min(num_groups, out_channels)
            while out_channels % shortcut_groups != 0:
                shortcut_groups -= 1
            
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(shortcut_groups, out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        
        if self.se:
            out = self.se(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class AtariImgEncoder(nn.Module):
    """
    最终推荐的Atari图像编码器
    - 使用GroupNorm保证序列独立性
    - 包含SE模块和ResNet结构
    - 适合 (batch_size, seq_length, 4, 84, 84) 输入
    """
    
    def __init__(self, d_model: int = 512, num_blocks: int = 4, 
                 use_se: bool = True, num_groups: int = 8):
        """
        Args:
            d_model: 输出特征维度
            num_blocks: 卷积块数量 (推荐2-5)
            use_se: 是否使用SE模块
            num_groups: GroupNorm的组数 (推荐8-16)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        # 初始卷积层: 84x84 -> 21x21
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),
            nn.GroupNorm(min(num_groups, 32), 32),
            nn.ReLU(inplace=True)
        )
        
        # 渐进式通道增加
        channels = [32, 64, 128, 256, 512]
        strides = [2, 2, 2, 1]  # 控制下采样: 21->10->5->2->2
        
        self.blocks = nn.ModuleList()
        in_channels = 32
        
        for i in range(num_blocks):
            out_channels = channels[min(i + 1, len(channels) - 1)]
            stride = strides[min(i, len(strides) - 1)]
            
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                use_se=use_se,
                num_groups=num_groups
            )
            self.blocks.append(block)
            in_channels = out_channels
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(in_channels, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, 4, 84, 84)
        Returns:
            (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = x.shape[:2]
        
        # 重塑为CNN输入格式
        x = x.view(batch_size * seq_length, 4, 84, 84)
        
        # 特征提取
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        # 池化和投影
        x = self.global_pool(x)
        x = x.view(batch_size * seq_length, -1)
        x = self.output_proj(x)
        
        # 恢复序列维度
        return x.view(batch_size, seq_length, self.d_model)

# 预设配置
def get_encoder_config(config_name: str = "balanced"):
    """
    获取预设配置
    
    Args:
        config_name: 配置名称
            - "fast": 快速实验 (少参数)
            - "balanced": 平衡配置 (推荐)
            - "high_quality": 高质量 (多参数)
            - "small_batch": 小batch训练
    """
    configs = {
        "fast": {
            "d_model": 256,
            "num_blocks": 2,
            "use_se": True,
            "num_groups": 4
        },
        "balanced": {
            "d_model": 512,
            "num_blocks": 4,
            "use_se": True,
            "num_groups": 8
        },
        "high_quality": {
            "d_model": 768,
            "num_blocks": 5,
            "use_se": True,
            "num_groups": 16
        },
        "small_batch": {
            "d_model": 512,
            "num_blocks": 4,
            "use_se": True,
            "num_groups": 4  # 小batch时用少一些groups
        }
    }
    
    if config_name not in configs:
        config_name = "balanced"
        print(f"未知配置，使用默认配置: {config_name}")
    
    return configs[config_name]

def create_encoder(config_name: str = "balanced", **kwargs):
    """便捷的编码器创建函数"""
    config = get_encoder_config(config_name)
    config.update(kwargs)  # 允许覆盖配置
    
    encoder = AtariImgEncoder(**config)
    
    param_count = sum(p.numel() for p in encoder.parameters())
    print(f"创建编码器 '{config_name}':")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_blocks: {config['num_blocks']}")
    print(f"  参数量: {param_count:,}")
    
    return encoder

# 使用示例
if __name__ == "__main__":
    print("=== Atari图像编码器使用示例 ===")
    
    # 模拟数据
    batch_size = 4
    seq_length = 33
    test_input = torch.randn(batch_size, seq_length, 4, 84, 84)
    
    # 创建不同配置的编码器
    print("\n1. 快速配置 (适合实验):")
    encoder_fast = create_encoder("fast")
    output = encoder_fast(test_input)
    print(f"   输出形状: {output.shape}")
    
    print("\n2. 平衡配置 (推荐日常使用):")
    encoder_balanced = create_encoder("balanced")
    output = encoder_balanced(test_input)
    print(f"   输出形状: {output.shape}")
    
    print("\n3. 高质量配置 (重要任务):")
    encoder_hq = create_encoder("high_quality")
    output = encoder_hq(test_input)
    print(f"   输出形状: {output.shape}")
    
    print("\n4. 自定义配置:")
    encoder_custom = create_encoder("balanced", d_model=1024)  # 自定义d_model
    output = encoder_custom(test_input)
    print(f"   输出形状: {output.shape}")
    
    print("\n=== 与DataLoader配合使用 ===")
    print("""
# 在你的训练代码中使用:
encoder = create_encoder("balanced")  # 或其他配置

for batch in dataloader:
    observations = batch['observations']  # (batch, 33, 4, 84, 84)
    encoded = encoder(observations)       # (batch, 33, 512)
    
    # 分离状态用于RL训练
    states = encoded[:, :-1, :]      # (batch, 32, 512)
    next_states = encoded[:, 1:, :]  # (batch, 32, 512)
    actions = batch['actions']       # (batch, 32)
    rewards = batch['rewards']       # (batch, 32)
    """)