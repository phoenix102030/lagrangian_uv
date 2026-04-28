import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerSpatialExtractor(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_sites=3, spatial_dim=2):
        super().__init__()
        self.num_sites = num_sites
        self.spatial_dim = spatial_dim
        
        # 1. CNN 提取背景局部纹理 (感受野较小，提取底层物理规律)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 2. Transformer 编码器：捕捉站点间的空间长距离依赖 (风场的遥相关)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True, 
            dropout=0.1
        )
        # 2层足够，太深容易在只有3个点的数据上过拟合
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 站点位置编码 (告诉 Transformer 谁是 A 点，谁是 B 点)
        self.site_pos_embedding = nn.Parameter(torch.randn(1, num_sites, hidden_dim))

        # 3. 物理参数输出头
        self.advection_head = nn.Linear(hidden_dim, spatial_dim) 
        # 输出3个值，用于构建 2x2 的下三角矩阵 (L11, L21, L22)
        self.cov_head = nn.Linear(hidden_dim, 3) 

    def forward(self, nwp_u: torch.Tensor, nwp_v: torch.Tensor, site_coords_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        nwp_u, nwp_v: [Batch, Time, H, W]
        site_coords_norm: [3, 2] 经纬度必须归一化到 [-1, 1] 才能用于 grid_sample
        """
        b, t, h, w = nwp_u.shape
        x = torch.cat([nwp_u.unsqueeze(2), nwp_v.unsqueeze(2)], dim=2) # [B, T, 2, H, W]
        x = x.view(b * t, 2, h, w)
        
        # 1. 提取全局物理特征图
        feature_map = self.cnn(x) # [B*T, 64, H, W]
        
        # 2. 精准采样：基于站点的真实坐标提取特征
        # 构建 grid: shape [B*T, num_sites, 1, 2]
        grids = site_coords_norm.unsqueeze(0).unsqueeze(2).expand(b * t, -1, -1, -1)
        point_features = F.grid_sample(feature_map, grids, align_corners=True) # [B*T, 64, 3, 1]
        point_features = point_features.squeeze(-1).permute(0, 2, 1) # [B*T, 3, 64]
        
        # 3. Transformer 空间交互
        point_features = point_features + self.site_pos_embedding # 加上位置编码
        attended_features = self.transformer(point_features) # [B*T, 3, 64]
        
        # 4. 生成平流向量 (Advection)
        advection = self.advection_head(attended_features) # [B*T, 3, 2]
        
        # 5. 生成协方差矩阵 (Covariance) 并保证正定
        cov_raw = self.cov_head(attended_features) # [B*T, 3, 3]
        
        # 构建 Cholesky 下三角矩阵 L
        l11 = F.softplus(cov_raw[..., 0]) + 1e-4
        l21 = cov_raw[..., 1]
        l22 = F.softplus(cov_raw[..., 2]) + 1e-4
        
        zero = torch.zeros_like(l11)
        row1 = torch.stack([l11, zero], dim=-1)
        row2 = torch.stack([l21, l22], dim=-1)
        cholesky = torch.stack([row1, row2], dim=-2) # [B*T, 3, 2, 2]
        
        # Sigma = L * L^T (保证严格正定)
        covariance = cholesky @ cholesky.transpose(-1, -2) # [B*T, 3, 2, 2]

        return {
            "drift_terms": advection.view(b, t, self.num_sites, self.spatial_dim),
            "dispersion_terms": covariance.view(b, t, self.num_sites, self.spatial_dim, self.spatial_dim),
            "inv_dispersion_terms": torch.linalg.inv(covariance).view(b, t, self.num_sites, self.spatial_dim, self.spatial_dim)
        }