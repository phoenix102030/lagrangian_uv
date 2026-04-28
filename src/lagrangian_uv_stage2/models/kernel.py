import torch

class StochasticAdvectionKernel:
    def __init__(self, kernel_jitter=1e-6):
        self.kernel_jitter = kernel_jitter

    def compute_transition(self, site_coords: torch.Tensor, drift_terms: torch.Tensor, inv_dispersion: torch.Tensor) -> torch.Tensor:
        """
        计算 3x3 的系统状态转移矩阵
        site_coords: [3, 2] 实际物理坐标 (不需要归一化)
        drift_terms: [B, T, 3, 2] (advection)
        inv_dispersion: [B, T, 3, 2, 2] (inv_covariance)
        """
        b, t, num_sites, _ = drift_terms.shape
        
        # coords_i 是目标点, coords_j 是出发点
        coords_i = site_coords.unsqueeze(1) # [3, 1, 2]
        coords_j = site_coords.unsqueeze(0) # [1, 3, 2]
        
        # 核心物理意义：风从 j 出发，平流移动了 drift_terms[j] 距离
        # 所以扩散核的中心从 coords_j 偏移到了 coords_j + drift_terms[j]
        drift_j = drift_terms.unsqueeze(2) # [B, T, 1, 3, 2]
        
        # 计算点 i 距离扩散中心的向量差: (x_i - (x_j + v_j))
        diff = coords_i.unsqueeze(0).unsqueeze(0) - coords_j.unsqueeze(0).unsqueeze(0) - drift_j 
        # diff shape: [B, T, 3(i), 3(j), 2]
        
        inv_cov_j = inv_dispersion.unsqueeze(2) # [B, T, 1, 3, 2, 2]
        diff_unsqueeze = diff.unsqueeze(-1) # [B, T, 3, 3, 2, 1]
        
        # 二次型计算: diff^T * Sigma^{-1} * diff
        mahalanobis_sq = torch.matmul(
            diff.unsqueeze(-2), 
            torch.matmul(inv_cov_j, diff_unsqueeze)
        ).squeeze(-1).squeeze(-1) # [B, T, 3, 3]
        
        # 构建高斯核转移率
        transition = torch.exp(-0.5 * mahalanobis_sq)
        
        # 【极其重要】：这里去掉了对 transition 沿 dim=1 的强制除法 (row_sum 归一化)
        # 因为在物理上，如果平流场将物质吹出了这 3 个点组成的闭包，物质总量本就会减少！
        # 乘以 0.99 作为微小衰减，保证动力系统稳定不发散
        transition = transition * 0.99 
        
        return transition # [B, T, 3, 3]