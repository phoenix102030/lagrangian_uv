import torch

def _kalman_update(prior_mean: torch.Tensor, prior_cov: torch.Tensor, 
                   observation: torch.Tensor, obs_matrix: torch.Tensor, 
                   obs_noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    无阻塞的 GPU 卡尔曼滤波更新 (使用 torch.linalg.solve)
    prior_mean: [B, state_dim]
    prior_cov: [B, state_dim, state_dim]
    observation: [B, obs_dim]
    obs_matrix (H): [obs_dim, state_dim]
    obs_noise (R): [obs_dim, obs_dim]
    """
    b, state_dim = prior_mean.shape
    
    # 1. 预测观测 (y_pred)
    pred_obs = torch.matmul(obs_matrix, prior_mean.unsqueeze(-1)).squeeze(-1)
    
    # 2. 新息 (Innovation)
    innovation = observation - pred_obs # [B, obs_dim]
    
    # 3. 新息协方差 S = H * P * H^T + R
    P_Ht = torch.matmul(prior_cov, obs_matrix.transpose(-1, -2)) # [B, state_dim, obs_dim]
    S = torch.matmul(obs_matrix, P_Ht) + obs_noise # [B, obs_dim, obs_dim]
    
    # 注入微小扰动保证 S 正定
    S_reg = S + torch.eye(S.shape[-1], device=S.device, dtype=S.dtype) * 1e-4
    
    try:
        # 4. 计算卡尔曼增益 K (避免显式矩阵求逆)
        # 解方程 S * K^T = H * P  ==>  K^T = S^{-1} * (H * P)
        # linalg.solve 求解 AX = B 中的 X
        Kt = torch.linalg.solve(S_reg, P_Ht.transpose(-1, -2)) # [B, obs_dim, state_dim]
        K = Kt.transpose(-1, -2) # [B, state_dim, obs_dim]
        
        # 5. 更新后验均值
        post_mean = prior_mean + torch.matmul(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        # 6. 使用约瑟夫形式 (Joseph Form) 更新协方差，数学上绝对保证正定
        # P = (I - KH) * P * (I - KH)^T + K * R * K^T
        I = torch.eye(state_dim, device=prior_cov.device, dtype=prior_cov.dtype)
        I_KH = I - torch.matmul(K, obs_matrix)
        
        term1 = torch.matmul(I_KH, torch.matmul(prior_cov, I_KH.transpose(-1, -2)))
        term2 = torch.matmul(K, torch.matmul(obs_noise, K.transpose(-1, -2)))
        post_cov = term1 + term2
        
        return post_mean, post_cov
        
    except RuntimeError:
        # 如果矩阵病态到 linalg.solve 报错 (由于软约束，极少发生)，直接回退到先验
        # 这样梯度图不会因为 NaN 崩溃
        return prior_mean, prior_cov