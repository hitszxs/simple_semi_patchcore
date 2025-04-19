import torch
from anomalib.models.components import DynamicBufferModule, FeatureExtractor, KCenterGreedy

def batched_emd(features1: torch.Tensor, features2: torch.Tensor, epsilon=0.01, max_iters=100) -> torch.Tensor:

    B, N, D = features1.shape
    _, M, _ = features2.shape

    weight1 = torch.full((B, N), 1.0 / N, device=features1.device)
    weight2 = torch.full((B, M), 1.0 / M, device=features2.device)

    dist_matrix = torch.cdist(features1, features2, p=2)

    K = torch.exp(-dist_matrix / epsilon)  

    u = torch.ones_like(weight1)  
    v = torch.ones_like(weight2) 

    for _ in range(max_iters):
        u = weight1 / (K @ v.unsqueeze(-1)).squeeze(-1)  
        v = weight2 / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)  

    transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(1)  # shape: [B, N, M]

    # 根据运输矩阵计算EMD距离
    emd_distances = torch.sum(transport_plan * dist_matrix, dim=(1, 2))  # shape: [B]

    return emd_distances


def subsample_embedding(self, embedding: Tensor, sampling_ratio: float) -> None:
    """Subsample embedding based on coreset sampling and store to memory.

    Args:
        embedding (np.ndarray): Embedding tensor from the CNN
        sampling_ratio (float): Coreset sampling ratio
    """

    # Coreset Subsampling
    sampler = KCenterGreedy(embedding=embedding, sampling_ratio=sampling_ratio)
    coreset = sampler.sample_coreset()
    self.memory_bank = coreset