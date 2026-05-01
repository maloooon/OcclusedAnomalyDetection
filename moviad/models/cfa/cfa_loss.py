import torch

def soft_boundary(phi_p: torch.Tensor, memory_bank, K, J, r, alpha, nu) -> float:
    """
    Loss calculation

    Args:
        phi_p (torch.Tensor) : batch of transformed features

    Returns:
        loss (float) : CFA loss
    """


    features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
    centers  = torch.sum(torch.pow(memory_bank, 2), 0, keepdim=True)
    f_c      = 2 * torch.matmul(phi_p, (memory_bank))
    dist     = features + centers - f_c
    n_neighbors = K + J
    dist     = dist.topk(n_neighbors, largest=False).values

    score = (dist[:, : , :K] - r**2)
    L_att = (1/nu) * torch.mean(torch.max(torch.zeros_like(score), score))

    score = (r**2 - dist[:, : , J:])
    L_rep  = (1/nu) * torch.mean(torch.max(torch.zeros_like(score), score - alpha))

    loss = L_att + L_rep

    return loss