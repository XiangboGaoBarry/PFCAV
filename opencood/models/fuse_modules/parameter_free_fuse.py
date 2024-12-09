# -*- coding: utf-8 -*-
# Author: Adapted from Xiangbo Gao <xiangbog@umich.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Implementation of spatial fusion methods including:
- Max Pooling
- Average Pooling
- Weighted Average Pooling (Entropy-based, Cosine Similarity-based, Distance-based, Combined)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_based_weights(x, eps=1e-6):
    # x: (B, C, H, W)
    # Compute softmax along the channel dimension
    # Reshape to (B, C, -1) for convenience
    B, C, H, W = x.shape
    x_reshaped = x.view(B, C, -1)  # (B, C, HW)
    x_softmax = F.softmax(x_reshaped, dim=1)  # (B, C, HW)

    entropy = -torch.sum(x_softmax * torch.log(x_softmax + eps), dim=1)  # (B, HW)
    # Average entropy across spatial locations for weight assignment
    entropy = entropy.mean(dim=1, keepdim=True)  # (B, 1)

    # Normalize
    weights = entropy / (entropy.sum(dim=0, keepdim=True) + eps)
    return weights


def cosine_similarity_weights(x, eps=1e-6):
    # x: (B, C, H, W)
    # Flatten spatial dimensions
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)  # (B, C, HW)
    # Compute normalized features for each agent vector
    x_normed = x_flat / (torch.norm(x_flat, dim=1, keepdim=True) + eps)  # (B, C, HW)

    # Compute pairwise similarity: S_km = sum over all HW of (v_k^T v_m)
    # We'll average over spatial dims to get a scalar similarity per pair
    S = torch.einsum('bch,bcm->bhm', x_normed, x_normed)  # (B, B, HW)
    # Average over spatial dimension
    S = S.mean(dim=2)  # (B, B)

    total_sim = S.sum(dim=1) - 1.0  # (B,)

    # Invert to get weights
    w_inv = 1.0 / (total_sim + eps)
    # Normalize
    weights = w_inv / (w_inv.max() + eps)
    weights = weights.unsqueeze(1)  # (B, 1)
    return weights


def distance_based_weights(x, eps=1e-6):
    B, C, H, W = x.shape
    center_i = (H - 1) / 2.0
    center_j = (W - 1) / 2.0
    
    i_coords = torch.arange(H, dtype=torch.float32, device=x.device)
    j_coords = torch.arange(W, dtype=torch.float32, device=x.device)
    ii, jj = torch.meshgrid(i_coords, j_coords, indexing='ij')
    
    dist_map = torch.sqrt((ii - center_i)**2 + (jj - center_j)**2)  # (H, W)
    
    w_inv = 1.0 / (dist_map + eps)
    w_norm = w_inv / (w_inv.max() + eps)
    
    weights = w_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return weights


class SpatialFusion(nn.Module):
    def __init__(self, fusion_method="max"):
        """
        fusion_method: str, one of ["max", "avg", "entropy", "cosine", "distance", "combined"]
        agent_positions: torch.Tensor, shape (B, 2) if needed for distance weighting.
        spatial_locs: torch.Tensor, shape (H, W, 2) for spatial coordinates.
        """
        super(SpatialFusion, self).__init__()
        self.fusion_method = fusion_method

    def regroup(self, x, record_len):
        # x: (B, C, H, W)
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        # x: (B, C, H, W)
        # split_x: list of tensors [(B1, C, H, W), (B2, C, H, W), ...]
        split_x = self.regroup(x, record_len)
        out = []

        for xx in split_x:
            # xx: (B_group, C, H, W)
            if self.fusion_method == "max":
                # Max pooling along the agent dimension
                fused = torch.max(xx, dim=0, keepdim=True)[0]  # (1, C, H, W)
            elif self.fusion_method == "avg":
                fused = torch.mean(xx, dim=0, keepdim=True)  # (1, C, H, W)
            else:
                # Weighted methods
                B_group, C, H, W = xx.shape
                if self.fusion_method == "entropy":
                    weights = entropy_based_weights(xx)  # (B_group, 1)
                elif self.fusion_method == "cosine":
                    weights = cosine_similarity_weights(xx)  # (B_group, 1)
                elif self.fusion_method == "distance":
                    weights = distance_based_weights(xx)  # (B_group, 1)
                elif self.fusion_method == "combined":
                    # combined = cosine + distance
                    w_cos = cosine_similarity_weights(xx)
                    w_dis = distance_based_weights(xx)
                    w_combined = w_cos + w_dis
                    # Normalize combined weights
                    w_combined = w_combined / (w_combined.max() + 1e-6)
                    weights = w_combined
                else:
                    raise ValueError("Unknown fusion method")

                # Apply weights: weights shape (B_group, 1), broadcast to (B_group, C, H, W)
                weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (B_group, 1, 1, 1)
                fused = torch.sum(weights_expanded * xx, dim=0, keepdim=True) / B_group

            out.append(fused)

        return torch.cat(out, dim=0)