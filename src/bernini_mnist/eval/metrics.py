import torch
import numpy as np
from typing import Dict, Any, List

@torch.no_grad()
def evaluate_classifier_accuracy(
    images: torch.Tensor,
    target_labels: torch.Tensor,
    classifier: torch.nn.Module,
    device: torch.device
) -> float:
    """
    Compute accuracy of generated images under the classifier oracle.
    
    Args:
        images: Generated images (B, 1, 28, 28) in range [0, 1]
        target_labels: Intended class labels (B,)
        classifier: Pretrained classifier expecting inputs in [-1, 1]
    Returns:
        accuracy: Float between 0.0 and 1.0
    """
    classifier.eval()
    # Normalize [0, 1] -> [-1, 1]
    norm_imgs = (images * 2.0 - 1.0).to(device)
    target_labels = target_labels.to(device)
    
    logits, _ = classifier(norm_imgs)
    preds = logits.argmax(dim=-1)
    acc = (preds == target_labels).float().mean().item()
    return acc

@torch.no_grad()
def evaluate_within_class_diversity(
    images_per_class: Dict[int, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute intra-class variance and mean pairwise pixel distance across generated samples.
    
    Args:
        images_per_class: Dict mapping class_id (0-9) to Tensor of shape (K, 1, 28, 28)
    Returns:
        dict containing mean variance and pairwise distances
    """
    class_variances = []
    pairwise_distances = []

    for c, imgs in images_per_class.items():
        if imgs.shape[0] < 2:
            continue
        flat = imgs.view(imgs.shape[0], -1)  # (K, 784)
        var = flat.var(dim=0).mean().item()
        class_variances.append(var)

        # Pairwise Euclidean distances
        pdist = torch.cdist(flat, flat)
        # Extract upper triangle without diagonal
        triu_indices = torch.triu_indices(imgs.shape[0], imgs.shape[0], offset=1)
        dist_vals = pdist[triu_indices[0], triu_indices[1]]
        pairwise_distances.append(dist_vals.mean().item())

    return {
        "mean_intra_class_variance": float(np.mean(class_variances)) if class_variances else 0.0,
        "mean_intra_class_pairwise_dist": float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
    }

@torch.no_grad()
def evaluate_nearest_neighbor_distance(
    generated_images: torch.Tensor,
    real_images: torch.Tensor
) -> float:
    """
    Compute average minimum distance from generated samples to the real training set.
    Detects if model has memorized training examples (very small dist) or generalized.
    
    Args:
        generated_images: (B, 1, 28, 28)
        real_images: (N_real, 1, 28, 28)
    Returns:
        mean_min_dist: Average L2 distance to nearest real sample
    """
    gen_flat = generated_images.view(generated_images.shape[0], -1)
    real_flat = real_images.view(real_images.shape[0], -1)

    min_dists = []
    # Compute in chunks to avoid GPU OOM
    chunk_size = 100
    for i in range(0, gen_flat.shape[0], chunk_size):
        gen_chunk = gen_flat[i:i + chunk_size]
        dists = torch.cdist(gen_chunk, real_flat)  # (chunk, N_real)
        min_val, _ = dists.min(dim=1)
        min_dists.extend(min_val.cpu().tolist())

    return float(np.mean(min_dists))
