import argparse
import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from bernini_mnist.models.vit_encoder import MNISTViT
from bernini_mnist.models.pipeline import BerniniMNISTPipeline
from bernini_mnist.data.mnist_datamodule import get_mnist_dataloaders
from bernini_mnist.eval.metrics import (
    evaluate_classifier_accuracy,
    evaluate_within_class_diversity,
    evaluate_nearest_neighbor_distance
)

def evaluate_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[Evaluation] Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load ViT Classifier Oracle
    print(f"Loading classifier oracle from {args.vit_path}...")
    vit_encoder = MNISTViT(
        img_size=28,
        patch_size=7,
        embed_dim=args.semantic_dim,
        depth=4,
        num_heads=4
    ).to(device)
    vit_ckpt = torch.load(args.vit_path, map_location=device)
    vit_encoder.load_state_dict(vit_ckpt["model_state_dict"])
    vit_encoder.eval()

    # 2. Load Pipeline
    print(f"Loading pipeline from {args.planner_path} and {args.renderer_path}...")
    pipeline = BerniniMNISTPipeline.from_pretrained(
        planner_path=args.planner_path,
        renderer_path=args.renderer_path,
        mllm_model_name=args.mllm_model_name,
        semantic_dim=args.semantic_dim,
        device=device
    )

    # 3. Load real MNIST test set for nearest neighbor evaluation
    _, test_loader = get_mnist_dataloaders(data_dir=args.data_dir, batch_size=500, num_workers=2)
    real_images_list = []
    for imgs, _ in test_loader:
        real_images_list.append((imgs + 1.0) / 2.0)  # [0, 1]
    real_images = torch.cat(real_images_list, dim=0)[:2000].to(device)  # 2000 real images

    # 4. Generate samples for evaluation (e.g. 50 samples per class = 500 samples total)
    print(f"Generating {args.samples_per_class} samples per class across classes 0-9...")
    all_gen_images = []
    all_target_labels = []
    images_by_class = {}

    for c in range(10):
        c_labels = torch.full((args.samples_per_class,), c, dtype=torch.long, device=device)
        with torch.no_grad():
            gen_imgs, _ = pipeline.generate(
                class_labels=c_labels,
                num_refinement_steps=args.refinement_steps,
                temperature=args.temperature,
                seed=args.seed + c * 100
            )
        all_gen_images.append(gen_imgs)
        all_target_labels.append(c_labels)
        images_by_class[c] = gen_imgs

    all_gen_images = torch.cat(all_gen_images, dim=0)
    all_target_labels = torch.cat(all_target_labels, dim=0)

    # Save visual grid (10 rows, first 10 columns)
    grid_imgs = []
    for c in range(10):
        grid_imgs.append(images_by_class[c][:10])
    grid_imgs = torch.cat(grid_imgs, dim=0)
    save_path = os.path.join(args.output_dir, "evaluation_grid.png")
    save_image(grid_imgs, save_path, nrow=10)
    print(f"Saved 10x10 evaluation grid to {save_path}")

    # 5. Compute Metrics
    print("\n" + "="*50)
    print("=== Quantitative Evaluation Results ===")
    print("="*50)

    # Metric 1: Classifier Accuracy
    acc = evaluate_classifier_accuracy(all_gen_images, all_target_labels, vit_encoder, device)
    print(f"1. Generation Classification Accuracy: {acc * 100:.2f}%")

    # Metric 2: Intra-Class Diversity
    div = evaluate_within_class_diversity(images_by_class)
    print(f"2. Intra-Class Pixel Variance:        {div['mean_intra_class_variance']:.6f}")
    print(f"   Intra-Class Pairwise L2 Distance:  {div['mean_intra_class_pairwise_dist']:.4f}")

    # Metric 3: Nearest Neighbor Distance to Real Data
    nn_dist = evaluate_nearest_neighbor_distance(all_gen_images, real_images)
    print(f"3. Mean Distance to Nearest Real Digit: {nn_dist:.4f}")
    print("="*50)

    # Save summary report
    report_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(report_path, "w") as f:
        f.write("=== Bernini MNIST Evaluation Summary ===\n")
        f.write(f"Classifier Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Intra-Class Variance: {div['mean_intra_class_variance']:.6f}\n")
        f.write(f"Intra-Class Pairwise L2: {div['mean_intra_class_pairwise_dist']:.4f}\n")
        f.write(f"Mean Nearest Neighbor Dist: {nn_dist:.4f}\n")
    print(f"Saved evaluation report to {report_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bernini MNIST Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--mllm_model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vit_path", type=str, default="./checkpoints/vit_mnist.pt")
    parser.add_argument("--planner_path", type=str, default="./checkpoints/planner_qwen.pt")
    parser.add_argument("--renderer_path", type=str, default="./checkpoints/renderer_semantic.pt")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation")
    parser.add_argument("--samples_per_class", type=int, default=30)
    parser.add_argument("--semantic_dim", type=int, default=256)
    parser.add_argument("--refinement_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    evaluate_pipeline(args)

if __name__ == "__main__":
    main()
