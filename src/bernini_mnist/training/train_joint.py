import argparse
import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from bernini_mnist.models.vit_encoder import MNISTViT
from bernini_mnist.models.pipeline import BerniniMNISTPipeline
from bernini_mnist.data.mnist_datamodule import get_mnist_dataloaders
from bernini_mnist.utils.seed import set_seed

def train_joint(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[Joint Training] Using device: {device}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 1. Load frozen ViT feature extractor oracle
    print(f"Loading pretrained ViT from {args.vit_path}...")
    vit_encoder = MNISTViT(
        img_size=28,
        patch_size=7,
        embed_dim=args.semantic_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads
    ).to(device)
    vit_ckpt = torch.load(args.vit_path, map_location=device)
    vit_encoder.load_state_dict(vit_ckpt["model_state_dict"])
    vit_encoder.eval()
    for p in vit_encoder.parameters():
        p.requires_grad = False

    # 2. Load Pipeline with pretrained planner & renderer
    pipeline = BerniniMNISTPipeline.from_pretrained(
        planner_path=args.planner_path,
        renderer_path=args.renderer_path,
        mllm_model_name=args.mllm_model_name,
        semantic_dim=args.semantic_dim,
        renderer_hidden=args.renderer_hidden,
        renderer_layers=args.renderer_layers,
        planner_head_hidden=args.head_hidden_size,
        planner_head_layers=args.head_layers,
        device=device
    )

    # Trainable parameters across planner (without frozen Qwen) + renderer
    trainable_params = [
        p for p in pipeline.planner.parameters() if p.requires_grad
    ] + list(pipeline.renderer.parameters())

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss = float("inf")
    test_classes = torch.arange(10, device=device).repeat(2)

    for epoch in range(1, args.epochs + 1):
        pipeline.planner.train()
        pipeline.renderer.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Joint Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            B = images.size(0)
            x_1 = images.flatten(1)

            with torch.no_grad():
                z_1 = vit_encoder.extract_features(images)

            optimizer.zero_grad()

            # Planner loss
            loss_plan = pipeline.planner.compute_loss(labels, z_1, mask_ratio=args.mask_ratio)

            # Renderer loss (trained on ground truth or generated tokens)
            x_t, target_v_pixel, t_render = pipeline.scheduler.add_noise(x_1)
            pred_v_pixel = pipeline.renderer(x_t, t_render, z_1)
            loss_render = pipeline.scheduler.compute_loss(pred_v_pixel, target_v_pixel)

            total_step_loss = args.lambda_plan * loss_plan + args.lambda_render * loss_render
            total_step_loss.backward()
            optimizer.step()

            total_loss += total_step_loss.item() * B
            total_count += B
            pbar.set_postfix({
                "plan_loss": f"{loss_plan.item():.4f}",
                "rend_loss": f"{loss_render.item():.4f}",
                "tot_loss": f"{total_step_loss.item():.4f}"
            })

        lr_scheduler.step()
        epoch_loss = total_loss / total_count
        print(f"Epoch {epoch}: Joint Average Loss = {epoch_loss:.6f}")

        # Visual Sampling
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            pipeline.planner.eval()
            pipeline.renderer.eval()
            with torch.no_grad():
                gen_images, _ = pipeline.generate(
                    test_classes,
                    num_refinement_steps=args.num_refinement_steps,
                    ode_steps_planner=20,
                    ode_steps_renderer=30
                )
                sample_path = os.path.join(args.sample_dir, f"epoch_{epoch:03d}_joint_sample.png")
                save_image(gen_images, sample_path, nrow=10)
                print(f"Saved joint generated samples to {sample_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"--> Saving best joint checkpoint to {args.save_path}")
            torch.save({
                "epoch": epoch,
                "planner_state_dict": {k: v for k, v in pipeline.planner.state_dict().items() if "mllm" not in k},
                "renderer_state_dict": pipeline.renderer.state_dict(),
                "loss": best_loss,
                "args": vars(args)
            }, args.save_path)

    print(f"[Joint Training] Complete. Best Loss: {best_loss:.6f}")
    return best_loss

def main():
    parser = argparse.ArgumentParser(description="Joint Fine-Tuning of Bernini Planner and Renderer")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--mllm_model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vit_path", type=str, default="./checkpoints/vit_mnist.pt")
    parser.add_argument("--planner_path", type=str, default="./checkpoints/planner_qwen.pt")
    parser.add_argument("--renderer_path", type=str, default="./checkpoints/renderer_semantic.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/joint_pipeline.pt")
    parser.add_argument("--sample_dir", type=str, default="./outputs/samples_joint")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--lambda_plan", type=float, default=1.0)
    parser.add_argument("--lambda_render", type=float, default=1.0)
    parser.add_argument("--semantic_dim", type=int, default=256)
    parser.add_argument("--head_hidden_size", type=int, default=512)
    parser.add_argument("--head_layers", type=int, default=3)
    parser.add_argument("--renderer_hidden", type=int, default=512)
    parser.add_argument("--renderer_layers", type=int, default=4)
    parser.add_argument("--vit_depth", type=int, default=4)
    parser.add_argument("--vit_heads", type=int, default=4)
    parser.add_argument("--num_refinement_steps", type=int, default=4)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    train_joint(args)

if __name__ == "__main__":
    main()
