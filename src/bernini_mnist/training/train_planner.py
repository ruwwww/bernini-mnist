import argparse
import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from bernini_mnist.models.vit_encoder import MNISTViT
from bernini_mnist.models.planner import BerniniSemanticPlanner
from bernini_mnist.models.renderer import MLPFlowRenderer
from bernini_mnist.models.flow_matching import FlowMatchScheduler
from bernini_mnist.data.mnist_datamodule import get_mnist_dataloaders
from bernini_mnist.utils.seed import set_seed

def train_planner(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[Train Planner] Using device: {device}, MLLM backbone: {args.mllm_model_name}")

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

    # 2. Load optional pretrained renderer for visual inspection during planner training
    renderer = None
    if os.path.exists(args.renderer_path):
        print(f"Loading pretrained renderer from {args.renderer_path} for visual tracking...")
        renderer = MLPFlowRenderer(
            in_dim=784,
            hidden_size=args.renderer_hidden,
            num_layers=args.renderer_layers,
            semantic_dim=args.semantic_dim,
            num_semantic_tokens=args.num_semantic_tokens,
            cond_type="semantic"
        ).to(device)
        rend_ckpt = torch.load(args.renderer_path, map_location=device)
        renderer.load_state_dict(rend_ckpt["model_state_dict"])
        renderer.eval()
        for p in renderer.parameters():
            p.requires_grad = False

    # 3. Initialize Bernini Semantic Planner (Qwen backbone frozen)
    planner = BerniniSemanticPlanner(
        mllm_model_name=args.mllm_model_name,
        semantic_dim=args.semantic_dim,
        num_semantic_tokens=args.num_semantic_tokens,
        num_classes=10,
        head_hidden_size=args.head_hidden_size,
        head_layers=args.head_layers,
        freeze_mllm=True
    ).to(device)

    # Collect only trainable parameters (class_embed, mask_tokens, vit_proj, fm_head)
    trainable_params = [p for p in planner.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    print(f"Number of trainable planner parameters: {num_params:,}")

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_loss = float("inf")
    test_classes = torch.arange(10, device=device).repeat(2)  # 20 samples (two sets of 0-9)

    for epoch in range(1, args.epochs + 1):
        planner.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Planner Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            B = images.size(0)

            with torch.no_grad():
                z_1 = vit_encoder.extract_features(images)

            optimizer.zero_grad()
            loss = planner.compute_loss(labels, z_1, mask_ratio=args.mask_ratio)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_count += B
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        lr_scheduler.step()
        epoch_loss = total_loss / total_count
        print(f"Epoch {epoch}: Planner Train FM Loss = {epoch_loss:.6f}")

        # Periodic visual generation check if renderer is loaded
        if (epoch % args.sample_every == 0 or epoch == args.epochs) and renderer is not None:
            planner.eval()
            with torch.no_grad():
                gen_semantic = planner.sample_iterative(
                    test_classes,
                    num_refinement_steps=args.num_refinement_steps,
                    ode_steps_per_refinement=20
                )
                
                # Render generated semantic tokens to pixels
                scheduler = FlowMatchScheduler()
                def rend_vel_fn(x, t_step):
                    return renderer(x, t_step, gen_semantic)

                sampled_x = scheduler.sample_euler(
                    rend_vel_fn,
                    shape=(test_classes.shape[0], 784),
                    device=device,
                    steps=30
                )
                sampled_imgs = (sampled_x.view(-1, 1, 28, 28) + 1.0) / 2.0
                sampled_imgs = torch.clamp(sampled_imgs, 0.0, 1.0)

                sample_path = os.path.join(args.sample_dir, f"epoch_{epoch:03d}_planner_rendered.png")
                save_image(sampled_imgs, sample_path, nrow=10)
                print(f"Saved planner generated image grid to {sample_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"--> Saving best planner checkpoint to {args.save_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v for k, v in planner.state_dict().items() if "mllm" not in k},
                "loss": best_loss,
                "args": vars(args)
            }, args.save_path)

    print(f"[Train Planner] Training complete. Best Loss: {best_loss:.6f}")
    return best_loss

def main():
    parser = argparse.ArgumentParser(description="Train Bernini Semantic Planner on MNIST")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--mllm_model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vit_path", type=str, default="./checkpoints/vit_mnist.pt")
    parser.add_argument("--renderer_path", type=str, default="./checkpoints/renderer_semantic.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/planner_qwen.pt")
    parser.add_argument("--sample_dir", type=str, default="./outputs/samples_planner")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--semantic_dim", type=int, default=256)
    parser.add_argument("--num_semantic_tokens", type=int, default=16)
    parser.add_argument("--head_hidden_size", type=int, default=512)
    parser.add_argument("--head_layers", type=int, default=3)
    parser.add_argument("--vit_depth", type=int, default=4)
    parser.add_argument("--vit_heads", type=int, default=4)
    parser.add_argument("--renderer_hidden", type=int, default=512)
    parser.add_argument("--renderer_layers", type=int, default=4)
    parser.add_argument("--num_refinement_steps", type=int, default=4)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    train_planner(args)

if __name__ == "__main__":
    main()
