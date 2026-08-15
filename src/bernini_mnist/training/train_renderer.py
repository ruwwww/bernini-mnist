import argparse
import os
import torch
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from bernini_mnist.models.vit_encoder import MNISTViT
from bernini_mnist.models.flow_matching import FlowMatchScheduler
from bernini_mnist.models.renderer import MLPFlowRenderer
from bernini_mnist.data.mnist_datamodule import get_mnist_dataloaders
from bernini_mnist.utils.seed import set_seed

def train_renderer(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[Train Renderer] Using device: {device}, cond_type: {args.cond_type}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load frozen ViT feature extractor & classifier oracle
    vit_encoder = None
    if args.cond_type == "semantic" or args.eval_classifier:
        print(f"Loading pretrained ViT from {args.vit_path}...")
        vit_encoder = MNISTViT(
            img_size=28,
            patch_size=7,
            embed_dim=args.semantic_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads
        ).to(device)
        ckpt = torch.load(args.vit_path, map_location=device)
        vit_encoder.load_state_dict(ckpt["model_state_dict"])
        vit_encoder.eval()
        for p in vit_encoder.parameters():
            p.requires_grad = False
        print(f"Loaded ViT with pre-recorded test accuracy: {ckpt.get('test_acc', 'N/A')}")

    # Flow matching scheduler & renderer model
    scheduler = FlowMatchScheduler()
    renderer = MLPFlowRenderer(
        in_dim=784,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        semantic_dim=args.semantic_dim,
        num_semantic_tokens=args.num_semantic_tokens,
        num_classes=10,
        cond_type=args.cond_type
    ).to(device)

    optimizer = optim.AdamW(renderer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_loss = float("inf")

    # Fixed test batch for consistent visual tracking across epochs
    fixed_test_images, fixed_test_labels = next(iter(test_loader))
    fixed_test_images = fixed_test_images[:16].to(device)
    fixed_test_labels = fixed_test_labels[:16].to(device)
    fixed_cond = None
    if args.cond_type == "semantic":
        with torch.no_grad():
            fixed_cond = vit_encoder.extract_features(fixed_test_images)
    else:
        fixed_cond = fixed_test_labels

    for epoch in range(1, args.epochs + 1):
        renderer.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Renderer Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)

            # Flatten images to (B, 784)
            x_1 = images.flatten(1)

            # Get conditioning
            if args.cond_type == "semantic":
                with torch.no_grad():
                    cond = vit_encoder.extract_features(images)
            else:
                cond = labels

            # Sample flow matching noisy state & target velocity
            x_t, target_v, t = scheduler.add_noise(x_1)

            optimizer.zero_grad()
            pred_v = renderer(x_t, t, cond)
            loss = scheduler.compute_loss(pred_v, target_v)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_count += B
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        lr_scheduler.step()
        epoch_loss = total_loss / total_count
        print(f"Epoch {epoch}: Average Train Loss = {epoch_loss:.6f}")

        # Evaluation & Sampling
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            renderer.eval()
            with torch.no_grad():
                def velocity_fn(x, t_step):
                    return renderer(x, t_step, fixed_cond)

                sampled_x = scheduler.sample_euler(
                    velocity_fn,
                    shape=(fixed_cond.shape[0], 784),
                    device=device,
                    steps=args.eval_steps
                )
                # Reshape to (B, 1, 28, 28) and unnormalize [-1, 1] -> [0, 1]
                sampled_imgs = (sampled_x.view(-1, 1, 28, 28) + 1.0) / 2.0
                sampled_imgs = torch.clamp(sampled_imgs, 0.0, 1.0)

                sample_path = os.path.join(args.sample_dir, f"epoch_{epoch:03d}_{args.cond_type}.png")
                save_image(sampled_imgs, sample_path, nrow=4)
                print(f"Saved generated visual sample to {sample_path}")

                # Quantitative check if ViT oracle available
                if vit_encoder is not None:
                    # Norm to [-1, 1] for ViT input
                    vit_in = sampled_imgs * 2.0 - 1.0
                    logits, _ = vit_encoder(vit_in)
                    preds = logits.argmax(dim=-1)
                    acc = (preds == fixed_test_labels).float().mean().item()
                    print(f"--> Validation sample classifier accuracy: {acc * 100:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"--> Saving best renderer checkpoint to {args.save_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": renderer.state_dict(),
                "loss": best_loss,
                "args": vars(args)
            }, args.save_path)

    print(f"[Train Renderer] Training complete. Best Loss: {best_loss:.6f}")
    return best_loss

def main():
    parser = argparse.ArgumentParser(description="Train Pixel-Space Flow Matching Renderer")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--vit_path", type=str, default="./checkpoints/vit_mnist.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/renderer_semantic.pt")
    parser.add_argument("--sample_dir", type=str, default="./outputs/samples_renderer")
    parser.add_argument("--cond_type", type=str, default="semantic", choices=["semantic", "class"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--semantic_dim", type=int, default=256)
    parser.add_argument("--num_semantic_tokens", type=int, default=16)
    parser.add_argument("--vit_depth", type=int, default=4)
    parser.add_argument("--vit_heads", type=int, default=4)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=30)
    parser.add_argument("--eval_classifier", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    train_renderer(args)

if __name__ == "__main__":
    main()
