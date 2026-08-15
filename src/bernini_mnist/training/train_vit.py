import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from bernini_mnist.models.vit_encoder import MNISTViT
from bernini_mnist.data.mnist_datamodule import get_mnist_dataloaders
from bernini_mnist.utils.seed import set_seed

def train_vit(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[Train ViT] Using device: {device}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    train_loader, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = MNISTViT(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                loss = criterion(logits, labels)
                test_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += images.size(0)

        test_acc = test_correct / test_total
        test_loss = test_loss / test_total
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"--> Saving best model with Test Acc: {best_acc:.4f} to {args.save_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_acc": test_acc,
                "args": vars(args)
            }, args.save_path)

    print(f"[Train ViT] Training complete. Best Test Accuracy: {best_acc:.4f}")
    return best_acc

def main():
    parser = argparse.ArgumentParser(description="Train MNIST ViT Feature Extractor & Classifier")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="./checkpoints/vit_mnist.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    train_vit(args)

if __name__ == "__main__":
    main()
