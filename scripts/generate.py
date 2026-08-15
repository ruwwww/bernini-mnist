import argparse
import os
import torch
from torchvision.utils import save_image

from bernini_mnist.models.pipeline import BerniniMNISTPipeline

def main():
    parser = argparse.ArgumentParser(description="Generate MNIST Digits using Bernini Pipeline")
    parser.add_argument("--digits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="Digit classes to generate")
    parser.add_argument("--samples_per_digit", type=int, default=4, help="Number of distinct samples per digit class")
    parser.add_argument("--mllm_model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--planner_path", type=str, default="./checkpoints/planner_qwen.pt")
    parser.add_argument("--renderer_path", type=str, default="./checkpoints/renderer_semantic.pt")
    parser.add_argument("--output_path", type=str, default="./outputs/generated_grid.png")
    parser.add_argument("--refinement_steps", type=int, default=4, help="K steps for masked token refinement")
    parser.add_argument("--ode_steps_planner", type=int, default=20)
    parser.add_argument("--ode_steps_renderer", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("Loading Bernini MNIST Pipeline...")
    pipeline = BerniniMNISTPipeline.from_pretrained(
        planner_path=args.planner_path,
        renderer_path=args.renderer_path,
        mllm_model_name=args.mllm_model_name,
        device=device
    )

    # Build batch of digit labels: repeat each digit samples_per_digit times
    labels_list = []
    for d in args.digits:
        labels_list.extend([d] * args.samples_per_digit)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)

    print(f"Generating {len(labels)} images for digits: {args.digits} with K={args.refinement_steps} refinement steps...")
    images, _ = pipeline.generate(
        class_labels=labels,
        num_refinement_steps=args.refinement_steps,
        ode_steps_planner=args.ode_steps_planner,
        ode_steps_renderer=args.ode_steps_renderer,
        temperature=args.temperature,
        seed=args.seed
    )

    save_image(images, args.output_path, nrow=args.samples_per_digit)
    print(f"Saved generated digit grid to: {args.output_path}")

if __name__ == "__main__":
    main()
