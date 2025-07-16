from pointnet.train import train_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet model")

    parser.add_argument("--num_classes", type=int, default=40, help="Number of output classes")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points per cloud")
    parser.add_argument("--model_save_path", type=str, default="pointnet_best.pth", help="Path to save the model")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(
        num_classes=args.num_classes,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_points=args.num_points,
        model_save_path=args.model_save_path
    )