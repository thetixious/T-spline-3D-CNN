
import argparse
import torch
from pointnet.model import PointNetClassifier
from data.loader import get_dataloaders

def evaluate(model_path, num_points=1024, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetClassifier(k=40)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    train_loader, test_loader = get_dataloaders(
        root_dir="ModelNet40",
        batch_size=16,
        num_points=1024
    )
    correct = total = 0
    with torch.no_grad():
        for points, labels in test_loader:
            points, labels = points.to(device), labels.to(device)
            outputs, _ = model(points)
            preds = outputs.max(1)[1]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Accuracy on test set: {acc:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PointNet model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_path, args.num_points, args.batch_size)