import torch
import torch.nn as nn
import torch.optim as optim
from pointnet.model import PointNetClassifier
from data.loader import get_dataloaders
from utils.metrics import accuracy, feature_transform_regularizer
from utils.visualizer import plot_loss_curve

def train_model(
    num_classes=40,
    num_epochs=20,
    batch_size=16,
    lr=0.001,
    num_points=1024,
    model_save_path="pointnet_best.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetClassifier(k=num_classes).to(device)
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, num_points=num_points)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_test_acc = 0.0
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, trans_feat = model(points)
            loss = criterion(outputs, labels)
            # Регуляризация feature transform
            if trans_feat is not None:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        test_acc = accuracy(model, test_loader, device)
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Test acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_test_acc:
            torch.save(model.state_dict(), model_save_path)
            best_test_acc = test_acc

    plot_loss_curve(train_losses)
    print(f"Best test accuracy: {best_test_acc:.2f}%")
