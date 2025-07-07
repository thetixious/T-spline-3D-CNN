import torch

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)[0] if isinstance(model(points), tuple) else model(points)
            preds = outputs.max(1)[1]
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def feature_transform_regularizer(trans):
    """
    Регуляризация для feature transform матрицы
    """
    batchsize = trans.size(0)
    d = trans.size(1)
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
