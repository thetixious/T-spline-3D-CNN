from pointnet.train import train_model

if __name__ == '__main__':
    # Можно настроить параметры здесь:
    train_model(
        num_classes=40,
        num_epochs=20,
        batch_size=16,
        lr=0.001,
        num_points=1024,
        model_save_path="pointnet_best.pth"
    )