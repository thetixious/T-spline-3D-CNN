import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

class CachedModelNet40Dataset(Dataset):
    def __init__(self, root_dir, num_points=1024, split='train', cache_dir='cache'):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.cache_dir = os.path.join(root_dir, cache_dir, split)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.files = []
        self.classes = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            split_path = os.path.join(class_path, split)
            if not os.path.isdir(split_path):
                continue

            self.classes.append(class_name)

            for file in os.listdir(split_path):
                if file.endswith('.obj'):
                    obj_path = os.path.join(split_path, file)
                    npy_path = os.path.join(self.cache_dir, f"{class_name}_{file}.npy")
                    self.files.append((obj_path, npy_path, class_name))

        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(self.classes)))}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj_path, npy_path, class_name = self.files[idx]
        label = self.class_to_idx[class_name]

        if os.path.exists(npy_path):
            points = np.load(npy_path)
        else:
            mesh = trimesh.load(obj_path, process=False)
            points = np.array(mesh.vertices)
            np.save(npy_path, points)

        # Приведение к num_points
        if points.shape[0] < self.num_points:
            diff = self.num_points - points.shape[0]
            points = np.concatenate([points, points[np.random.choice(points.shape[0], diff)]])
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice]

        # Нормализация
        points = points - np.mean(points, axis=0)
        points = points / np.max(np.linalg.norm(points, axis=1))

        return torch.from_numpy(points).float(), label

def get_dataloaders(
    train_path="data/train_data.npz",
    test_path="data/test_data.npz",
    train_cache="data/train_cache.pkl",
    test_cache="data/test_cache.pkl",
    batch_size=16,
    num_points=1024,
    train_transform=None,
    test_transform=None
):
    train_dataset = CachedModelNet40Dataset(
        train_path, cache_path=train_cache, num_points=num_points, transform=train_transform
    )
    test_dataset = CachedModelNet40Dataset(
        test_path, cache_path=test_cache, num_points=num_points, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader