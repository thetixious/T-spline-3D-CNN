import trimesh
import numpy as np

def sample_points_from_mesh(mesh_path, num_points=1024):
    """
    Загружает 3D-модель и возвращает point cloud с равномерным сэмплированием.
    """
    mesh = trimesh.load(mesh_path)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)

def batch_sample_from_dir(input_dir, output_dir, num_points=1024, ext=".obj"):
    """
    Сэмплирует point cloud для всех моделей в директории и сохраняет их в npy.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(ext):
            mesh_path = os.path.join(input_dir, fname)
            points = sample_points_from_mesh(mesh_path, num_points=num_points)
            out_path = os.path.join(output_dir, fname.replace(ext, ".npy"))
            np.save(out_path, points)
            print(f"Saved: {out_path}")
