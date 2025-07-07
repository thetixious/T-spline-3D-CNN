import trimesh
from vedo import Mesh, Points, Plotter, show
import numpy as np

def show_point_cloud(mesh):
    vedo_mesh = Mesh([mesh.vertices, mesh.faces])
    vedo_mesh.c('lightblue').alpha(0.5).linewidth(0.5)

    sampled_points = mesh.sample(2048)
    vedo_points = Points(sampled_points, r=4, c='red')

    plot1 = Plotter(title="1. Исходная 3D-модель", axes=1)
    plot1.show(vedo_mesh, viewup='z')

    plot2 = Plotter(title="2. Облако точек", axes=1)
    plot2.show(vedo_points, viewup='z')

def show_mesh(trimesh_mesh, color):

    vertices, faces = trimesh_mesh.vertices, trimesh_mesh.faces

    vedo_mesh = Mesh([vertices, faces])
    vedo_mesh.c(color).alpha(1).linewidth(1)

    plotter = Plotter(title="3D Модель", axes=1)
    plotter.show(vedo_mesh, viewup="z", interactive=True)

def show_voxelizated_mesh(mesh):
    mesh.apply_translation(-mesh.bounds[0])
    mesh.apply_scale(1.0 / mesh.extents.max())

    voxelized = mesh.voxelized(pitch=0.025)  # увеличь точность

    voxel_mesh = voxelized.as_boxes()
    show(voxel_mesh, axes=1, bg='white')


def show_deformed_mesh(trimesh_mesh, color='orange', noise_strength=0.0001):
    vertices, faces = trimesh_mesh.vertices, trimesh_mesh.faces

    bbox_size = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    noise_level = noise_strength * bbox_size

    noisy_vertices = vertices + np.random.normal(scale=noise_level, size=vertices.shape)

    vedo_mesh = Mesh([noisy_vertices, faces])
    vedo_mesh.c(color).alpha(1).linewidth(1)

    plotter = Plotter(title="Меш с шумом", axes=1)
    plotter.show(vedo_mesh, viewup='z')
    plotter.close()



def show_point_cloud_with_highlights(mesh, n_total=10048, n_highlight=25):
    sampled_points = mesh.sample(n_total)

    highlight_indices = np.random.choice(n_total, size=n_highlight, replace=False)
    highlight_points = sampled_points[highlight_indices]
    other_indices = list(set(range(n_total)) - set(highlight_indices))
    other_points = sampled_points[other_indices]

    vedo_points_main = Points(other_points, r=4, c='gray', alpha=0.6)
    vedo_points_highlight = Points(highlight_points, r=10, c='red', alpha=1)

    plotter = Plotter(title="Облако точек с подсвеченными областями", axes=1)
    plotter.show(vedo_points_main, vedo_points_highlight, viewup='z', interactive=True)



mesh = trimesh.load('/Users/tix/itmo/diploma/native_dataset/ModelNet40/person/test/person_0108.obj')
show_point_cloud(mesh)
show_mesh(mesh, color="tomato")
show_point_cloud_with_highlights(mesh)
show_deformed_mesh(mesh, color='tomato', noise_strength=0.02)