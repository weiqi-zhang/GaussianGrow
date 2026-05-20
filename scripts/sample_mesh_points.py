import trimesh
import numpy as np


def sample_points(in_path, out_path, num_samples):
    mesh = trimesh.load(in_path)

    if not mesh.is_watertight:
        mesh = mesh.subdivide()

    faces = mesh.faces
    vertices = mesh.vertices

    areas = mesh.area_faces

    cdf = np.cumsum(areas)
    cdf /= cdf[-1]

    def sample_mesh_vectorized(mesh, num_samples):
        r = np.random.rand(num_samples)
        triangle_indices = np.searchsorted(cdf, r)

        triangle_indices = np.clip(triangle_indices, 0, len(faces) - 1)

        selected_faces = faces[triangle_indices]
        v0 = vertices[selected_faces[:, 0]]
        v1 = vertices[selected_faces[:, 1]]
        v2 = vertices[selected_faces[:, 2]]

        u = np.random.rand(num_samples)
        v = np.random.rand(num_samples)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v

        samples = (v0.T * u + v1.T * v + v2.T * w).T

        n0 = mesh.vertex_normals[selected_faces[:, 0]]
        n1 = mesh.vertex_normals[selected_faces[:, 1]]
        n2 = mesh.vertex_normals[selected_faces[:, 2]]
        normals = (n0.T * u + n1.T * v + n2.T * w).T
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)

        return samples, normals

    print("Sampling mesh points...")
    sampled_points, sampled_normals = sample_mesh_vectorized(mesh, num_samples)
    print("Sampled points:", sampled_points.shape)
    print("Sampled normals:", sampled_normals.shape)

    def save_ply(filename, points, normals):
        num_vertices = points.shape[0]
        header = f'''ply
    format ascii 1.0
    element vertex {num_vertices}
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    end_header
    '''
        print(f"Saving PLY to {filename}...")
        with open(filename, 'w') as f:
            f.write(header)
            data = np.hstack((points, normals))
            np.savetxt(f, data, fmt='%.6f %.6f %.6f %.6f %.6f %.6f')
        print("PLY saved.")

    save_ply(f"{out_path}/example.ply", sampled_points, sampled_normals)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample oriented points from a mesh and write example.ply.")
    parser.add_argument("--input_path", required=True, help="Input mesh path.")
    parser.add_argument("--output_dir", required=True, help="Output directory for example.ply.")
    parser.add_argument("--num_samples", type=int, default=400000, help="Number of points to sample.")
    args = parser.parse_args()

    sample_points(args.input_path, args.output_dir, args.num_samples)
