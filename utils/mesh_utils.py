import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
import open3d as o3d

def is_mesh_valid(mesh):
    """
    Check validity of pytorch3D mesh

    Args:
        mesh: pytorch3D mesh

    Returns:
        validity: validity of the mesh
    """
    validity = True

    # Check if the mesh is not empty
    if mesh.isempty():
        validity = False

    # Check if vertices in the mesh are valid
    verts = mesh.verts_packed()
    if not torch.isfinite(verts).all() or torch.isnan(verts).all():
        validity = False

    # Check if vertex normals in the mesh are valid
    v_normals = mesh.verts_normals_packed()
    if not torch.isfinite(v_normals).all() or torch.isnan(v_normals).all():
        validity = False

    # Check if face normals in the mesh are valid
    f_normals = mesh.faces_normals_packed()
    if not torch.isfinite(f_normals).all() or torch.isnan(f_normals).all():
        validity = False

    return validity

def normalize_mesh(verts, faces):
    """
    Normalize and center input mesh to fit in a sphere of radius 1 centered at (0,0,0)

    Args:
        mesh: pytorch3D mesh

    Returns:
        mesh, faces, verts, edges, v_normals, f_normals: normalized pytorch3D mesh and other mesh
        information
    """
    verts = verts - verts.mean(0)
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    mesh = Meshes(verts=[verts], faces=[faces])
    faces = mesh.faces_packed().squeeze(0)
    verts = mesh.verts_packed().squeeze(0)
    edges = mesh.edges_packed().squeeze(0)
    v_normals = mesh.verts_normals_packed().squeeze(0)
    f_normals = mesh.faces_normals_packed().squeeze(0)

    return mesh, faces, verts, edges, v_normals, f_normals


def pytorch3D_mesh(f_path, device):
    """
    Read pytorch3D mesh from path

    Args:
        f_path: obj file path

    Returns:
        mesh, faces, verts, edges, v_normals, f_normals: pytorch3D mesh and other mesh information
    """
    if not f_path.endswith('.obj'):
        raise ValueError('Input files should be in obj format.')
    mesh = load_objs_as_meshes([f_path], device)
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    v_normals = mesh.verts_normals_packed()
    f_normals = mesh.faces_normals_packed()
    return mesh, faces, verts, edges, v_normals, f_normals
