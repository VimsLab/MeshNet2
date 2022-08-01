"""
Collection of sanity check functions
"""
import math
import numpy as np
from numpy.random import choice
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_Input(verts, faces, centers, corners, targets):
    """
    Visually verify inputs to MeshNet2
    Args:
        verts: padded mesh vertices
        [num_meshes, ?, 3]

        faces: faces in mesh/es
        [num_meshes, num_faces, 3]

        centers: face center of mesh/es
        [num_meshes, num_faces, 3]

        corners: corners of mesh/es
        [num_meshes, num_faces, 3, 3]

        target: class labels
        [num_meshes]
    """
    num_meshes = faces.shape[0]
    # Randomly select 4 meshes in a batch
    visualize_at = choice(range(num_meshes), 4, replace=False)
    # Zoom to visualize centers and corners
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for enum_idx, m_idx in enumerate(visualize_at):
        vert = verts[m_idx].cpu().numpy()
        triangles = faces[m_idx].cpu().numpy()
        center = centers[m_idx].cpu().numpy()
        corner = corners[m_idx].cpu().numpy().reshape(-1, 3)
        target = str(targets[m_idx].cpu().numpy())
        # Plot
        ax = fig.add_subplot(2, 2, enum_idx + 1, projection='3d')
        ax.scatter(center[:, 0], center[:, 1], center[:, 2],
                   color='red',
                   edgecolor='black',
                   s=5,
                   alpha=1)
        ax.scatter(corner[:, 0], corner[:, 1], corner[:, 2],
                   color='blue',
                   edgecolor='black',
                   s=5,
                   alpha=1)
        ax.plot_trisurf(vert[:, 0], vert[:, 1], vert[:, 2],
                        triangles=triangles,
                        color='whitesmoke',
                        edgecolor="gainsboro",
                        linewidth=0.5,
                        alpha=0.1)
        # Display settings
        ax.title.set_text(target)
        ax.view_init(elev=60., azim=45.)
        ax.set_aspect('auto')
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
    plt.show()

def check_ConvSurface(verts,
                      faces,
                      centers,
                      points_neighbor,
                      neighbor_direction,
                      neighbor_direction_norm,
                      support_direction_norm):
    """
    Visually verify surface points on ConvSurface layers.
    Note: Visualizations won't work with Multi-GPU training
    Args:
        meshes: pytorch 3D mesh/es
        [num_meshes]

        faces: faces in mesh/es
        [num_meshes, num_faces, 3]

        centers: face center of mesh/es
        [num_meshes, num_faces, 3]

        points_neighbor: sampled point cloud on the surface of neighbor faces
        [num_meshes, num_faces, num_neighbor, 3]

        neighbor_direction: directional vector between centers and point_neighbors
        [num_meshes, num_faces, num_neighbor, 3]

        neighbor_direction_norm: normalized directional vector between centers and point_neighbors
        [num_meshes, num_faces, num_neighbor, 3]

        support_direction_norm: normalized kernel
        [3, num_kernel]
    """
    # m_idx = choice(range(num_meshes), 1, replace=False)
    m_idx = 0
    verts = verts[m_idx].squeeze(0).cpu().numpy()
    triangles = faces[m_idx].squeeze(0).cpu().numpy()
    centers = centers[m_idx].squeeze(0).cpu().numpy()
    point_neighbors = points_neighbor[m_idx].squeeze(0).cpu().numpy()[0:5]
    neighbor_directions = neighbor_direction[m_idx].squeeze(0).cpu().numpy()
    neighbor_direction_norms = neighbor_direction_norm[m_idx].squeeze(0).cpu().numpy()
    support_direction_norm = support_direction_norm.detach().cpu().numpy().transpose(1, 0)[0:5]

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    gs = GridSpec(2, 2, width_ratios=[2, 3])

    for face_idx, point_neighbor in enumerate(point_neighbors):
        ax1 = fig.add_subplot(gs[:, 1], projection='3d')
        center = centers[face_idx, :]
        ax1.scatter(center[0], center[1], center[2],
                    color='red',
                    edgecolor='black',
                    s=15,
                    alpha=1)
        ax1.scatter(point_neighbor[:, 0], point_neighbor[:, 1], point_neighbor[:, 2],
                    color='blue',
                    edgecolor='black',
                    s=15,
                    alpha=1)
        ax1.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                         triangles=triangles,
                         color='whitesmoke',
                         edgecolor=(0, 0, 0, 0.05),
                         linewidth=0.5,
                         alpha=0.1)
        neighbor_direction = neighbor_directions[face_idx]
        for direction in neighbor_direction:
            ax1.quiver(center[0], center[1], center[2],
                       direction[0], direction[1], direction[2],
                       edgecolor="grey", linewidth=1.5)

        # Display settings
        ax1.view_init(elev=60., azim=45.)
        ax1.set_aspect('auto')
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        ax1.zaxis.set_ticklabels([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.xaxis.zoom(1.2)
        ax1.yaxis.zoom(1.2)
        ax1.yaxis.zoom(1.2)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor('w')
        ax1.yaxis.pane.set_edgecolor('w')
        ax1.zaxis.pane.set_edgecolor('w')
        ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.grid(False)
        csfont = {'fontname':'Times New Roman'}
        ax1.set_title('Varying Local Structures in a Mesh', fontweight="bold", **csfont)
        ax3 = fig.add_subplot(gs[0, 0], projection='3d')
        for direction in support_direction_norm:
            ax3.quiver(0, 0, 0,
                       direction[0], direction[1], direction[2],
                       edgecolor="gainsboro", linewidth=1.5)
        ax3.view_init(elev=60., azim=45.)
        ax3.set_aspect('auto')
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_zlim(-1, 1)
        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])
        ax3.zaxis.set_ticklabels([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.xaxis.pane.fill = False
        ax3.yaxis.pane.fill = False
        ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor('w')
        ax3.yaxis.pane.set_edgecolor('w')
        ax3.zaxis.pane.set_edgecolor('w')
        ax3.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax3.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax3.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax3.grid(False)
        csfont = {'fontname':'Times New Roman'}
        ax3.set_title('Learnable Weight', fontweight="bold", **csfont)
        ax2 = fig.add_subplot(gs[1, 0], projection='3d')
        neighbor_direction_norm = neighbor_direction_norms[face_idx]
        for direction in neighbor_direction_norm:
            p = ax2.quiver(0, 0, 0,
                           direction[0], direction[1], direction[2],
                           edgecolor="gainsboro", linewidth=1.5, cmap='bwr')
        ax2.view_init(elev=60., azim=45.)
        ax2.set_aspect('auto')
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_zlim(-1, 1)
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ax2.zaxis.set_ticklabels([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('w')
        ax2.yaxis.pane.set_edgecolor('w')
        ax2.zaxis.pane.set_edgecolor('w')
        ax2.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax2.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax2.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax2.grid(False)
        csfont = {'fontname':'Times New Roman'}
        cbar = fig.colorbar(p, ax=ax2, ticks=list(range(0, 2)))
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.set_yticklabels(['-ve', '+ve'])
        cbar.ax.set_ylabel('Cosine Similarity', rotation=90, fontsize=15, **csfont)
        ax2.set_title('Local Structure', fontweight="bold", **csfont)

        for n_direction in neighbor_direction_norm:
            for s_direction in support_direction_norm:
                ax3.quiver(0, 0, 0,
                           s_direction[0], s_direction[1], s_direction[2],
                           edgecolor=(1, 0, 1), linewidth=0.5)

                corr = n_direction @ s_direction
                if corr > 0:
                    ax2.quiver(0, 0, 0,
                               n_direction[0], n_direction[1], n_direction[2],
                               edgecolor=(1, 0, 0, (corr+1)/2), linewidth=0.5)
                    plt.pause(0.5)
                    ax2.quiver(0, 0, 0,
                               n_direction[0], n_direction[1], n_direction[2],
                               edgecolor="gainsboro", linewidth=1.5)
                else:
                    ax2.quiver(0, 0, 0,
                               n_direction[0], n_direction[1], n_direction[2],
                               edgecolor=(0, 0, 1, (corr+1)/2), linewidth=0.5)
                    plt.pause(0.5)
                    ax2.quiver(0, 0, 0,
                               n_direction[0], n_direction[1], n_direction[2],
                               edgecolor="gainsboro", linewidth=1.5)

            ax2.quiver(0, 0, 0,
                       n_direction[0], n_direction[1], n_direction[2],
                       edgecolor="gainsboro", linewidth=1.5)

            for s_direction in support_direction_norm:
                ax3.quiver(0, 0, 0,
                           s_direction[0], s_direction[1], s_direction[2],
                           edgecolor="gainsboro", linewidth=1.5)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
