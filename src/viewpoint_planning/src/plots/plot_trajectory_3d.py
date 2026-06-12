"""
3D Trajectory plot — Burusa et al. (ICRA 2024) style.
RH-NBV only: bunny mesh + occluder box(es) + trajectory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Single source of truth for occluder geometry, kept in sync with the spawn
# positions in viewpoint_planning.py. Values are (center_xyz, half_extent_xyz).
try:
    from occluder_geometry import OCCLUDERS
except ImportError:
    from plots.occluder_geometry import OCCLUDERS


def _draw_box(ax, center, half_ext, color='goldenrod', alpha=0.45):
    cx, cy, cz = center
    hx, hy, hz = half_ext
    ax.bar3d(
        cx - hx, cy - hy, cz - hz,
        2 * hx, 2 * hy, 2 * hz,
        color=color, alpha=alpha,
        edgecolor='saddlebrown', linewidth=0.6,
        zsort='average',
    )


def plot_3d_trajectory(
    trail,
    mesh_coordinates,
    occlusion_type='none',
    save_path=None,
    title=None,
    method_label='RH-NBV',
    elev=20,
    azim=-60,
):
    trail_arr = np.asarray(trail)
    mesh_arr  = np.asarray(mesh_coordinates)

    if title is None:
        title = f"{method_label} 3D Trajectory (Occlusion: {occlusion_type})"

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Bunny mesh
    ax.scatter(
        mesh_arr[:, 0], mesh_arr[:, 1], mesh_arr[:, 2],
        c='red', s=1, alpha=0.5, rasterized=True,
    )

    # Occluder(s)
    for center, half_ext in OCCLUDERS.get(occlusion_type, []):
        _draw_box(ax, center, half_ext)

    # Trajectory
    ax.plot(
        trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2],
        color='green', linestyle='--', linewidth=1.4,
        marker='x', markersize=7, markeredgewidth=1.5,
        label=method_label, zorder=4,
    )
    # Start point
    ax.scatter(
        trail_arr[0, 0], trail_arr[0, 1], trail_arr[0, 2],
        color='red', s=60, marker='o', zorder=5,
    )

    ax.set_xlabel('X (m)', fontsize=11, labelpad=6)
    ax.set_ylabel('Y (m)', fontsize=11, labelpad=6)
    ax.set_zlabel('Z (m)', fontsize=11, labelpad=6)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.7)

    # Equal aspect
    all_pts = np.vstack([trail_arr, mesh_arr])
    for center, _ in OCCLUDERS.get(occlusion_type, []):
        all_pts = np.vstack([all_pts, np.atleast_2d(center)])
    mid  = all_pts.mean(axis=0)
    span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2
    span = max(span, 0.20)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)

    return fig, ax
