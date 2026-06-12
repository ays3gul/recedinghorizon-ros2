"""
Candidate Sequences Plot — RH-NBV decision visualization.

Shows all K candidate sequences considered at each iteration:
- Rejected sequences: faded gray
- Selected sequence: bold green
- Actually executed first step: red

This visualizes the receding horizon principle: planner considers
K alternatives, picks the best, executes only the first step.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Single source of truth for occluder geometry, shared with the trajectory
# plot and kept in sync with viewpoint_planning.py. Values are
# (center_xyz, half_extent_xyz) — note half-extent, so bar3d uses 2*h below.
try:
    from occluder_geometry import OCCLUDERS
except ImportError:
    from plots.occluder_geometry import OCCLUDERS


def plot_candidate_sequences(
    candidate_history,
    mesh_coordinates,
    occlusion_type='none',
    iteration_to_plot=None,
    save_path=None,
    title=None,
):
    """
    Plot all K candidate sequences for ONE iteration of RH-NBV.
    
    Args:
        candidate_history: vp.rh_planner.candidate_history (list of dicts)
        mesh_coordinates: bunny mesh points
        occlusion_type: 'complex', 'easy', etc.
        iteration_to_plot: which iter to plot (default: middle iteration for 
                          interesting decision-making)
        save_path: optional PNG path
        title: figure title
    """
    if len(candidate_history) == 0:
        print("No candidate history!")
        return None

    if iteration_to_plot is None:
        # Default: 3rd iteration (early enough for active exploration)
        iteration_to_plot = min(2, len(candidate_history) - 1)

    iter_data = candidate_history[iteration_to_plot]
    start_pos = iter_data['start_pos']           # (3,)
    sequences = iter_data['sequences']           # (K, H, 3)
    scores = iter_data['scores']                 # (K,)
    best_idx = iter_data['best_idx']

    K, H, _ = sequences.shape

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1) Bunny mesh — kırmızı scatter dots
    mesh = np.asarray(mesh_coordinates)
    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c='red', s=1, alpha=0.5)

    # 2) Occluders (OCCLUDERS stores half-extents)
    for occ_pos, occ_half in OCCLUDERS.get(occlusion_type, []):
        x, y, z = occ_pos
        hx, hy, hz = occ_half
        ax.bar3d(
            x - hx, y - hy, z - hz, 2 * hx, 2 * hy, 2 * hz,
            color='orange', alpha=0.3, edgecolor='darkorange',
        )

    # 3) Reddedilen sequences — soluk gri
    for k in range(K):
        if k == best_idx:
            continue  # skip the best, draw it on top
        seq = sequences[k]  # (H, 3)
        # Concatenate start + sequence to draw the full path
        path = np.vstack([start_pos[None, :], seq])  # (H+1, 3)
        ax.plot(
            path[:, 0], path[:, 1], path[:, 2],
            color='gray', alpha=0.35, linewidth=1, linestyle='--',
            marker='o', markersize=4,
        )

    # 4) Seçilen sequence — kalın yeşil
    best_seq = sequences[best_idx]
    best_path = np.vstack([start_pos[None, :], best_seq])
    ax.plot(
        best_path[:, 0], best_path[:, 1], best_path[:, 2],
        color='green', linewidth=2.5, marker='o', markersize=8,
        label=f'Selected (J={scores[best_idx]:.2f})',
    )

    # 5) Yürütülen ilk step — kırmızı yıldız
    ax.scatter(
        *best_seq[0], color='red', s=300, marker='*',
        edgecolors='darkred', linewidths=2, zorder=10,
        label='Executed first step',
    )

    # 6) Start position — büyük yeşil kare
    ax.scatter(
        *start_pos, color='darkgreen', s=200, marker='s',
        edgecolors='black', linewidths=2, zorder=10,
        label=f'Start (iter {iteration_to_plot})',
    )

    # Dummy entry for rejected legend
    ax.plot([], [], [], color='gray', alpha=0.4, linestyle='--',
            label=f'Rejected ({K-1} sequences)')

    # Configure
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    if title is None:
        title = (f'RH-NBV Candidate Sequences at Iteration {iteration_to_plot}\n'
                 f'K={K} sequences of H={H} steps — only first step executed')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    # Equal aspect
    all_pts = np.vstack([
        mesh,
        sequences.reshape(-1, 3),
        np.atleast_2d(start_pos),
    ])
    for occ_pos, _ in OCCLUDERS.get(occlusion_type, []):
        all_pts = np.vstack([all_pts, np.atleast_2d(occ_pos)])

    mid = all_pts.mean(axis=0)
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

    return fig, ax


def plot_candidate_sequences_grid(
    candidate_history,
    mesh_coordinates,
    occlusion_type='none',
    iterations_to_plot=None,
    save_path=None,
):
    """
    Multi-iteration grid: shows candidate sequences across multiple iterations.
    Useful to see how the decision-making evolves.
    
    Args:
        iterations_to_plot: list of iteration indices to show (default: [0, 3, 6, 9])
    """
    if iterations_to_plot is None:
        n = len(candidate_history)
        iterations_to_plot = [0, n//4, n//2, 3*n//4][:4]
        iterations_to_plot = [i for i in iterations_to_plot if i < n]

    n_plots = len(iterations_to_plot)
    fig = plt.figure(figsize=(7 * n_plots, 7))

    mesh = np.asarray(mesh_coordinates)

    for idx, iter_num in enumerate(iterations_to_plot):
        ax = fig.add_subplot(1, n_plots, idx + 1, projection='3d')
        iter_data = candidate_history[iter_num]
        start_pos = iter_data['start_pos']
        sequences = iter_data['sequences']
        scores = iter_data['scores']
        best_idx = iter_data['best_idx']
        K, H, _ = sequences.shape

        # Bunny
        ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c='red', s=1, alpha=0.4)

        # Occluders (half-extents)
        for occ_pos, occ_half in OCCLUDERS.get(occlusion_type, []):
            x, y, z = occ_pos
            hx, hy, hz = occ_half
            ax.bar3d(x - hx, y - hy, z - hz, 2 * hx, 2 * hy, 2 * hz,
                     color='orange', alpha=0.3, edgecolor='darkorange')

        # Rejected
        for k in range(K):
            if k == best_idx:
                continue
            path = np.vstack([start_pos[None, :], sequences[k]])
            ax.plot(path[:, 0], path[:, 1], path[:, 2],
                    color='gray', alpha=0.3, linewidth=1, linestyle='--')

        # Selected
        best_path = np.vstack([start_pos[None, :], sequences[best_idx]])
        ax.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2],
                color='green', linewidth=2, marker='o', markersize=5)

        # Executed first step
        ax.scatter(*sequences[best_idx][0], color='red', s=200, marker='*',
                   edgecolors='darkred', linewidths=2, zorder=10)

        # Start
        ax.scatter(*start_pos, color='darkgreen', s=120, marker='s',
                   edgecolors='black', linewidths=1.5, zorder=10)

        ax.set_title(f'Iter {iter_num}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)

        # Equal aspect
        all_pts = np.vstack([
            mesh, sequences.reshape(-1, 3), np.atleast_2d(start_pos)
        ])
        mid = all_pts.mean(axis=0)
        span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 0.20)
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_box_aspect([1, 1, 1])

    fig.suptitle('RH-NBV Decision-Making Across Iterations\n'
                 'Gray dashed = rejected sequences | Green = selected | Red star = executed',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
