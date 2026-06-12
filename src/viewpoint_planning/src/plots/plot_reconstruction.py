"""
Reconstruction comparison plot.
Left : reconstructed TARGET voxels (blue) — bunny only, background removed.
Right: ground-truth mesh (red).

Why this version fixes the "floating blob + vertical line" artefact
-------------------------------------------------------------------
The grid stores background/occluder voxels too. Those must NOT appear in the
reconstruction figure. We remove them in TWO independent ways so the plot is
clean no matter what the caller passes:

  1. If the caller passes `voxel_class_ids`, we keep only class == 0 (the
     target/bunny). This is the correct, semantic filter.
  2. We ALSO clip to the mesh bounding box as a geometric safety net. The old
     version only did (2), and the stray vertical line happened to fall inside
     the bunny's x/y bounds, so it survived. Doing (1) first removes it.

If `target_voxels` is already filtered (class 0 only) — which is what the
fixed planners now store — step (1) is a no-op and everything still works.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (needed for 3d proj)


def plot_reconstruction_comparison(
    target_voxels,
    mesh_coordinates,
    save_path=None,
    method_label='RH-NBV',
    voxel_class_ids=None,     # optional (N,) array of semantic class ids
    elev=20,
    azim=-60,
):
    """
    Side-by-side 3D scatter: reconstructed target voxels vs ground-truth mesh.

    Args:
        target_voxels    : (N, 3) reconstructed voxel positions
        mesh_coordinates : (M, 3) ground-truth mesh points
        save_path        : PNG output path (optional)
        method_label     : label shown in subplot title and figure title
        voxel_class_ids  : (N,) semantic class id per voxel; if given, only
                           class 0 (target) voxels are plotted
        elev, azim       : 3D view angle
    """
    # Guard: nothing to plot yet
    if (target_voxels is None
            or not isinstance(target_voxels, np.ndarray)
            or target_voxels.ndim < 2
            or len(target_voxels) == 0):
        print("Reconstruction plot skipped: no target voxels detected.")
        return None, None

    mesh_arr  = np.asarray(mesh_coordinates)
    voxel_arr = np.asarray(target_voxels)
    n_raw     = len(voxel_arr)

    # ── Filter 1: semantic class (keep only target class 0) ──────────
    if voxel_class_ids is not None:
        cls = np.asarray(voxel_class_ids)
        if len(cls) == len(voxel_arr):
            keep = (np.round(cls).astype(int) == 0)
            voxel_arr = voxel_arr[keep]

    # ── Filter 2: geometric clip to mesh bounding box (+ margin) ─────
    # Safety net for any background voxel that slipped through (or when no
    # class ids are available). Margin keeps legitimate surface voxels.
    margin = 0.03
    lo = mesh_arr.min(axis=0) - margin
    hi = mesh_arr.max(axis=0) + margin
    in_bounds = np.all((voxel_arr >= lo) & (voxel_arr <= hi), axis=1)
    voxel_filtered = voxel_arr[in_bounds]

    n_removed = n_raw - len(voxel_filtered)
    if n_removed > 0:
        print(f"[ReconPlot] Removed {n_removed} non-target/out-of-bounds voxels "
              f"({len(voxel_filtered):,} target voxels remain)")

    # ── Figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"{method_label} Reconstruction vs. Ground Truth",
                 fontsize=13, fontweight='bold', y=1.01)

    # Left: reconstructed voxels
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=elev, azim=azim)
    if len(voxel_filtered) > 0:
        ax1.scatter(
            voxel_filtered[:, 0], voxel_filtered[:, 1], voxel_filtered[:, 2],
            c='blue', s=1, alpha=0.6, rasterized=True,
        )
    ax1.set_title(
        f"Target Voxels Point Cloud\n"
        f"({method_label}: {len(voxel_filtered):,} target voxels)",
        fontsize=11, fontweight='bold',
    )
    ax1.set_xlabel('X (m)', fontsize=9)
    ax1.set_ylabel('Y (m)', fontsize=9)
    ax1.set_zlabel('Z (m)', fontsize=9)

    # Right: ground-truth mesh
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=elev, azim=azim)
    ax2.scatter(
        mesh_arr[:, 0], mesh_arr[:, 1], mesh_arr[:, 2],
        c='red', s=1, alpha=0.6, rasterized=True,
    )
    ax2.set_title(
        f"Mesh Coordinates Point Cloud\n(Ground truth: {len(mesh_arr):,} points)",
        fontsize=11, fontweight='bold',
    )
    ax2.set_xlabel('X (m)', fontsize=9)
    ax2.set_ylabel('Y (m)', fontsize=9)
    ax2.set_zlabel('Z (m)', fontsize=9)

    # Equal aspect on both axes using mesh bounds
    mid  = mesh_arr.mean(axis=0)
    span = max((mesh_arr.max(axis=0) - mesh_arr.min(axis=0)).max() / 2, 0.10)
    for ax in [ax1, ax2]:
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)

    return fig, (ax1, ax2)


# ----------------------------------------------------------------------
# Per-iteration reconstruction evolution.
# Added so supervisors can see HOW MUCH the reconstruction changed at each
# viewpoint (not just the final result). Two outputs:
#   * plot_reconstruction_evolution_grid  -> one figure, all iterations side by
#                                            side, GT in the last panel.
#   * plot_reconstruction_single_iter     -> one figure per iteration (calls the
#                                            existing comparison plot).
# Both reuse the SAME semantic + bounding-box filtering as the comparison plot,
# so the cleaned target-only point cloud is identical.
# ----------------------------------------------------------------------
def _filter_target_voxels(target_voxels, mesh_arr, voxel_class_ids=None,
                          margin=0.03):
    """Same two-stage filter the comparison plot uses (class 0 + bbox clip)."""
    if (target_voxels is None
            or not isinstance(target_voxels, np.ndarray)
            or target_voxels.ndim < 2
            or len(target_voxels) == 0):
        return np.zeros((0, 3))
    voxel_arr = np.asarray(target_voxels)
    if voxel_class_ids is not None:
        cls = np.asarray(voxel_class_ids)
        if len(cls) == len(voxel_arr):
            voxel_arr = voxel_arr[np.round(cls).astype(int) == 0]
    if len(voxel_arr) == 0:
        return np.zeros((0, 3))
    lo = mesh_arr.min(axis=0) - margin
    hi = mesh_arr.max(axis=0) + margin
    in_bounds = np.all((voxel_arr >= lo) & (voxel_arr <= hi), axis=1)
    return voxel_arr[in_bounds]


def plot_reconstruction_evolution_grid(
    voxel_snapshots,
    mesh_coordinates,
    save_path=None,
    method_label='RH-NBV',
    elev=20,
    azim=-60,
    show_gt_panel=True,
):
    """
    One figure showing the reconstructed target point cloud after each
    iteration, so the growth/change across viewpoints is visible at a glance.

    Args:
        voxel_snapshots : list of (N_i, 3) arrays — reconstructed target voxels
                          after iteration i (i = 1..K). Snapshot for iteration i
                          should be the cumulative reconstruction at that point.
        mesh_coordinates: (M, 3) ground-truth mesh points.
        save_path       : PNG output path (optional).
        method_label    : label shown in the figure title.
        show_gt_panel   : if True, append a final panel with the GT mesh (red).
    """
    mesh_arr = np.asarray(mesh_coordinates)
    n_iters  = len(voxel_snapshots)
    if n_iters == 0:
        print("Evolution plot skipped: no snapshots.")
        return None

    n_panels = n_iters + (1 if show_gt_panel else 0)
    ncols    = min(3, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))

    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))
    fig.suptitle(f"{method_label} Reconstruction Evolution per Viewpoint",
                 fontsize=14, fontweight='bold', y=1.005)

    # Shared equal-aspect bounds from the mesh, so panels are comparable.
    mid  = mesh_arr.mean(axis=0)
    span = max((mesh_arr.max(axis=0) - mesh_arr.min(axis=0)).max() / 2, 0.10)

    def _style(ax):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)

    prev_count = 0
    for i, snap in enumerate(voxel_snapshots):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        filt = _filter_target_voxels(snap, mesh_arr)
        count = len(filt)
        if count > 0:
            ax.scatter(filt[:, 0], filt[:, 1], filt[:, 2],
                       c='blue', s=1, alpha=0.6, rasterized=True)
        # delta vs previous iteration makes "how much changed" explicit
        delta = count - prev_count
        sign  = f"+{delta}" if delta >= 0 else f"{delta}"
        ax.set_title(f"View {i + 1}\n{count:,} voxels ({sign})",
                     fontsize=10, fontweight='bold')
        _style(ax)
        prev_count = count

    if show_gt_panel:
        ax = fig.add_subplot(nrows, ncols, n_panels, projection='3d')
        ax.scatter(mesh_arr[:, 0], mesh_arr[:, 1], mesh_arr[:, 2],
                   c='red', s=1, alpha=0.6, rasterized=True)
        ax.set_title(f"Ground Truth\n({len(mesh_arr):,} points)",
                     fontsize=10, fontweight='bold')
        _style(ax)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    return fig


def plot_reconstruction_single_iter(
    target_voxels,
    mesh_coordinates,
    iteration,
    save_path=None,
    method_label='RH-NBV',
    **kwargs,
):
    """
    Per-iteration reconstruction-vs-GT figure (thin wrapper around
    plot_reconstruction_comparison) so each viewpoint gets its own file.
    """
    label = f"{method_label} (View {iteration})"
    return plot_reconstruction_comparison(
        target_voxels=target_voxels,
        mesh_coordinates=mesh_coordinates,
        save_path=save_path,
        method_label=label,
        **kwargs,
    )
