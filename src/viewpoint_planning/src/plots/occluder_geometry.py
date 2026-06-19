"""
occluder_geometry.py — single source of truth for occluder visualisation.

The actual occluders are spawned in Gazebo by viewpoint_planning.py
(spawn_*_occlusion) and test_gradient_node.py (spawn_occlusion). The plotting
code needs the SAME geometry to draw the occluders correctly in the trajectory
and candidate-sequence figures.

To avoid the drift we kept hitting (plots showing old/wrong boxes), every plot
imports OCCLUDERS from here. If you ever move an occluder, change the spawn
position in viewpoint_planning.py AND the matching entry here.

Format: OCCLUDERS[scenario] = list of (center_xyz, half_extent_xyz).

Notes on coordinates:
  * Panels are spawned with spawn_named_model, which applies NO Z pivot
    offset, so the panel centre is exactly the spawn position.
  * Panels (including the frontal one) are spawned with spawn_named_model,
    which applies NO Z pivot offset.
  * half-extent = half the box side length (SDF <size> / 2).
"""

OCCLUDERS = {
    "none": [],

    # frontal: panel_front (0.20x0.02x0.14 m) at (0.5,-0.15,1.12); no collision →
    # arm passes through freely; depth camera still sees it (visual-only geometry).
    # Panel Z=[1.05,1.19] blocks bunny body; ears (top 3.5cm) and feet (1cm) show.
    "frontal": [
        ([0.50, -0.15, 1.12], (0.100, 0.010, 0.070)),
    ],

    # half_box: sides + back walled, front AND top open. panel_side =
    # 0.02x0.22x0.22, panel_back = 0.20x0.02x0.22.
    "half_box": [
        ([0.40, -0.25, 1.10], (0.010, 0.110, 0.110)),  # left  (X-)
        ([0.64, -0.25, 1.10], (0.010, 0.110, 0.110)),  # right (X+)
        ([0.50, -0.36, 1.10], (0.100, 0.010, 0.110)),  # back  (Y-)
    ],

    # tunnel: two thin panels with a ~12 cm corridor. panel_tunnel =
    # 0.02x0.10x0.20 at Y=-0.25.
    "tunnel": [
        ([0.43, -0.25, 1.10], (0.010, 0.050, 0.100)),  # left
        ([0.57, -0.25, 1.10], (0.010, 0.050, 0.100)),  # right
    ],

    # well: four short walls (only the top is open). panel_side_low =
    # 0.02x0.22x0.16, panel_front_low = 0.26x0.02x0.16, centred at Z=1.08.
    # Y-centred on the bunny at y=-0.25 (was y=-0.40, shifted +0.15 m).
    "well": [
        ([0.40, -0.25, 1.08], (0.010, 0.110, 0.080)),  # left  (X-)
        ([0.64, -0.25, 1.08], (0.010, 0.110, 0.080)),  # right (X+)
        ([0.52, -0.13, 1.08], (0.130, 0.010, 0.080)),  # front (Y+)
        ([0.52, -0.37, 1.08], (0.130, 0.010, 0.080)),  # back  (Y-)
    ],
}


def get_occluders(scenario):
    """Return the (center, half_extent) list for a scenario, or [] if unknown."""
    return OCCLUDERS.get(scenario, [])
