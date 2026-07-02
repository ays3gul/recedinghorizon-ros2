"""
occluder_geometry.py — single source of truth for occluder visualisation.

Bunny at (0.50, -0.30, 1.0).  Camera starts at y=+0.30 (standoff 0.60 m).
Panel sizes (SDF <size> = x y z, half = size/2):
  panel_front      : 0.20x0.02x0.20  -> half (0.10, 0.01, 0.10)
  panel_front_low  : 0.26x0.02x0.16  -> half (0.13, 0.01, 0.08)
  panel_side       : 0.02x0.22x0.22  -> half (0.01, 0.11, 0.11)
  panel_side_low   : 0.02x0.22x0.16  -> half (0.01, 0.11, 0.08)
  panel_back       : 0.20x0.02x0.22  -> half (0.10, 0.01, 0.11)
  panel_tunnel     : 0.02x0.10x0.20  -> half (0.01, 0.05, 0.10)

Format: OCCLUDERS[scenario] = list of (center_xyz, half_extent_xyz).
Keep in sync with spawn_*_occlusion() in viewpoint_planning.py.
"""

OCCLUDERS = {
    "none": [],

    # frontal: back face at bunny front face y=-0.229, no penetration.
    # Camera hits panel front (y=-0.209) before bunny (y=-0.229). 2 cm clearance.
    "frontal": [
        ([0.50, -0.219, 1.07], (0.100, 0.010, 0.100)),
    ],

    # half_box: two side walls + back wall, front and top open.
    # panel_side (0.02x0.22x0.22) at x=0.40 and x=0.64, centred on bunny y=-0.30.
    # panel_back (0.20x0.02x0.22) at y=-0.41 (11 cm behind bunny centre).
    "half_box": [
        ([0.40, -0.30, 1.10], (0.010, 0.110, 0.110)),  # left  (X-)
        ([0.64, -0.30, 1.10], (0.010, 0.110, 0.110)),  # right (X+)
        ([0.50, -0.41, 1.10], (0.100, 0.010, 0.110)),  # back  (Y-)
    ],

    # tunnel: front + back panel pairs at x=0.425/0.575, matching
    # ur5e_world_tunnel.sdf (panel size 0.07x0.02x0.30 -> half 0.035x0.010x0.150).
    "tunnel": [
        ([0.425, -0.23, 1.07], (0.035, 0.010, 0.150)),  # left front
        ([0.575, -0.23, 1.07], (0.035, 0.010, 0.150)),  # right front
        ([0.425, -0.39, 1.07], (0.035, 0.010, 0.150)),  # left back
        ([0.575, -0.39, 1.07], (0.035, 0.010, 0.150)),  # right back
    ],

    # covered_well: four short walls + top cover at z=1.185.
    "covered_well": [
        ([0.40, -0.30, 1.08],  (0.010, 0.110, 0.080)),  # left  (X-)
        ([0.64, -0.30, 1.08],  (0.010, 0.110, 0.080)),  # right (X+)
        ([0.52, -0.18, 1.08],  (0.130, 0.010, 0.080)),  # front (Y+)
        ([0.52, -0.42, 1.08],  (0.130, 0.010, 0.080)),  # back  (Y-)
        ([0.52, -0.30, 1.185], (0.130, 0.110, 0.010)),  # top
    ],

    # well: four short walls (top open). panel_side_low + panel_front_low at z=1.08.
    # Interior: x[0.41,0.63] y[-0.41,-0.19] z[1.00,1.16].
    "well": [
        ([0.40, -0.30, 1.08], (0.010, 0.110, 0.080)),  # left  (X-)
        ([0.64, -0.30, 1.08], (0.010, 0.110, 0.080)),  # right (X+)
        ([0.52, -0.18, 1.08], (0.130, 0.010, 0.080)),  # front (Y+)
        ([0.52, -0.42, 1.08], (0.130, 0.010, 0.080)),  # back  (Y-)
    ],
}


def get_occluders(scenario):
    """Return the (center, half_extent) list for a scenario, or [] if unknown."""
    return OCCLUDERS.get(scenario, [])
