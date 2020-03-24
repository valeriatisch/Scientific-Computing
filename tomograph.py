
from lib import trace, grid_intersect
import numpy as np


def take_measurement(n_grid: np.int, n_rays: np.int, r_theta: np.float64) -> (
np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Take a measurement with the tomograph from direction r_theta.


    Arguments:
    n_grid: number of cells of grid in each direction
    n_rays: number of parallel rays
    r_theta: direction of rays (in radians)

    Return:
    intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    ray_indices: indices of rays that intersect a cell
    isect_indices: indices of intersected cells
    lengths: lengths of segments in intersected cells

    The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

    Raised Exceptions:
    -

    Side Effects:
    -
    """

    # compute ray direction in Cartesian coordinates
    cs = np.cos(r_theta)
    sn = np.sin(r_theta)
    r_dir = np.array([-cs, -sn])

    # compute start positions for rays
    r_pos = np.zeros((n_rays, 2))
    for i, g in enumerate(np.linspace(-0.99, 0.99, n_rays)):
        r_pos[i] = np.array([cs - sn * g, sn + cs * g])
    else:
        r_pos[0] = np.array([cs, sn])

    # compute measures intensities for each ray
    intensities = np.zeros(n_rays)
    for i, rs in enumerate(r_pos):
        intensities[i] = trace(rs, r_dir)
    # take exponential fall off into account
    intensities = np.log(1.0 / intensities)

    # compute traversal distance in each grid cell
    ray_indices, isect_indices, lengths = grid_intersect(n_grid, r_pos, r_dir)

    return intensities, ray_indices, isect_indices, lengths


