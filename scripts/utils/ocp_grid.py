import numpy as np
import numpy as np
from numba import jit


@jit(nopython=True)
def update_occupancy_grid(occupancy_grid, lidar_data, robot_position, grid_resolution, yaw):
    for angle_index, distance in enumerate(lidar_data):
        angle = (angle_index * np.pi / 180 + yaw) % (2 * np.pi)

        # Convert polar coordinates to grid coordinates
        end_x = int(robot_position[0] + (distance * np.cos(angle) / grid_resolution))
        end_y = int(robot_position[1] + (distance * np.sin(angle) / grid_resolution))

        # Use Bresenham's line algorithm to update cells along the beam path
        x0, y0 = robot_position
        dx = abs(end_x - x0)
        dy = abs(end_y - y0)
        sx = 1 if x0 < end_x else -1
        sy = 1 if y0 < end_y else -1
        err = dx - dy

        while (x0 != end_x or y0 != end_y) and 0 <= x0 < occupancy_grid.shape[0] and 0 <= y0 < occupancy_grid.shape[1]:
            occupancy_grid[x0, y0] = 10
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        if distance <= 4.75:  # Assumed maximum distance threshold for LiDAR
            if 0 <= end_x < occupancy_grid.shape[0] and 0 <= end_y < occupancy_grid.shape[1]:
                occupancy_grid[end_x, end_y] = 0

    return occupancy_grid


