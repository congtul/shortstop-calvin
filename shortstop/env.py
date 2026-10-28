import numpy as np


class Obstacle:
    """Sphere (3D) or circle (2D) obstacle, shared by the 2D prototype and CALVIN."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def contains(self, point):
        return np.linalg.norm(np.asarray(point) - self.center) <= self.radius
