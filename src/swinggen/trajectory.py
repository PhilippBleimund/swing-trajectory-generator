import numpy as np
from .swinggen import *

from abc import ABC, abstractmethod

class Motion(ABC):
    def __init__(self, points) -> None:
        pass

    @abstractmethod
    def __call__(self, t):
        pass

class PushMotion(Motion):
    def __init__(self, points) -> None:
            points = np.asarray(points)
    
            dt_move = 1.0
            v_move = (points[0] - points[2]) / dt_move
    
            pos_move = [
                [
                    {"t": 0.0, "pos": points[2][0], "vel": v_move[0]},
                    {"t": 0.0, "pos": points[2][1], "vel": v_move[1]},
                    {"t": 0.0, "pos": points[2][2], "vel": v_move[2]},
                ],
                [
                    {"t": 1.0, "pos": points[0][0], "vel": v_move[0]},
                    {"t": 1.0, "pos": points[0][1], "vel": v_move[1]},
                    {"t": 1.0, "pos": points[0][2], "vel": v_move[2]},
                ],
            ]
            self.splines_move = gen_xyz(pos_move, gen_poly_auto)
    
    def __call__(self, t):
        t = t % 1.0
        return [spline(t) for spline in self.splines_move]
class SwingMotion(Motion):
    def __init__(self, points) -> None:
        points = np.asarray(points)

        dt_move = 1.0
        v_move = (points[0] - points[2]) / dt_move

        d = points[2] - points[0]
        norm_d = np.linalg.norm(d)
        if norm_d == 0:
            dir_02 = np.zeros(3)
        else:
            dir_02 = d / norm_d
        v_mid = dir_02 * 1.0

        pos_swing = [
            [  # at t = 0
                {"t": 0.0, "pos": points[0][0], "vel": v_move[0]},
                {"t": 0.0, "pos": points[0][1], "vel": v_move[1]},
                {"t": 0.0, "pos": points[0][2], "vel": v_move[2]},
            ],
            [  # mid swing at t = 0.25 
                {"t": 0.5, "pos": points[1][0], "vel": v_mid[0]},
                {"t": 0.5, "pos": points[1][1], "vel": v_mid[1]},
                {"t": 0.5, "pos": points[1][2], "vel": v_mid[2]},
            ],
            [  # end of swing at t = 0.5
                {"t": 1.0, "pos": points[2][0], "vel": v_move[0]},
                {"t": 1.0, "pos": points[2][1], "vel": v_move[1]},
                {"t": 1.0, "pos": points[2][2], "vel": v_move[2]},
            ],
        ]
        self.splines_swing = gen_xyz(pos_swing, gen_poly_auto)

    def __call__(self, t):
        t = t % 1.0
        return [spline(t) for spline in self.splines_swing]

class FullMotion(Motion):
    """Periodic (T = 1.0) foot trajectory through three 3D waypoints.

    Args:
        points: Sequence of three (x, y, z) waypoints.

    Usage:
        >>> leg = FullMotion([(0.0, 0.0, 0.0), (0.05, 0.0, 0.03), (0.1, 0.0, 0.0)])
        >>> x, y, z = leg(0.25)   # position at mid swing
    """


    def __init__(self, points) -> None:
        points = np.asarray(points)

        dt_move = 1.0 - 0.5
        v_move = (points[0] - points[2]) / dt_move

        d = points[2] - points[0]
        norm_d = np.linalg.norm(d)
        if norm_d == 0:
            dir_02 = np.zeros(3)
        else:
            dir_02 = d / norm_d
        v_mid = dir_02 * 1.0

        pos_swing = [
            [  # at t = 0
                {"t": 0.0, "pos": points[0][0], "vel": v_move[0]},
                {"t": 0.0, "pos": points[0][1], "vel": v_move[1]},
                {"t": 0.0, "pos": points[0][2], "vel": v_move[2]},
            ],
            [  # mid swing at t = 0.25 
                {"t": 0.25, "pos": points[1][0], "vel": v_mid[0]},
                {"t": 0.25, "pos": points[1][1], "vel": v_mid[1]},
                {"t": 0.25, "pos": points[1][2], "vel": v_mid[2]},
            ],
            [  # end of swing at t = 0.5
                {"t": 0.5, "pos": points[2][0], "vel": v_move[0]},
                {"t": 0.5, "pos": points[2][1], "vel": v_move[1]},
                {"t": 0.5, "pos": points[2][2], "vel": v_move[2]},
            ],
        ]
        self.splines_swing = gen_xyz(pos_swing, gen_poly_auto)

        pos_move = [
            [
                {"t": 0.5, "pos": points[2][0], "vel": v_move[0]},
                {"t": 0.5, "pos": points[2][1], "vel": v_move[1]},
                {"t": 0.5, "pos": points[2][2], "vel": v_move[2]},
            ],
            [
                {"t": 1.0, "pos": points[0][0], "vel": v_move[0]},
                {"t": 1.0, "pos": points[0][1], "vel": v_move[1]},
                {"t": 1.0, "pos": points[0][2], "vel": v_move[2]},
            ],
        ]
        self.splines_move = gen_xyz(pos_move, gen_poly_auto)

    def __call__(self, t):
        t = t % 1.0
        if t <= 0.5:
            return [spline(t) for spline in self.splines_swing]
        else:
            return [spline(t) for spline in self.splines_move]