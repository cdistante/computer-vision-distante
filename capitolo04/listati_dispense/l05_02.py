import numpy as np
from itertools import combinations

def checked_points(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4:
        raise ValueError("Atteso array Nx2, N >= 4")
    if not np.isfinite(p).all():
        raise ValueError("Coordinate non finite")
    if len(np.unique(p, axis=0)) < 4:
        raise ValueError("Meno di quattro punti distinti")
    return p

def normalize_points(p):
    center = p.mean(axis=0)
    z = p - center
    mean_dist = np.linalg.norm(z, axis=1).mean()
    if mean_dist <= np.finfo(float).eps:
        raise ValueError("Estensione numericamente nulla")
    scale = np.sqrt(2.0) / mean_dist
    T = np.array([[scale, 0, -scale*center[0]],
                  [0, scale, -scale*center[1]],
                  [0, 0, 1.]])
    return scale*z, T

def check_minimal(p, area_tol=1e-6):
    # Punti gia' centrati e scalati: area adimensionale.
    if len(p) == 4:
        for i, j, k in combinations(range(4), 3):
            a, b = p[j]-p[i], p[k]-p[i]
            twice_area = abs(a[0]*b[1] - a[1]*b[0])
            if twice_area <= area_tol:
                raise ValueError("Terna (quasi) collineare")

