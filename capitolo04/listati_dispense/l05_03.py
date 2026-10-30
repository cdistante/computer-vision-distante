import cv2
import numpy as np

H, mask = cv2.findHomography(
    points_a.astype(np.float32),
    points_b.astype(np.float32),
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0,
    confidence=0.99,
    maxIters=5000,
)

if H is None or mask is None or not np.isfinite(H).all():
    raise RuntimeError("Omografia non stimabile")

inliers = mask.ravel().astype(bool)
if inliers.sum() < 4:
    raise RuntimeError("Consenso insufficiente")

# H porta sorgente -> destinazione; niente WARP_INVERSE_MAP.
# output_size = (larghezza, altezza), scelto per il canvas.
# Visualizzare maschera, residui e supporto nelle due immagini.
warped = cv2.warpPerspective(image_a, H, output_size)
