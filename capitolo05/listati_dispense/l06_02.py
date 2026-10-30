H, inlier_mask = cv2.findHomography(
    pts_src.astype(np.float32),
    pts_dst.astype(np.float32),
    method=cv2.RANSAC,
    ransacReprojThreshold=3.0,  # pixel
    maxIters=5000,
    confidence=0.99,
)

if H is None or inlier_mask is None:
    raise RuntimeError("Omografia non stimabile")

# dsize e origine vanno progettati trasformando il bordo.
warped = cv2.warpPerspective(
    image, H_canvas, dsize=(width, height),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
)
