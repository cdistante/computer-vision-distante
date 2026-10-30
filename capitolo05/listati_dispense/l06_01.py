rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None
)

ok, rvec, tvec, inliers = cv2.solvePnPRansac(
    object_pts, image_pts, K, dist,
    reprojectionError=3.0,  # pixel
    confidence=0.99,
)

if not ok:
    raise RuntimeError("Posa non stimabile")

# projectPoints consente di ricalcolare e visualizzare i residui.
pred, _ = cv2.projectPoints(object_pts, rvec, tvec, K, dist)
