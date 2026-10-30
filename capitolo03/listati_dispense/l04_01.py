if len(good) < 4:
    raise ValueError("Servono almeno quattro corrispondenze")
src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H, inlier_mask = cv2.findHomography(
    src, dst, cv2.RANSAC, ransacReprojThreshold=4.0)
if H is None or inlier_mask is None:
    raise ValueError("Omografia non stimabile con questi dati")
inlier_mask = inlier_mask.ravel().astype(bool)
