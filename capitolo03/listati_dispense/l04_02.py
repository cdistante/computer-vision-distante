orb = cv2.ORB_create(nfeatures=800, WTA_K=2)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
if des1 is None or des2 is None or len(des2) < 2:
    raise ValueError("Descrittori insufficienti per il ratio test")
pairs = bf.knnMatch(des1, des2, k=2)
good = []
for pair in pairs:
    if len(pair) != 2:
        continue
    m, n = pair
    if n.distance > 0 and m.distance < 0.65 * n.distance:
        good.append(m)
