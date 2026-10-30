R = cv2.cornerHarris(np.float32(gray), blockSize=3,
                     ksize=3, k=0.04)
peak = float(R.max())
mask = (R > 0.01 * peak) if peak > 0 else np.zeros_like(R, bool)
# candidati, non ancora punti sparsi

corners = cv2.goodFeaturesToTrack(
    gray, maxCorners=220, qualityLevel=0.01,
    minDistance=8, blockSize=5,
    useHarrisDetector=False)
