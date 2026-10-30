Xh = cv2.triangulatePoints(P0, P1, a.T, b.T).T
X = Xh[:, :3] / Xh[:, 3, None]
