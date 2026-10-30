raw = matcher.compute(gray_left, gray_right)
d = raw.astype(np.float32) / 16.0
valid = raw > -16  # minDisparity = 0
# Gli invalidi restano espliciti.
d[~valid] = np.nan
