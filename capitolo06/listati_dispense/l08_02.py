den = d + cal["doffs"]
z = np.full(d.shape, np.nan, dtype=np.float32)
valid = np.isfinite(d) & (den > 0)
z[valid] = (cal["K0"][0, 0] * cal["baseline_m"]
            / den[valid])
