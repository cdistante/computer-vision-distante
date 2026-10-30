def homography_dlt(source, target):
    source, target = checked_points(source), checked_points(target)
    if source.shape != target.shape:
        raise ValueError("Coppie non corrispondenti")
    p, T = normalize_points(source)
    q, Tp = normalize_points(target)
    check_minimal(p)
    check_minimal(q)
    rows = []
    for (x, y), (u, v) in zip(p, q):
        rows.append([-x,-y,-1, 0,0,0, u*x,u*y,u])
        rows.append([0,0,0, -x,-y,-1, v*x,v*y,v])
    A = np.asarray(rows)
    # Completa in 8x9 per NON perdere la nona direzione.
    _, sv, Vt = np.linalg.svd(A, full_matrices=(len(p) == 4))
    tol = max(A.shape)*np.finfo(float).eps*sv[0]
    if np.count_nonzero(sv > tol) < 8:
        raise ValueError("Sistema DLT di rango insufficiente")
    Hn = Vt[-1].reshape(3, 3)  # accatastamento per righe
    hs = np.linalg.svd(Hn, compute_uv=False)
    if hs[-1] <= 1e-12*hs[0]:
        raise ValueError("Omografia normalizzata quasi singolare")
    H = np.linalg.solve(Tp, Hn @ T)
    if not np.isfinite(H).all():
        raise ValueError("Denormalizzazione non finita")
    return H / np.linalg.norm(H)
