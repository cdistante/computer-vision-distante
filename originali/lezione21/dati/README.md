# Computer Vision — Dati della Lezione 21

`digits_cv2026.npz` è la stessa copia locale delle lezioni 9–12: 1797 immagini 8×8 con etichette 0–9, intensità uint8 0–16 e indici di training/validation/test. Qui usiamo esclusivamente training (1077) e validation (360), già consultata nel corso. Non leggiamo gli indici di test e non valutiamo il test; i dati delle identità degli scriventi non sono disponibili e la separazione per scrivente non è verificabile.

Fonte: E. Alpaydin e C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), distribuito con `sklearn.datasets.load_digits` 1.6.1. La copia scikit-learn corrisponde al sottoinsieme test del dataset UCI originale; gli split del corso sono didattici e non riproducono quel benchmark.

Modifiche della copia del corso: contenitore NumPy compresso, conversione senza perdita a uint8, indici originali preservati e aggiunta di indici per gli split. Il file non contiene le corruzioni: il notebook applica una media 3×3 con bordo esteso e una trasformazione di contrasto `0.6 x + 0.2`, dopo divisione per 16. È una diagnostica appaiata su validation già vista, non un test indipendente di dominio esterno. Le etichette restano assegnate ai campioni; la trasformazione può renderli ambigui.

SHA-256: `972401a5680cf0a1c5ef60295fb915f4efbd51d7a39b35c0cfa39d73cc33ca1e`.