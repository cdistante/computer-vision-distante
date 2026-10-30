# Immagini locali — Digits

Nei materiali del corso si usa il **punto decimale** (0.5) e la **virgola seguita da uno spazio** fra componenti di vettori e argomenti, per esempio (1.2, 1, −0.3). I conteggi interi non usano separatori delle migliaia.

`digits_cv2026.npz` contiene le 1797 immagini 8×8 di `sklearn.datasets.load_digits` 1.6.1, con etichette 0–9. Le intensità sono interi 0–16. La copia distribuita da scikit-learn corrisponde al sottoinsieme test del dataset UCI originale; il notebook crea un proprio split didattico, che non riproduce il benchmark originale.

Fonte: E. Alpaydin e C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49). [Licenza CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), come indicato nella [scheda UCI](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). [API load_digits 1.6.1](https://scikit-learn.org/1.6/modules/generated/sklearn.datasets.load_digits.html). Fonti verificate il 12/09/2026.

Modifiche per il corso: contenitore NumPy compresso; intensità ed etichette convertite senza perdita a uint8; indici originali preservati; aggiunti indici di training/validation/test. Nessuna immagine sintetica, alterazione o campione rimosso. Le 1797 immagini sono tutte distinte per contenuto.

Campi: `images` (1797, 8, 8), `target` (1797,), `train_idx` (1077,), `val_idx` (360,), `test_idx` (360,). Split stratificato con seed 20261012 per il test e 20261013 per il validation. I notebook ricostruiscono e verificano gli indici. È una partizione per immagini: le identità degli scriventi non sono disponibili nel file e non viene garantita la loro separazione.

SHA-256 del file: `972401a5680cf0a1c5ef60295fb915f4efbd51d7a39b35c0cfa39d73cc33ca1e`.