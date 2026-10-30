# Fotografia sottomarina — sezione 2 del notebook NumPy

## Fonte e crediti

- File: [ambient_mussels_600.jpg](ambient_mussels_600.jpg), 600 × 450 pixel, JPEG.
- Soggetto: fondale con mitili presso il vulcano Giggenbach, vicino alla Nuova Zelanda.
- [Pagina originale NOAA](https://archive.oceanexplorer.noaa.gov/explorations/05fire/logs/april18/media/ambient_mussels.html).
- [JPEG distribuito da NOAA](https://archive.oceanexplorer.noaa.gov/explorations/05fire/logs/april18/media/ambient_mussels_600.jpg).
- Crediti: **New Zealand-American Submarine Ring of Fire 2005 Exploration, NOAA Vents Program**.
- La [scheda NOAA Light and Color in the Deep Sea](https://oceanexplorer.noaa.gov/edu/materials/light-and-color-fact-sheet.pdf), pagina 1, descrive il fondale a circa 100 m di profondità, illuminato soltanto dalla luce solare. Questi dati provengono dalla fonte, non sono ricavati dal JPEG.
- Le [specifiche NOAA per i contenuti multimediali](https://oceanexplorer.noaa.gov/about/kit-specs/) indicano il pubblico dominio come regola generale; la pagina di questa foto riporta i crediti NOAA sopra citati e nessuna riserva di copyright distinta.

## Uso nel notebook

Il JPEG è conservato senza modifiche e incluso anche nella cella dati ripiegata del [notebook corrente](../Images_as_NumPy_Arrays_CV2026_revisionato.ipynb), sezione 2B. Si può quindi spostare il solo notebook ed eseguirlo senza download. La cella dati deve essere eseguita; non richiede modifiche.

L'analisi decodifica il JPEG in RGB `uint8`, visualizza i tre canali in grigio con `vmin=0, vmax=255` e calcola le medie sull'intera immagine. Non applica guadagni, normalizzazioni indipendenti o correzioni del colore. Il processamento precedente alla pubblicazione del JPEG e la risposta della camera non sono noti: il confronto è didattico, non una misura radiometrica calibrata.
