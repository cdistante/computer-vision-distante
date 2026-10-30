# Computer Vision · Lezione 15 in Colab

Questa è una dimostrazione completa, senza esercizi da completare o file da inviare per la Lezione 15. I salvataggi servono soltanto a conservare una copia personale.

1. Aprire Google Colab e usare **File → Carica notebook** con `studenti/Object_detection_studenti.ipynb`. Scegliere runtime Python CPU, senza acceleratore.
2. Eseguire C03 prima dell’esecuzione. Il notebook è autonomo: NumPy e IPython sono le sole dipendenze, normalmente già presenti in Colab. Non servono ZIP, dati esterni, pesi, Drive o installazioni.
3. Seguire i gateway delle figure. Cxx conta tutte le 27 celle, comprese quelle di testo; `In[]` conta le esecuzioni e non identifica una cella. Non aggiungere celle in mezzo.
4. Per una copia personale completa usare un kernel nuovo ed eseguire tutto. Non ci sono TODO; le letture N01–N12 spiegano già gli output. C27 salva `risultati_lezione15.json`, con le sole sezioni effettivamente calcolate.
5. Prima di chiudere scaricare il notebook dal menu File e il JSON dal pannello File. Salvare il notebook in Drive non conserva automaticamente i file della VM e le variabili Python.

Dopo un riavvio gli output visibili sono soltanto un documento: rieseguire le dipendenze. Coordinate: C03, C05; IoU/regressione: C03, C05, C07, C09; NMS: C03, C07, C13, C15; AP: C03, C07, C13, C17, C19, C21, C23. C03 azzera RESULTS: se la si ripete a metà, il JSON diventa parziale finché non si rieseguono i calcoli.

**Ambiente del riferimento:** CPU locale. I tempi non sono misure del servizio Colab. Riferimenti per ambiente e persistenza: [FAQ ufficiali](https://research.google.com/colaboratory/faq.html) e [gestione file](https://colab.research.google.com/notebooks/io.ipynb).