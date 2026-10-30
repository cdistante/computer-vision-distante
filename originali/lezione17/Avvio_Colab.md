# Avvio in Google Colab — Lezione 17

1. Aprire Google Colab e usare **File → Carica notebook** con [Dense_prediction_studenti.ipynb](studenti/Dense_prediction_studenti.ipynb). Il menu richiede il file `.ipynb` estratto dal pacchetto, non lo ZIP.
2. Scegliere runtime Python CPU, senza acceleratore. Il notebook usa NumPy e Matplotlib, normalmente disponibili in Colab. Non servono pesi, immagini esterne, Drive o GPU: i dati vengono creati nello stesso runtime. In locale installare le [dipendenze](requirements_lezione17.txt) e Jupyter.
3. Eseguire C03 prima dell’esecuzione. Cxx conta tutte le 27 celle, comprese quelle di testo; non coincide con il contatore delle esecuzioni `In[]`. Ci sono 13 celle di codice completo e nessun TODO.
4. Dopo un riavvio ripartire da C03. Ogni demo richiede solo C03, salvo C11 che richiede anche C09. I sei percorsi completi da kernel vuoto sono C03, C05, C07; C03, C09, C11; C03, C13, C15, C17; C03, C19; C03, C21; C03, C23, C25.
5. Per la consegna eseguire tutte le celle in ordine da un kernel nuovo. C27 scrive `output_lezione17/risultati_lezione17.json`; le celle grafiche scrivono cinque PNG nella stessa cartella. C27 salva soltanto le sezioni effettivamente calcolate: il messaggio «Salvato» non certifica una consegna completa.
6. Prima di chiudere scaricare JSON e figure dal pannello File e il notebook dal menu File. Salvare il solo notebook in Drive non conserva automaticamente variabili e file del runtime.

**Attenzione alle ripartenze:** C03 ricrea RESULTS e azzera il registro dei risultati, ma non elimina i PNG già presenti. Un JSON appena esportato e una vecchia figura possono quindi appartenere a percorsi diversi. Per la consegna completa usare Esegui tutto e controllare le undici sezioni: identita, softmax, metriche, imbalance, forme_unet_same, resize, trasposta, box_mask, depth, flow, aperture. Ambiente, protocollo e tempo sono metadati aggiuntivi.

**Ambiente del riferimento:** CPU locale. Le istruzioni Colab indicano la procedura di avvio e salvataggio; i tempi di riferimento non sono misure cloud o GPU.