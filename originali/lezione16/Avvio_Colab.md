# Computer Vision · Lezione 16 in Colab

Il notebook è individuale: nome/cognome in C01, completamenti C07/C12/C19, richiami scritti P01/C18 e P02/C26, sintesi F01–F05 in C36 entro 500 parole. Consegna  senza relazione separata. Conservare le 40 celle e le 23 celle di codice.

## Avvio nello stesso runtime

1. Scaricare `Pacchetto_studenti_Lezione_16.zip` e aprire in Colab `studenti/Detection_Studio_CV2026_studenti.ipynb`, estratto dallo ZIP. Caricare un file `.ipynb`, non lo ZIP, nel menu File → Carica notebook.
2. Scegliere CPU oppure una GPU CUDA disponibile. C03 usa `DEVICE=os.environ.get('CV_DEVICE', 'auto')`: auto sceglie CUDA se disponibile, altrimenti CPU. Per forzare CPU sostituire **quella riga** con `DEVICE='cpu'` prima dell’esecuzione. Il riferimento dell’archivio originale è locale CPU; una GPU non è necessaria per completare questo piccolo laboratorio.
3. Eseguire **C03** e caricare lo ZIP quando richiesto. Il caricamento avviene nella VM del notebook che si sta usando. C03 estrae in `/content/ComputerVision_Lezione16`, trova supporto e si sposta nella cartella studenti. Un nuovo lancio riusa i file presenti; una nuova VM richiede nuovamente il caricamento.
4. Conservare Torch/Torchvision preinstallati. C03 verifica che l’operatore NMS funzioni e installa soltanto `pycocotools==2.0.11`, senza aggiornare le altre dipendenze, se manca. Non installare il requirements locale di riferimento sopra l’ambiente Colab. Se gli import non funzionano, ripartire da un runtime pulito e annotare l’errore.
5. **C05 prima dell’esecuzione:** con rete disponibile scarica una sola volta dataset PennFudan e pesi COCO ufficiali (circa 51 MiB e 74 MiB), poi riusa la cache. Confronta SHA256 e provenienza con il manifest. Lo ZIP studenti non redistribuisce immagini originali o pesi COCO. In una VM nuova i download vanno ripetuti.
6. Completare **C07, C12, C19** nella propria copia durante la settimana, prima di eseguire il percorso in ordine. P01/P02 richiedono solo testo nelle celle originali. non usare Run All con i TODO aperti. Sono 40 celle complessive; Cxx comprende Markdown e non è il numero `In[]`.

## Salvataggio

C13 salva la testa iniziale; C23 salva quella finale; C37 salva JSON e predizioni; **C39 crea e scarica lo ZIP degli output**. Scaricare separatamente il notebook dal menu File, conservando nome/cognome, risposte e sintesi C36: non sono esportati tutti automaticamente nei JSON. Fare il download prima della disconnessione: salvare in Drive il solo notebook non conserva le variabili Python né gli altri file della VM.

C37 richiede che siano state eseguite anche la misura C29 e la verifica di ricarica C32. Un export prematuro genera un errore, non un record completo. C39 comprime ciò che esiste in OUT: perciò un archivio può essere parziale se C37 non è riuscita. Controllare JSON, checkpoint iniziale/finale e predictions.pt.

## Ripartenza e recupero

- **Ripetere il protocollo dall’inizio:** kernel nuovo, C03, C05, TODO completati e tutte le celle in ordine. Se il kernel è ancora valido ma il training va rifatto, C13 ricrea esattamente la predictor iniziale; rieseguire C17, C19, C20, C21, C23, C24, C25 e poi analisi/export. C23 da sola continuerebbe dai pesi correnti con un nuovo SGD, non riprodurrebbe il protocollo originale.
- **Solo lettura, senza kernel:** aprire `recupero/Risultati_riferimento.html`. Metriche e tempi sono quelli CPU originali, dichiarati come importati.
- **Nuova inferenza dal riferimento:** aprire `recupero/Ripristino_risultati.ipynb` e seguire R01–R08. R02 chiede lo ZIP nella propria VM; non presumere che condivida file o variabili col notebook studenti. R03 verifica manifest, dataset e pesi; R04–R05 caricano la testa finale e verificano una sola nuova inferenza. Il recupero non completa i TODO né ricostruisce lo stato di SGD per riprendere il fit.
- Un seed uguale non garantisce identità fra versioni o dispositivi. Registrare ambiente, protocollo e tempi della propria corsa; non attribuire a CUDA i tempi CPU importati. Se R05 differisce dal riferimento, conservare la diagnosi senza sostituire i numeri per farla apparire riuscita.

I due richiami scritti non richiedono altri download o inferenze. La stima di 4–6 ore comprende completamento e interpretazione, non indica il tempo di training o una misura del servizio Colab.

**Verifica effettuata:** percorso completo, avvio, omissioni, reset e recupero eseguiti in kernel locali CPU. Upload/riuso, ZIP errato, manifest errato e chiamata d’installazione del solo pacchetto mancante simulati localmente. La simulazione non esegue pip né autentica il servizio Google. Nessuna corsa GPU o sessione Colab effettiva è dichiarata. Riferimenti: [FAQ Colab](https://research.google.com/colaboratory/faq.html), [gestione file](https://colab.research.google.com/notebooks/io.ipynb), [riproducibilità PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).