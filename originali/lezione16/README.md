# Computer Vision — laboratorio originale 16

Materiale associato al capitolo 11 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Detection_Studio_CV2026_studenti.ipynb](studenti/Detection_Studio_CV2026_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_locale_riferimento.txt](requirements_locale_riferimento.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione16/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements_locale_riferimento.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione16/studenti/Detection_Studio_CV2026_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione16')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione16/studenti')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU rimane disponibile; una GPU compatibile può essere scelta secondo le istruzioni del notebook. I suoi tempi vanno misurati nella propria sessione. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Acquisizione di dati e modelli

Dataset e pesi dei modelli non sono inclusi. Seguire i downloader originali e le condizioni in [dati/README.md](dati/README.md) e [pesi/README.md](pesi/README.md). Le eventuali modalità con accesso regolato non vengono sostituite con pesi casuali o con modelli diversi.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le risorse nelle cartelle di riferimento conservano risultati già presenti nel pacchetto studenti originale, non nuove misure della propria corsa. Distinguere sempre lettura dei risultati forniti, riesecuzione e nuovi esperimenti.

**Limite del recupero:** pesi, teste e checkpoint addestrati non sono distribuiti. JSON/CSV, figure e gli eventuali HTML di riferimento restano consultabili; le celle di ripristino che richiedono tali artefatti necessitano dell’archivio originale e non costituiscono qui una corsa autonoma. Il notebook principale conserva il percorso per la propria esecuzione.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Detection Studio · notebook

I richiami iniziali collegano i componenti alla lezione 15 e definiscono holdout, tipi/forme dei tensori, hash/dHash e distanza di Hamming prima dei controlli. Predictor, parametri/buffer, modalità, invarianti e statistiche dei tempi precedono le misure. Restano 58 slide, 180 minuti, 40 celle e gli stessi cinque completamenti/richiami individuali, con nome e cognome e consegna .

Computer Vision · Lezione 16. **Colab:** aprire studenti/Detection_Studio_CV2026_studenti.ipynb, eseguire C03 e caricare Pacchetto_studenti_Lezione_16.zip nella stessa VM. Una GPU CUDA disponibile viene usata automaticamente, altrimenti CPU. Per forzare CPU impostare DEVICE='cpu' in C03. Il riferimento consegnato è una corsa locale CPU, non una prova fisica su Colab.

**Locale:** estrarre lo ZIP, mantenere la struttura e avviare il notebook dalla cartella studenti. Preparare prima di iniziare una coppia Torch/Torchvision compatibile. requirements_locale_riferimento.txt documenta l’ambiente CPU realmente usato. In Colab conservare la coppia preinstallata: C03 installa soltanto pycocotools se manca, senza aggiornare Torch/Torchvision.

C05 scarica il dataset ufficiale e i pesi COCO la prima volta (circa 51 MiB + 74 MiB), poi riusa la cache. Lo ZIP non li redistribuisce. Preparare rete e cache prima dell’esecuzione. Una nuova VM Colab potrebbe richiedere di nuovo il download. Il notebook ha 40 celle e 3 TODO: C07 maschera/box, C12 parametri trainabili, C19 xyxy/xywh. Cxx include celle di testo.

Tre epoche della sola testa di predizione, 144 aggiornamenti. I risultati confrontano la stessa testa iniziale e finale, non il detector COCO originale. Penn 96 training e Fudan 74 holdout, una categoria persona. L’esempio a due classi è ipotetico. I dati contengono 423 istanze da maschere della release, mentre il sito storico riporta 345.

recupero/Risultati_riferimento.html permette la lettura offline, senza kernel. recupero/Ripristino_risultati.ipynb carica i risultati CPU e verifica una nuova inferenza: richiede comunque pesi e una immagine del dataset. Non completa i TODO e non equivale a una nuova corsa di training. manifest.json registra hash e fonti.

Per la ripartenza del fit ricreare la testa con C13, con C07/C12/C19 già completate. I richiami P01/P02 restano nelle celle di testo originali e non aggiungono codice. Ripetere C23 da sola continua dai pesi correnti. C37 salva il record e le predizioni; le risposte e la sintesi C36 restano nel notebook. C39 scarica gli output in Colab. Scaricare anche l’ipynb con il menu File prima che la VM termini.


Sono richieste **cinque attività guidate**: i tre TODO **C07, C12, C19** e due richiami scritti **P01 in C18** (IoU e matching), **P02 in C26** (punto operativo e AP). Restano **40 celle, di cui 23 di codice**. P01/P02 non richiedono nuove esecuzioni, inferenze o download. Prevedere **4–6 ore complessive**, inclusi **20** per i richiami: è una stima organizzativa, non un tempo misurato. Il training resta quello già fissato di tre epoche, con il confronto iniziale/finale previsto.

Le risposte **N01–N08 sono brevi, 3–5 righe o una piccola tabella**; citare gli output senza ricopiarli. N05/N06 possono richiamare P02 senza rifare i conti. Scrivere **F01–F05 in C36 entro 500 parole complessive**, senza PDF o report separato. Per F04 usare le **tre immagini già prodotte da C31**, approfondendo la sola scena scelta in N06: nessuna nuova figura né tre analisi scritte aggiuntive. D01–D08, discussione conclusiva, repliche e challenge ulteriori sono facoltativi e non valutati; in F05 basta una proposta, senza eseguirla.

CPU resta disponibile; una GPU CUDA è opzionale e viene usata se presente. La corsa di riferimento è locale CPU: non attribuirne metriche o tempi a GPU/Colab. Dichiarare l'ambiente della propria esecuzione e le parti lette dai risultati di recupero.

## Uso nello studio

Non avviare Run All con i completamenti incompleti, né una valutazione o un fit su un modello iniziale/parziale per sostituire gli output attesi. 

## Avvio e salvataggio

Seguire [Avvio Colab](Avvio_Colab.md) per runtime, caricamento, ripartenze e download. Le prove documentate sono locali CPU; nessuna esecuzione autenticata nel servizio Colab è dichiarata.
