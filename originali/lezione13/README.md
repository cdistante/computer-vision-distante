# Computer Vision — laboratorio originale 13

Materiale associato al capitolo 10 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Dalle_CNN_ai_transformer_studenti.ipynb](Dalle_CNN_ai_transformer_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione13.txt](requirements_lezione13.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione13`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r requirements_lezione13.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione13/Dalle_CNN_ai_transformer_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione13')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione13')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Pesi ufficiali da acquisire separatamente

I pesi non sono distribuiti nel repository. Conservare le condizioni e le attribuzioni in [pesi/README.md](pesi/README.md); il codice scientifico originale verifica gli hash. Dalla cartella principale di questo laboratorio, scaricare soltanto i file ufficiali richiesti:

```python
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
pesi = Path("pesi"); pesi.mkdir(exist_ok=True)
file = pesi / 'deit_tiny_patch16_224-a1311bcf.pth'
if not file.exists():
    urlretrieve('https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth', file)
assert hashlib.sha256(file.read_bytes()).hexdigest() == 'a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97'
```

Non sostituire i checkpoint con varianti diverse. Le licenze del codice e le condizioni dei pesi restano distinte.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Notebook · Lezione 13

Il notebook `Dalle_CNN_ai_transformer_studenti.ipynb` è una dimostrazione completa: **35 celle totali, 17 di codice, nessun completamento o invio richiesto**. Le posizioni C01–C27 e il codice dei piccoli esempi analitici restano invariati. C28–C35 mostrano una fotografia, il calcolo della CLS-verso-patch con DeiT Tiny e i tre pannelli immagine/mappa/sovrapposizione. Le letture guidate N01–N15 hanno spiegazioni già svolte. La sintesi F01–F05 in C26 è già compilata.

Estrarre tutto `Dalle_CNN_ai_transformer_studenti.zip` e aprire il notebook dalla cartella estratta. Conservare le sottocartelle `supporto/`, `input/` e `pesi/`: contengono il modulo PyTorch, la foto Chelsea (Stefan van der Walt, CC0) e le istruzioni per acquisire il checkpoint ufficiale DeiT Tiny non distillato. Dati e helper sono inclusi; il checkpoint ufficiale si acquisisce separatamente con la procedura del README. Le dipendenze sono in [requirements_lezione13.txt](requirements_lezione13.txt); non servono timm o torchvision.

## Esecuzione e lettura

Per tutti i risultati usare Restart Kernel and Run All. Cxx conta tutte le celle, comprese quelle Markdown; non coincide con `In[]`. C03 prepara gli esempi analitici su CPU in float64, con seed 20261026. La dimostrazione fotografica usa invece CPU float32 e un checkpoint preaddestrato in `eval`, senza training o accesso al test set.

- Esempi analitici: seguire C03–C27. Per le sole permutazioni da kernel nuovo: C03, C13, C15, C21, C23.
- Sola fotografia da kernel nuovo: **C29, C31, C33, C35**. C29 prepara risorse e modello, C31 esegue inferenza ed estrae i pesi, C33 mostra e salva i tre pannelli, C35 conserva le misure.
- La mappa è la media dei pesi grezzi CLS-verso-patch delle tre head dell’ultimo blocco, rinormalizzata sulle 196 patch e disposta in una griglia 14×14. Non è una probabilità di classe, una segmentazione o un attention rollout.

C27 salva `risultati_lezione13.json`; C35 aggiorna lo stesso file con la sezione `cls_patch`. C33 salva `output_cls/cls_attention_chelsea.png`; C35 salva `output_cls/cls_attention_misure.npz`. I salvataggi comprendono soltanto i blocchi effettivamente eseguiti: la ripartenza autonoma C29–C35 conserva la sola dimostrazione CLS. Questi file possono essere tenuti per consultazione personale, senza consegna.

Per seguire le dimostrazioni, leggere prima residuo e gradiente, LayerNorm, patch ed embedding, query/key/value, una riga di attenzione e multihead. Gli esempi con numeri assegnati sono distinti dall’inferenza fotografica con pesi preaddestrati.

Nei materiali del corso si usa il **punto decimale** (0.5) e la **virgola seguita da uno spazio** fra componenti dei vettori, per esempio (1.2, 1, −0.3). I conteggi interi non usano separatori delle migliaia.

## Esecuzione su Google Colab

Seguire [Avvio Colab](Avvio_Colab.md): caricare ed estrarre lo ZIP completo nel runtime CPU, poi eseguire il notebook. Il solo `.ipynb` non contiene le risorse della dimostrazione fotografica. Non reinstallare automaticamente le versioni locali nel runtime Colab. L’esecuzione autenticata in Colab non è stata effettuata; la verifica disponibile è locale su CPU.

Il lavoro individuale della Lezione 14 si introduce il 29 ottobre e  secondo le sue istruzioni. Gli esercizi D01–D08 della dispensa sono facoltativi per il ripasso; la guida docente ne conserva le soluzioni.
