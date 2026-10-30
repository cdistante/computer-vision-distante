# Computer Vision — laboratorio originale 14

Materiale associato al capitolo 10 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Transfer_Learning_Studio_CV2026_studenti.ipynb](studenti/Transfer_Learning_Studio_CV2026_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements.txt](requirements.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione14/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione14/studenti/Transfer_Learning_Studio_CV2026_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione14')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione14/studenti')
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
file = pesi / 'resnet18-f37072fd.pth'
if not file.exists():
    urlretrieve('https://download.pytorch.org/models/resnet18-f37072fd.pth', file)
assert hashlib.sha256(file.read_bytes()).hexdigest() == 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
```

Non sostituire i checkpoint con varianti diverse. Le licenze del codice e le condizioni dei pesi restano distinte.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le risorse nelle cartelle di riferimento conservano risultati già presenti nel pacchetto studenti originale, non nuove misure della propria corsa. Distinguere sempre lettura dei risultati forniti, riesecuzione e nuovi esperimenti.

**Limite del recupero:** pesi, teste e checkpoint addestrati non sono distribuiti. JSON/CSV, figure e gli eventuali HTML di riferimento restano consultabili; le celle di ripristino che richiedono tali artefatti necessitano dell’archivio originale e non costituiscono qui una corsa autonoma. Il notebook principale conserva il percorso per la propria esecuzione.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Transfer Learning Studio · pacchetto studenti

Dispensa della Lezione 14 — 38 pagine, con architetture di ResNet18 e DeiT Tiny, dimensioni e parti congelate o trainabili.

Computer Vision, Lezione14. Estrarre lo ZIP e conservare la struttura. Aprire `studenti/Transfer_Learning_Studio_CV2026_studenti.ipynb` da Jupyter. I dati sono inclusi e i pesi ufficiali si acquisiscono separatamente: il laboratorio usa i dati locali e richiede i pesi ufficiali separati descritti nel README. Installare prima di iniziare le dipendenze in `requirements.txt` in un ambiente Python3.12 compatibile. Il riferimento usa CPU, un thread, float32. Lo ZIP non include l’interprete Python o le librerie.

Il notebook ha 39 celle, di cui 21 di codice, e tre TODO in C11, C19, C29. Completare questi blocchi e i richiami testuali P01 in C13/P02 in C17 secondo la consegna. Gli output andranno nella sottocartella `output_lezione14`. Il primo import può essere lento. Il training di riferimento LP+FT dei due modelli richiede circa9.2minuti, più setup e stress. Il portatile può richiedere tempi diversi.

`recupero/Risultati_riferimento.html` permette la lettura senza kernel. `recupero/Ripristino_risultati.ipynb` ricarica i checkpoint, verifica le predizioni pulite e registra la provenienza. Non equivale ad avere eseguito il training. Gli originali e i delta restano separati, per rendere esplicito che cosa è stato adattato.

`manifest.json` contiene gli hash dei dati, dei pesi e dei risultati di riferimento. Le fonti sono nei README dati/pesi. I risultati non sono una classifica generale delle architetture. Nessuna valutazione del test viene eseguita.


Sono richieste **cinque attività guidate**: i tre TODO di codice **C11, C19, C29** e i richiami testuali **P01 in C13** (forme e parametri della testa LP), **P02 in C17** (residuo e validità della cache). Restano **39 celle, di cui 21 di codice**; P01/P02 non richiedono nuovo codice o training. Il carico stimato è **4–6 ore complessive**, comprendendo circa **20** per i richiami: è una stima organizzativa, non una misura dei tempi di esecuzione.

Le risposte **N01–N06 sono brevi, 3–5 righe o una piccola tabella**. N06 richiede di leggere la diagnosi C34 già fornita, senza sviluppare un'altra challenge. Scrivere **F01–F05 direttamente in C36, entro 500 parole complessive**, richiamando risposte e output senza ricopiarli; **non serve una relazione separata**. D01–D08, discussione conclusiva, repliche e nuovi esperimenti sono autostudio facoltativo e non valutato. Eseguire solo LP20/FT3 epoche per ciascuno dei due modelli, come già previsto, senza aggiungere corse o consultare il test.

## Consegna e salvataggio

Salvare il notebook con identità, completamenti, risposte P01/P02 e N01–N06, sintesi C36 e output. Dopo C37 scaricare anche l’intera cartella `output_lezione14/` compressa: salvare il notebook non conserva automaticamente checkpoint, JSON e CSV del runtime. I risultati di recupero vanno attribuiti al docente e separati dalle operazioni svolte personalmente.

## Esecuzione su Google Colab

Seguire [Avvio Colab](Avvio_Colab.md): caricamento del notebook, CPU, percorsi e download dei risultati. Le celle originali Cxx restano invariate. Le versioni locali di riferimento non vanno reinstallate automaticamente nel runtime Colab.
