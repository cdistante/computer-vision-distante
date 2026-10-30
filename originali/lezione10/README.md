# Computer Vision — laboratorio originale 10

Materiale associato al capitolo 7 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [PyTorch_Studio_CV2026.ipynb](PyTorch_Studio_CV2026.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione10.txt](requirements_lezione10.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione10`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r requirements_lezione10.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione10/PyTorch_Studio_CV2026.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione10')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione10')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Notebook · PyTorch Studio

Computer Vision, Lezione 10, 15 ottobre 2026. Il notebook usa dati locali e CPU. Scaricare lo ZIP, estrarlo interamente e aprire `PyTorch_Studio_CV2026.ipynb` dalla cartella estratta. Nel progetto del corso il file si trova invece in `studenti/`.

## Simboli e lessico prima di iniziare

Manteniamo la notazione della lezione 9: $s$ è il vettore degli **score prima della softmax**, detti anche *logits*, e $S$ raccoglie gli score del batch; `logits` è il nome della variabile nel codice. $p$ indica le probabilità e $C=10$ il numero delle classi. La standardizzazione o *z-score* produce $\tilde{x}_j=(x_j-\mu_j)/a_j$: la scala $a_j$ usa la deviazione standard del training, sostituita con 1 per una coordinata costante. Nel codice `mean` e `scale` conservano queste statistiche. La scala $a_j$ non è lo score $s_j$.

Un **batch** contiene $B$ campioni elaborati insieme. Uno **step** è un aggiornamento dell'ottimizzatore; qui si esegue una volta per batch di training. Una **epoca** percorre tutti i batch del training: 16 di 64 e uno di 53, cioè 17 aggiornamenti; il budget di 30 epoche ne produce 510. Validation e test percorrono batch a pesi fissi e non producono aggiornamenti.

Il **checkpoint** è una copia indipendente dello stato del modello. C24 la conserva quando macro-F1 validation migliora strettamente; a parità esatta rimane la prima epoca. Dopo tutte le 30 epoche C29 ripristina la copia e produce il **report finale**, il riepilogo di protocollo, metriche, errori e limiti sul test L9 già osservato, dichiarato replica didattica. Questa scelta entro un budget fisso non è early stopping. Il checkpoint esportato per inferenza non garantisce da solo una ripresa esatta del training. Le definizioni e il raccordo al codice sono in C01, C04, C18 e C28.

## Preparazione prima dell’esecuzione

Ambiente verificato: Python 3.12.14, PyTorch 2.14.0, NumPy 2.3.5 e Matplotlib 3.11.2. Jupyter serve per l’interfaccia. La verifica è stata svolta su macOS arm64, CPU, un thread e zero worker. Altre piattaforme possono richiedere una distribuzione PyTorch compatibile e possono differire numericamente.

In un ambiente Python 3.12 dedicato, installare `requirements_lezione10.txt`, quindi avviare Jupyter e aprire il notebook. Con strumenti a riga di comando:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_lezione10.txt
python -m jupyterlab
```

Su Windows l’attivazione dell’ambiente cambia secondo la shell. Le dipendenze sono scaricate soltanto durante la preparazione; il notebook non scarica dati o modelli. C03 cerca la cartella `dati/` nella posizione del pacchetto e verifica l’impronta del file.

## Avvio in Google Colab

Aprire `PyTorch_Studio_CV2026.ipynb` in Colab con **File → Carica notebook**. Scegliere un runtime Python con **acceleratore hardware Nessuno**: la baseline lavora su CPU anche quando una GPU è disponibile. Prima di C03, caricare nello **stesso runtime** lo ZIP completo `PyTorch_Studio_studenti.zip`, che contiene `dati/`, `supporto/` e il notebook. Il solo caricamento del `.ipynb` non trasferisce questi file.

Eseguire una cella temporanea di preparazione prima di C03:

Per caricare il pacchetto completo e scegliere la cartella corrente seguire le istruzioni Colab nel README di questo laboratorio. Le celle scientifiche Cxx rimangono invariate.

La radice interna dello ZIP è `PyTorch_Studio_studenti/`; C03 deve quindi partire da `/content/PyTorch_Studio_studenti`. Prima dell’esecuzione controllare che `import torch, numpy, matplotlib` riesca. Colab può già fornire queste librerie: se una manca, installare una distribuzione compatibile con il runtime mediante `%pip install torch numpy matplotlib`. Dopo un’installazione che richiede il riavvio, ripetere preparazione e import. Non serve installare JupyterLab nel servizio Colab. C03 registra le versioni effettive e imposta un thread; i loader usano zero worker e C19 costruisce il modello su CPU. Le versioni locali elencate sopra non sono una dichiarazione delle versioni disponibili in Colab.

Eliminare la cella temporanea dopo la preparazione: conservare le **40 celle C01–C40**, con gli stessi ID. Per una ripartenza ricreare prima cartella e dipendenze, poi eseguire l’ordine completo dei blocchi già risolti; gli output visibili non ricostruiscono le variabili del runtime. 

Prima di scollegare il runtime, salvare il notebook con output tramite **File → Scarica → Scarica .ipynb**. Dopo C36 e dopo aver scritto `output_lezione10/diagnosi.md`, usare un’altra cella temporanea per scaricare tutti i file esportati:

```python
import shutil
from google.colab import files

shutil.make_archive('/content/output_lezione10', 'zip', root_dir=str(OUT))
files.download('/content/output_lezione10.zip')
```

Eliminare anche questa cella prima della consegna. Salvare il notebook non conserva automaticamente i file temporanei del runtime. La corsa documentata nei materiali è stata verificata **localmente su CPU**; non è stata eseguita una prova nel servizio Colab. Per la propria esecuzione registrare ambiente e risultati effettivi, senza attribuirle i tempi locali preparati nell’archivio originale. [Riferimento: guida ufficiale Colab](https://research.google.com/colaboratory/faq.html).

## Sequenza di studio

| Blocco | Celle | Rientro nelle slide |
|---|---|---|
| Avvio | C03, C05 | 13 |
| Lab 1 | C07–C09, risposte C10 | 19 |
| Lab 2 | C12–C15, risposte C16 | 27 |
| Lab 3 | C19–C26, risposte C27 | 38 |
| Lab 4 | C29–C31, risposte C32 | 46 |
| Challenge | Previsione C33, prova C34, diagnosi C35 | 54 |
| Export | C36, consegna C40 | 56 |

C08, C12 e C21 sono gli unici tre blocchi da completare. Non è un notebook eseguibile integralmente prima del loro completamento: gli arresti `NotImplementedError` identificano gli esercizi. Dopo averli risolti, eseguire le celle nell’ordine numerico. Il docente dispone di recuperi negli stessi ID, senza cambiare la struttura del notebook.

C25 esegue due corse da uno stato nuovo per controllare ripetibilità nell’ambiente effettivo. C36 esporta configurazione, storia, figure, metriche, predizioni e un checkpoint per inferenza e verifica il ricaricamento. Completare le risposte nel notebook e salvare la diagnosi, massimo 150 parole, in `output_lezione10/diagnosi.md`.

Il file `supporto/studio_utils.py` contiene funzioni fornite per misure, figure, export e challenge. Dataset, DataLoader e update restano nel notebook, perché sono i tre esercizi di codice. Il checkpoint conserva i pesi selezionati, preprocessing, indici e configurazione, ma non tutti gli stati necessari a riprendere esattamente il training.

## Dati e limiti

Digits: 1797 immagini 8×8, dieci classi. Gli indici L9 restano 1077/360/360. Preprocessing adattato al solo training. Il test è già stato osservato in L9: qui è una replica didattica, non una nuova valutazione indipendente dello sviluppo. Mancano gli ID degli scriventi. Provenienza e licenza nel file `dati/README_fonte_L9.md`.
