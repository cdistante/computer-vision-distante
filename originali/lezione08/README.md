# Computer Vision — laboratorio originale 08

Materiale associato al capitolo 6 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Geometry_3D_Studio_CV2026.ipynb](Geometry_3D_Studio_CV2026.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione08.txt](requirements_lezione08.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. Il riferimento originale usa Python 3.12. La cartella corrente del kernel deve essere `originali/lezione08`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r requirements_lezione08.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione08/Geometry_3D_Studio_CV2026.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione08')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione08')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# 3D Geometry Studio — Computer Vision

Estrarre tutta la cartella e aprire Geometry_3D_Studio_CV2026.ipynb. Usare Python 3.12 con le dipendenze in requirements_lezione08.txt. Eseguire le celle fornite in ordine: dati, calibrazione e modulo sono inclusi, senza download aggiuntivi durante gli esperimenti.

Compilare Nome e Cognome all'inizio del notebook. Completare tutti i checkpoint e i tre esercizi E01–E03 distribuiti in C06, C10 e C21. Per E02 ed E03 copiare il codice guidato in nuove celle e completare una riga per ciascun blocco. La baseline viene riusata. Una sola risposta può coprire ciascun esercizio e il checkpoint corrispondente; non sono richiesti testi duplicati, altri esperimenti o una relazione separata. Le etichette C01–C33 delle celle fornite restano i riferimenti della percorso.

La consegna comprende notebook eseguito, risposte e prove degli esercizi, quattro figure principali, confronto CSV, diagnosi di 80–120 parole e discussione conclusiva individuale con nome e cognome. Leggere Consegna_3D_Geometry_Studio.md per la traccia completa. L'discussione conclusiva è uno solo, quello di C30. Ogni studente svolge una sola challenge.

Nei materiali del corso si usa il **punto decimale** (0.5) e la **virgola seguita da uno spazio** fra componenti di vettori e argomenti, per esempio (1.2, 1, −0.3). I conteggi interi non usano separatori delle migliaia.

## Avvio in Google Colab

Aprire il notebook in Colab e caricare anche questo ZIP nel medesimo runtime. Seguire C02 per estrarre il pacchetto, caricare dati e modulo, installare le dipendenze e riavviare il runtime quando richiesto. La modalità CPU è sufficiente; non è richiesta una GPU. Gli output sono salvati in output_lezione08: scaricarli insieme al notebook eseguito prima di chiudere il runtime. Le misure di riferimento provengono da CPU locale; non sono prove svolte nel servizio Colab.

Le dispense citate nella scheda sono materiali separati del corso; non sono incluse in questo ZIP. Nessuna soluzione docente è inclusa nel pacchetto.

