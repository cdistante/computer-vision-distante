# Computer Vision — laboratorio originale 15

Materiale associato al capitolo 11 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Object_detection_studenti.ipynb](studenti/Object_detection_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione15.txt](requirements_lezione15.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione15/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements_lezione15.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione15/studenti/Object_detection_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione15')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione15/studenti')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Computer Vision · Notebook della Lezione 15

Le letture iniziali definiscono box, GT e score; anchor e offset precedono la codifica, assignment precede i target, e inviluppo precede AP. Le figure di anchor/target e campione bilineare affiancano le formule. Restano 58 slide, 120 minuti e un notebook dimostrativo già svolto di 27 celle, senza consegna.

Notebook dimostrativo autonomo, senza dati esterni o TODO: 27 celle, di cui 13 di codice già complete. N01–N12 contengono letture guidate con spiegazioni disponibili; F01–F04 in C26 sono una sintesi già svolta. La versione docente conserva risposte estese private. **Non occorre completare codice, preparare una scheda o inviare file per la Lezione 15.** Gli esercizi D01–D08 della dispensa sono facoltativi per studio personale.

Per una consultazione completa si può eseguire tutto da un kernel nuovo. Il JSON esportato in C27 e il notebook sono copie personali per il ripasso. I quattro passaggi dalle slide e le loro dipendenze restano quelli della percorso. Il lavoro individuale della Lezione 16 si avvia il 12 novembre e  secondo le sue istruzioni.

## Avvio e salvataggio

Seguire [Avvio Colab](Avvio_Colab.md) per runtime, caricamento, ripartenze e download. Le prove documentate sono locali CPU; nessuna esecuzione autenticata nel servizio Colab è dichiarata.
