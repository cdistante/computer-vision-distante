# Computer Vision — laboratorio originale 20

Materiale associato al capitolo 15 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Modern_3D_Vision_Studio_CV2026_studenti.ipynb](studenti/Modern_3D_Vision_Studio_CV2026_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione20.txt](requirements_lezione20.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione20/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements_lezione20.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione20/studenti/Modern_3D_Vision_Studio_CV2026_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione20')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione20/studenti')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le risorse nelle cartelle di riferimento conservano risultati già presenti nel pacchetto studenti originale, non nuove misure della propria corsa. Distinguere sempre lettura dei risultati forniti, riesecuzione e nuovi esperimenti.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.
