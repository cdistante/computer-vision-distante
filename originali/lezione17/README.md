# Computer Vision — laboratorio originale 17

Materiale associato al capitolo 12 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Dense_prediction_studenti.ipynb](studenti/Dense_prediction_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione17.txt](requirements_lezione17.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione17/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements_lezione17.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione17/studenti/Dense_prediction_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione17')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione17/studenti')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Notebook — Dense prediction

[Notebook studenti](studenti/Dense_prediction_studenti.ipynb) · [Avvio Colab](Avvio_Colab.md) · [Dipendenze locali](requirements_lezione17.txt).

27 celle, 13 di codice. Tutto il codice è completo; le domande concettuali richiedono risposte personali. Gli ID Cxx contano tutte le celle. Esempi sintetici deterministici: nessun training, inferenza, peso o immagine esterna.

Il docente trova soluzioni argomentate e output nella sottocartella riservata docente. La distribuzione studenti include solo il notebook pulito, istruzioni, dispensa e consegna. In Colab basta caricare il singolo notebook e usare CPU. In locale occorrono Python, Jupyter e le dipendenze indicate.

C03 azzera il registro RESULTS ma conserva i PNG sul disco. C27 esporta anche percorsi parziali: controllare le undici sezioni elencate in Avvio_Colab.md. I blocchi indicano i prerequisiti e le ripartenze.

Le definizioni di encoder, decoder, upsampling e target precedono le relative formule. Il percorso distingue metriche hard e loss soft, poi profondità e flusso; lo schema di campionamento chiarisce verso del flow e griglia del warp.
