# Computer Vision — laboratorio originale 21

Materiale associato al capitolo 16 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Trustworthy_Computer_Vision_CV2026_demo.ipynb](studenti/Trustworthy_Computer_Vision_CV2026_demo.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements.txt](requirements.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione21/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione21/studenti/Trustworthy_Computer_Vision_CV2026_demo.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione21')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione21/studenti')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le risorse nelle cartelle di riferimento conservano risultati già presenti nel pacchetto studenti originale, non nuove misure della propria corsa. Distinguere sempre lettura dei risultati forniti, riesecuzione e nuovi esperimenti.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Computer Vision — Notebook della Lezione 21

Notebook **dimostrativo completo** su CPU: [versione studenti](studenti/Trustworthy_Computer_Vision_CV2026_demo.ipynb), versione con output, HTML consultabile. Le domande N01–N10 sostengono la discussione; non è un nuovo homework.

Eseguire in ordine da C03; i blocchi richiedono lo stato precedente e dopo un riavvio occorre ripartire dalla preparazione.

In Colab caricare notebook e ZIP nello stesso runtime, CPU; C03 contiene l’opzione di caricamento. Sono richiesti NumPy, Matplotlib e IPython; C03 installa solo i mancanti in Colab. Locale: scegliere un ambiente con queste librerie e il supporto Jupyter del proprio editor. Nessun download di dati o pesi. La cartella `dati/` deve essere accanto a `studenti/`.

Gli output salvati provengono da una corsa locale CPU, distinta dall’esecuzione effettiva nel servizio Colab. I confronti Digits sono diagnostiche appaiate su validation già vista; gli esempi di gruppi, calibrazione, astensione e Pareto sono analitici assegnati. Il test non è valutato. C31 salva risultati, protocollo, CSV e figure in `output_lezione21.zip`; il disco Colab è temporaneo.

Le domande N01–N10 guidano calcoli, interpretazioni e limiti. [Schede clinic](consegna/Clinic_progetti_Lezione_21.md).
