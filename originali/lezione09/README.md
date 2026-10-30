# Computer Vision — laboratorio originale 09

Materiale associato al capitolo 7 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Dalle_feature_all_apprendimento_CV2026.ipynb](Dalle_feature_all_apprendimento_CV2026.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione09.txt](requirements_lezione09.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione09`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r requirements_lezione09.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione09/Dalle_feature_all_apprendimento_CV2026.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione09')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione09')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Notebook — Lezione 9, Computer Vision

Due baseline su cifre 8×8: k-NN e classificatore lineare, con split train/validation/test. Dati locali, nessuna GPU. Versione revisionata il 25 settembre 2026.

## Studenti

Usiamo **C** per il numero di classi (qui 10), **λ** per il coefficiente della penalità nell’obiettivo e **C_reg** per l’iperparametro inverso di regolarizzazione. Un C_reg piccolo regolarizza di più. La libreria scikit-learn conserva il nome dell’argomento `C`: nel notebook si scrive `LogisticRegression(C=C_reg)`.

Estrarre tutto `Dalle_feature_all_apprendimento_studenti.zip`. Aprire `Dalle_feature_all_apprendimento_CV2026.ipynb` in Jupyter con Python 3.12 e le dipendenze indicate in `requirements_lezione09.txt`. Ad esempio, nell’ambiente scelto: `python -m pip install -r requirements_lezione09.txt`. L’interfaccia Jupyter non è inclusa nelle dipendenze di calcolo; usare un’installazione esistente. L’installazione iniziale può richiedere rete, l’esecuzione usa soltanto dati inclusi.

Da kernel vuoto eseguire tutto dall’alto in basso (“Restart Kernel and Run All”). Compilare le sette risposte e seguire `CONSEGNA.md`. Gli output vengono salvati in `risultati_lezione09` accanto al notebook. Non ampliare la griglia per migliorare il test.

Le etichette C01–C27 contano anche le celle di testo; sono uguali nelle versioni studente e docente. Il contatore In [ ] dipende dalle esecuzioni e non identifica queste celle.

| Celle | Contenuto |
|---|---|
| C03, C05 | Ambiente, dati e split; mosaico training |
| C08, C09 | Fit delle pipeline, griglie e curve di validation |
| C12 | Decisione congelata prima del test |
| C15 | Score e pesi lineari stimati sul training |
| C17 | Due esempi costruiti di softmax e cross-entropy |
| C20, C21 | Valutazione test e conteggi della classe 8 |
| C24 | Nove errori ed esportazione di CSV, JSON e figure |

**C24 dipende anche da C15 e C17.** Le sue esportazioni includono `toy_p` e `toy_losses`: eseguire soltanto C20 non prepara l’intero stato necessario. Tutte le dieci celle di codice devono essere eseguite nell’ordine indicato.


## Verifica e limiti

Versioni effettivamente provate: Python 3.12.14, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.6.1 e Matplotlib 3.11.2. Il notebook docente e quello estratto dallo ZIP studente sono stati eseguiti con dieci celle di codice e risultati uguali.

Scaler e modelli sono stimati sul training; validation sceglie configurazioni e modello principale; il test è consultato dopo il congelamento. Nessun refit train+validation. Il lineare ottiene un errore in meno sul test, ma il principale rimane il k-NN selezionato in validation. I dati non hanno ID dello scrivente: la prova non dimostra generalizzazione a nuovi autori, fotografie o oggetti naturali. Provenienza e licenza sono in `dati/README.md`.
