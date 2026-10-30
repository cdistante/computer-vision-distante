# Computer Vision — laboratorio originale 12

Materiale associato al capitolo 9 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [CNN_Training_Studio_CV2026_studenti.ipynb](studenti/CNN_Training_Studio_CV2026_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements.txt](requirements.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione12/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione12/studenti/CNN_Training_Studio_CV2026_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione12')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione12/studenti')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le risorse nelle cartelle di riferimento conservano risultati già presenti nel pacchetto studenti originale, non nuove misure della propria corsa. Distinguere sempre lettura dei risultati forniti, riesecuzione e nuovi esperimenti.

**Limite del recupero:** pesi, teste e checkpoint addestrati non sono distribuiti. JSON/CSV, figure e gli eventuali HTML di riferimento restano consultabili; le celle di ripristino che richiedono tali artefatti necessitano dell’archivio originale e non costituiscono qui una corsa autonoma. Il notebook principale conserva il percorso per la propria esecuzione.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Notebook — CNN Training Studio

Computer Vision · Lezione 12.

Aprire `studenti/CNN_Training_Studio_CV2026_studenti.ipynb` nella cartella completa. Contiene 43 celle e 20 celle di codice: completare **C10 architettura, C13 receptive field, C17 modalità** e le attività brevi **P01 in C07/C08, P02 in C16/C17, P03 in C27**. Sono cinque completamenti di codice in **C08/C10/C13/C17**: C17 contiene modalità BN e P02; P03 è testuale. Eseguire in ordine. Un semplice Run All prima del completamento non è previsto.


Il percorso richiesto comprende i tre blocchi di codice **C10, C13, C17**, le attività brevi **P01** (neurone e backward, C07/C08), **P02** (accumulo dei gradienti, C16/C17), **P03** (checkpoint, C27), i controlli e le due corse A/B già previste. Rispondere a **N01–N15 in 3–5 righe per punto**, usando output e calcoli una sola volta: kernel e BatchNorm non richiedono un secondo svolgimento nella sintesi. Scrivere **F01–F05 in C40, entro 500 parole complessive**; non occorre un PDF separato. Challenge C01–C03, repliche N16/C39 ed esercizi D01–D08 sono **facoltativi e non valutati**; E01–E05 sono autoverifica, senza risposta obbligatoria nella consegna.

Prevedere **4–6 ore complessive** nella settimana: è una stima organizzativa, non un tempo misurato di esecuzione. Le attività P01–P03 richiedono indicativamente 30 inclusi nel totale. Non sono richiesti altri training oltre alle due corse A/B e al piccolo fit di controllo già fornito; il test resta escluso.

## Avvio locale

Estrarre l’intero `CNN_Training_Studio_studenti.zip`, mantenendo `studenti/`, `dati/`, `supporto/` e `recupero/`. Prima di iniziare installare Python 3.12 e i pacchetti di `requirements.txt` in un ambiente dedicato. Configurazione collaudata: Python 3.12.14, PyTorch 2.14.0, NumPy 2.3.5, Matplotlib 3.11.2, CPU con un thread e zero worker. Le versioni alternative possono dare differenze numeriche e di tempo. Non è richiesta una GPU. I dati sono inclusi: nessun download di dati durante lo studio.

La cartella `supporto/` contiene strumenti forniti per valutazione, diagnostica, grafici ed export. `dati/` conserva la stessa copia Digits e gli stessi split della L9. `recupero/` contiene risultati reali dell’archivio originale; consultare il README interno se si usano al posto di una corsa completa. C03 cerca `dati/` nella cartella corrente e nelle sue superiori; `output_lezione12/` viene creata **nella cartella corrente del runtime**, non necessariamente accanto al file notebook.

## Avvio in Google Colab

Aprire `studenti/CNN_Training_Studio_CV2026_studenti.ipynb` con **File → Carica notebook** e scegliere un runtime Python con **acceleratore hardware Nessuno**. Caricare nello stesso runtime l’intero `CNN_Training_Studio_studenti.zip`: il solo notebook non trasferisce dati e strumenti di supporto.

Prima di C03 eseguire una cella temporanea:

Per caricare il pacchetto completo e scegliere la cartella corrente seguire le istruzioni Colab nel README di questo laboratorio. Le celle scientifiche Cxx rimangono invariate.

La radice dello ZIP è `CNN_Training_Studio_studenti/`; usare quindi `/content/CNN_Training_Studio_studenti` come cartella corrente prima di C03. Verificare gli import di `torch`, `numpy` e `matplotlib`. Se una libreria manca, installare soltanto quella con `%pip install nome_pacchetto`, in una cella temporanea; scegliere versioni compatibili con il runtime. Non occorre JupyterLab in Colab. Dopo un riavvio ripetere caricamento, estrazione e scelta della cartella, poi le celle del percorso in ordine. Eliminare le celle temporanee, conservando i **43 ID C01–C43**.

C03 configura **CPU, un thread e algoritmi deterministici**. Gli input e i modelli CNN sono float32; gli esempi analitici in C08, C15 e la prova scalare P02 in C17 usano float64. C03 registra Python, PyTorch e NumPy; annotare anche la versione di Matplotlib e il dispositivo della propria esecuzione. I tempi e i risultati di riferimento sono stati misurati localmente su CPU. **Il servizio Colab non è stato provato nella verifica documentata**: versioni diverse possono produrre scarti, da dichiarare senza cercare un seed favorevole.

## Percorso, ripartenze e recupero

I nove passaggi slide → celle → slide sono nella percorso docente; i cinque blocchi da completare nelle quattro celle C08/C10/C13/C17 e le dipendenze sono gli stessi in Colab. Dopo un riavvio eseguire le celle operative da C03 fino al punto raggiunto. Completare C08/C10/C13/C17 nelle celle originali: P01 in C08; architettura in C10; RF in C13; modalità BN e P02 in C17. C17 deve impostare esplicitamente `eval_copy.eval()`; `no_grad()` da solo non cambia la modalità BatchNorm.

C20 ricrea gli stati ufficiali A/B e le copie iniziali. Dopo averla rieseguita, rifare C25 per A e C31 per B, poi C33/C35; non mescolare nuovi modelli con risultati precedenti. C25 e C31 ricominciano rispettivamente da `initial_A` e `initial_B`. C30 contiene l’ipotesi individuale, da scrivere **prima** di C31. C39 conserva `RUN_REPLICATES=False`: le repliche sono facoltative e non valutate, anche nel lavoro in autonomia. Il test resta escluso dalla selezione e dalla consegna.

C41 richiede anche `summary` (C33), `tiny` (C22), `kernel_result` (C08), `bn_result` (C15) e `mode_result` (C17), oltre ai risultati P01/P02 calcolati in C08/C17. Un checkpoint di recupero consente il forward e la valutazione, ma non ricrea questi oggetti. Alcuni file possono essere stati scritti prima di un errore: verificare il completamento dell’export, oltre alla presenza dei file.

## Salvataggio e consegna

Conservare il notebook con nome e cognome in C01, svolgimenti P01–P03, completamenti nelle quattro celle C08/C10/C13/C17, ipotesi C30, risposte brevi N01–N15 e sintesi F01–F05 in C40 (entro 500 parole). Challenge, repliche e consolidamento non sono richiesti; C42 è autoverifica. In Colab usare **File → Scarica → Scarica .ipynb**. Dopo C41 scaricare anche tutti i file di `output_lezione12/`: history CSV, risultati JSON, checkpoint A/B, metriche e immagini. Una cella temporanea può preparare il download:

```python
import shutil
from google.colab import files

shutil.make_archive('/content/output_lezione12', 'zip', root_dir=str(OUT))
files.download('/content/output_lezione12.zip')
```

Eliminare questa cella prima della consegna. Salvare il notebook non conserva automaticamente i file del runtime. C41 non salva le risposte discorsive: mantenerle nelle celle Markdown originali. Se si sceglie una challenge facoltativa, conservare anche la diagnosi nel notebook. La sintesi F01–F05 va scritta in C40, senza relazione PDF separata; F04 può usare una normale osservazione delle curve o delle figure A/B. Quando si usano log forniti, dichiararne la fonte e distinguere i propri passaggi realmente eseguiti.

[Riferimento operativo: FAQ ufficiale Colab](https://research.google.com/colaboratory/faq.html).
