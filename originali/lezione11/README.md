# Computer Vision — laboratorio originale 11

Materiale associato al capitolo 8 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Come_impara_una_rete_CV2026.ipynb](Come_impara_una_rete_CV2026.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_lezione11.txt](requirements_lezione11.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione11`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r requirements_lezione11.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione11/Come_impara_una_rete_CV2026.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione11')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione11')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU è adeguata al percorso indicato. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Computer Vision — Come impara una rete

Lezione 11, 19 ottobre 2026. Il notebook contiene 29 celle, di cui 14 di codice, complete e senza TODO. **È una dimostrazione da leggere o rieseguire, senza esercizi da completare e senza consegna per la Lezione 11.** Gli spunti N01–N12 hanno spiegazioni disponibili nello stesso notebook; le risposte estese sono nella versione docente. Il notebook della Lezione 12 contiene invece il lavoro individuale da completare e  , secondo le sue istruzioni.

Gli esempi analitici usano **CPU e float64**; non eseguono un nuovo training Digits. Le celle Cxx contano anche i Markdown.

Nei materiali del corso si usa il punto decimale (0.5) e la virgola seguita da uno spazio fra componenti di vettori, per esempio (1.2, 1, −0.3).

C04 richiama autograd e il significato dei tensori; C16 introduce la memoria esponenziale di Adam e i due epsilon con ruoli distinti. C18 definisce inizializzazione, fan_in/fan_out e distribuzioni prima dei calcoli; C26 orienta al confronto BatchNorm/Identity dello Studio 12. Questi sono testi di accompagnamento della dimostrazione: non aggiungono celle o attività da completare.

## Avvio locale

Estrarre tutto `Come_impara_una_rete_studenti.zip`. La cartella contiene notebook, `dati/`, dipendenze e scheda di lettura con esempi svolti. Installare prima dell’esecuzione `requirements_lezione11.txt` in un ambiente Python compatibile e aprire il notebook dalla cartella estratta. Nella struttura del corso il notebook è invece in `studenti/`, sotto la cartella che contiene `dati/`; C03 cerca i dati anche nelle cartelle superiori.

La prova locale documentata usa Python 3.12.14, PyTorch 2.14.0 e NumPy 2.3.5, su CPU con un thread. Matplotlib e Pillow servono ai visual; JupyterLab è l’interfaccia locale. Le versioni e il dispositivo della propria esecuzione vanno registrati.

## Avvio in Google Colab

Aprire `Come_impara_una_rete_CV2026.ipynb` con **File → Carica notebook** e scegliere un runtime Python con **acceleratore hardware Nessuno**. Caricare nello stesso runtime lo ZIP intero: aprire il solo `.ipynb` non trasferisce `dati/`.

Prima di C03 eseguire una cella temporanea:

Per caricare il pacchetto completo e scegliere la cartella corrente seguire le istruzioni Colab nel README di questo laboratorio. Le celle scientifiche Cxx rimangono invariate.

La radice dello ZIP è `Come_impara_una_rete_studenti/`; la cartella corrente deve quindi essere `/content/Come_impara_una_rete_studenti`. Verificare gli import di `torch`, `numpy`, `matplotlib` e `PIL`. Il servizio può già fornire queste librerie: installare soltanto le dipendenze mancanti con `%pip install nome_pacchetto`; il pacchetto che fornisce `PIL` si chiama **Pillow**. Usare versioni compatibili con il runtime; non serve installare JupyterLab in Colab. Se un’installazione richiede il riavvio, ripetere la preparazione. C03 registra PyTorch, NumPy e Python; annotare anche le versioni di Matplotlib e Pillow quando si confrontano ambienti diversi.

Eliminare la cella temporanea dopo il setup, conservando i **29 ID C01–C29**. C03 mantiene un thread, CPU e float64. Dopo un nuovo runtime ripetere preparazione e sequenza ordinata dei blocchi: gli output visibili non ricreano variabili e gradienti. Il servizio Colab non è stato provato nella verifica documentata; i risultati di riferimento e i tempi dei materiali sono locali su CPU, senza misure GPU.

## Demo, ripartenza e consultazione

La demo alla **slide 22** esegue C03, C05, C07, C09 e C11, in ordine, poi torna alla **slide 23**. Preparare C03 prima dell’esecuzione; le altre celle sono esempi svolti da consultare facoltativamente. Non aggiungono esercizi da completare ai quattro minuti della demo.

C09 crea la MLP, calcola il backward e conserva le copie iniziali. C11 dipende da quello stato. **C13 modifica i pesi e azzera i gradienti:** prima di ripetere C11, eseguire nuovamente C09 e poi C11. Per un’esecuzione completa partire da runtime pulito ed eseguire tutti i blocchi in ordine. C29 esporta soltanto le voci effettivamente calcolate; non completa risultati mancanti.

`elapsed_cells_seconds` in C29 misura il tempo trascorso fra C03 e C29: se si eseguono le celle manualmente include lettura, attese e pause. Non è un tempo di training né una misura del solo uso della CPU.

C22 introduce il dropout come mascheramento temporaneo delle attivazioni e distingue p_drop da q_keep = 1 − p_drop prima dell’esempio. C23, già completa, mostra poi lo scaling 1/q_keep in training, l’identità in eval e il limite della media dopo una non linearità. Le slide di raccordo sono S47–S49; non sono richiesti nuovi esercizi.

## Conservare una copia personale

Non è richiesto inviare il notebook della Lezione 11. Per conservarne una copia personale si possono salvare notebook e output. In Colab usare **File → Scarica → Scarica .ipynb**. Dopo C29 scaricare anche `output_lezione11/`, che contiene il registro JSON e, se C25 è stata eseguita, la figura delle trasformazioni. Una cella temporanea può preparare il download:

```python
import shutil
from google.colab import files

shutil.make_archive('/content/output_lezione11', 'zip', root_dir=str(out))
files.download('/content/output_lezione11.zip')
```

Eliminare anche questa cella dopo il download per conservare i 29 ID. Salvare il notebook non conserva automaticamente i file del runtime. [Riferimento operativo: FAQ ufficiale Colab](https://research.google.com/colaboratory/faq.html).
