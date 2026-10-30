# Computer Vision — laboratorio originale 18

Materiale associato al capitolo 13 del libro di Arcangelo Distante e Cosimo Distante. Notebook principale: [Foundation_Vision_Studio_CV2026_studenti.ipynb](studenti/Foundation_Vision_Studio_CV2026_studenti.ipynb). Il laboratorio conserva gli esercizi e il codice scientifico originali. I notebook delle cartelle `capitoloNN/` sono supplementi separati.

## Avvio locale

Conservare tutta questa cartella, inclusi dati, helper e attribuzioni. Installare le dipendenze di [requirements_locale_riferimento.txt](requirements_locale_riferimento.txt) in un ambiente Python compatibile e aprire il notebook con Jupyter. La cartella corrente del kernel deve essere `originali/lezione18/studenti`. Eseguire le celle nell'ordine indicato, completando gli eventuali TODO: gli output salvati non sostituiscono le variabili del kernel.

```text
python -m pip install -r ../requirements_locale_riferimento.txt
```

## Google Colab

Aprire il [notebook in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/v1.0.0/originali/lezione18/studenti/Foundation_Vision_Studio_CV2026_studenti.ipynb). Il solo caricamento del notebook non carica i materiali nel runtime. Per preparare questa distribuzione completa, eseguire prima una cella temporanea:

```python
from pathlib import Path
import os, subprocess
materiali = Path('/content/ComputerVision_Lezione18')
if not (materiali / '.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.0',
                    'https://github.com/cdistante/computer-vision-distante.git',
                    str(materiali)], check=True)
os.chdir(materiali / 'originali/lezione18/studenti')
```

Installare le dipendenze solo se mancanti, usando il file requisiti e riavviando il runtime quando necessario. Eliminare poi la cella temporanea per conservare gli identificativi Cxx del laboratorio. La CPU rimane disponibile; una GPU compatibile può essere scelta secondo le istruzioni del notebook. I suoi tempi vanno misurati nella propria sessione. Le istruzioni Colab non attestano un’esecuzione misurata nel servizio: i riferimenti disponibili documentano il rispettivo ambiente originale, senza trasferire tempi CPU locali a GPU o cloud.

## Acquisizione di dati e modelli

Dataset e pesi dei modelli non sono inclusi. Seguire i downloader originali e le condizioni in [dati/README.md](dati/README.md) e [pesi/README.md](pesi/README.md). Le eventuali modalità con accesso regolato non vengono sostituite con pesi casuali o con modelli diversi.

## Salvataggio e riferimenti

Salvare una copia del notebook con completamenti, ipotesi e risposte, insieme a JSON/CSV, figure e cartelle di output indicati nelle celle. In Colab scaricare i file prima della chiusura della VM: salvare il notebook non conserva automaticamente il filesystem del runtime. Registrare versioni, dispositivo, seed e le prove effettivamente eseguite.

Le risorse nelle cartelle di riferimento conservano risultati già presenti nel pacchetto studenti originale, non nuove misure della propria corsa. Distinguere sempre lettura dei risultati forniti, riesecuzione e nuovi esperimenti.

Il riferimento originale dichiara DINOv3 non eseguito. La presenza delle istruzioni per un modello non costituisce evidenza della sua esecuzione.

Le attribuzioni, le fonti dei dati e le licenze nei README delle sottocartelle devono accompagnare ogni uso e ridistribuzione consentita.

## Documentazione del laboratorio originale

# Notebook · Foundation Vision Studio

Computer Vision · Lezione 18. [Avvio Colab](Avvio_Colab.md), [notebook studenti](studenti/Foundation_Vision_Studio_CV2026_studenti.ipynb), [recupero leggero](recupero/Ripristino_risultati.ipynb), [HTML offline](recupero/Risultati_riferimento.html).

45 celle totali, contando testo e codice. TODO C09 (coseno), C23 (IoU/Dice), C35 (unione maschere). Le soluzioni restano nella cartella docente, esclusa dallo ZIP studenti. C03 carica il pacchetto nello stesso runtime nella cartella Colab dedicata e controlla il manifest della lezione 18 prima di preparare il supporto. C05 scarica i dati/pesi ufficiali e verifica gli hash dei pesi. Dataset e checkpoint originali non sono inclusi nella distribuzione.

Il protocollo usa sei immagini, SAM 2.1 Hiera tiny FP32 e LRASPP MobileNetV3, senza training locale. Il riferimento è realmente eseguito su CPU locale con un thread. CUDA può essere usata quando disponibile, ma tempi e differenze numeriche vanno misurati nella propria corsa. Le prove non equivalgono a un'esecuzione autenticata su Google Colab.

DINOv3 richiede un checkpoint autorizzato personale ViT-S/16. C12 configura `DINO_CHECKPOINT` e `DINO_REPO`; C13–C14 estraggono feature e una mappa di similarità. Se i file mancano il ramo dichiara la mancata esecuzione. C10/C16 usano vettori didattici, non output DINOv3. Non vengono scaricati pesi da copie non autorizzate e non sono inclusi token nel pacchetto.

C40 controlla che tutte le sei immagini abbiano risultati SAM e LRASPP completi, poi salva JSON e maschere. C12 è necessaria anche senza DINOv3. C42 rifiuta output assenti/incompleti e crea/scarica uno ZIP con JSON e sei NPZ; include le feature DINOv3 solo quando il ramo è stato realmente eseguito. Dopo nuove misure rieseguire C40 prima di C42. Salvare separatamente l'ipynb dal menu File. C44 ricalcola una metrica dai file. Il recupero pubblico legge e verifica output salvati senza modello e senza fornire le implementazioni dei TODO. Per nuova inferenza seguire le istruzioni del notebook.

La lettura introduce feature, prompt e supervisione prima dei modelli. Griglia delle patch e norma L2/coseno precedono Gram anchoring; support e query precedono i prototipi. Le sei immagini, i tre completamenti di codice e i tempi della percorso rimangono quelli del laboratorio.
