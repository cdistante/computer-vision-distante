# Computer Vision

**Arcangelo Distante e Cosimo Distante**  
**Geometria, apprendimento e modelli fondazionali**

Questo repository raccoglie i notebook che accompagnano il libro. Ogni laboratorio è collegato a un capitolo e permette di eseguire i calcoli, osservare le figure e interpretare i risultati con ipotesi esplicite.

La versione **0.1.0** comprende il laboratorio del **capitolo 7, Dalle feature all’apprendimento**. Le risorse dei capitoli successivi saranno aggiunte nelle rispettive cartelle.

[Repository del libro](https://github.com/cdistante/computer-vision-distante)

## Materiali disponibili

| Capitolo | Risorse | Ambiente |
|---|---|---|
| 7 Dalle feature all’apprendimento | [Laboratorio del capitolo 7](capitolo07/07_feature_apprendimento.ipynb) | Python e NumPy; CPU sufficiente |

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/main/capitolo07/07_feature_apprendimento.ipynb)

I dati del laboratorio pilota sono sintetici e vengono costruiti dal notebook. Gli esempi servono a studiare rappresentazioni, classificazione, loss e valutazione; i loro risultati non costituiscono un benchmark su immagini reali. Non sono richiesti dataset esterni, checkpoint o una GPU.

## Avvio in tre passi

1. Aprire il notebook con il pulsante **Apri in Colab**, oppure scaricarlo e aprirlo con Jupyter nell’ambiente locale.
2. In Colab scegliere un runtime Python con **CPU**. Per un’installazione locale, preparare NumPy con `python -m pip install -r requirements.txt`, nello stesso ambiente Python usato da Jupyter.
3. Eseguire le celle dall’inizio nell’ordine proposto. Prima di chiudere Colab salvare una copia del notebook con gli output e scaricare gli eventuali file prodotti.

I dati sintetici sono definiti nelle celle: non occorre caricare altri materiali nello stesso runtime. Dopo un riavvio, rieseguire le celle iniziali prima dei calcoli successivi. Gli output visibili di una sessione precedente non ripristinano le variabili.

## Risultati e riproducibilità

Il notebook registra l’ambiente e distingue i calcoli analitici dai risultati dell’esperimento sintetico. Per confrontare un’esecuzione con quella di riferimento, conservare versione del codice, parametri, seme, versioni delle librerie e output. Piccole differenze di arrotondamento vanno interpretate rispetto alla precisione numerica usata.

Il laboratorio del capitolo 7 è stato eseguito il 30 ottobre 2026 in un kernel Python locale nuovo, su CPU, con Python **3.12.14** e NumPy **2.3.5**: tutte le sette celle di codice sono terminate senza errori. I [risultati di riferimento](capitolo07/risultati_riferimento/) comprendono JSON e CSV; questa esecuzione non è una prova effettuata nel servizio Colab.

La modalità CPU è sufficiente al laboratorio. L’apertura in Colab consente di ripetere la prova in un ambiente distinto: i risultati locali non devono essere attribuiti a una GPU o al servizio Google Colab.

## Versioni e collegamento al libro

La numerazione delle cartelle segue i capitoli del libro. La versione associata a un’edizione conserva notebook e risultati coerenti con il testo; gli aggiornamenti successivi sono descritti nel [registro delle modifiche](CHANGELOG.md).

Nell’edizione Kindle i collegamenti alle risorse sono cliccabili. Nel cartaceo il QR è accompagnato da un indirizzo leggibile e conduce alla pagina delle risorse del libro.

## Citazione

Gli autori e la versione sono indicati in [CITATION.cff](CITATION.cff). Per citare un risultato specifico, indicare anche il capitolo e la versione dei materiali utilizzata.

## Diritti e materiali riservati

© 2026 Arcangelo Distante e Cosimo Distante. La licenza di distribuzione e riuso sarà indicata dagli autori. In questa versione non è attribuita una licenza aperta.

Le soluzioni riservate al docente sono conservate separatamente dai materiali pubblici.
