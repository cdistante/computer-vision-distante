# Dalle feature all’apprendimento

Laboratorio del capitolo 7 di **Computer Vision**, di **Arcangelo Distante e Cosimo Distante**.

[Apri il notebook in Google Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/main/capitolo07/07_feature_apprendimento.ipynb)

Il notebook contiene tre esperimenti con dati sintetici definiti nelle celle:

1. Scala delle feature, fit sul training e nearest neighbor.
2. Softmax stabile, temperatura, cross-entropy e hinge.
3. Matrice di confusione, metriche per classe e aggregazioni.

È sufficiente una CPU e l’unica dipendenza esterna è NumPy. Non servono immagini, checkpoint o dati personali. Eseguire le celle nell’ordine proposto; dopo una ripartenza rieseguire dall’inizio.

Gli output vengono salvati nella cartella `output_capitolo07` del runtime. In Colab scaricare i file JSON e CSV e conservare una copia del notebook con gli output prima di chiudere la sessione.

La prova di riferimento è locale CPU, con Python 3.12.14 e NumPy 2.3.5, eseguita il 30 ottobre 2026. I [file di riferimento](risultati_riferimento/) permettono di confrontare i calcoli. Non sono benchmark su immagini reali; nessun tempo locale è attribuito a Colab o a una GPU.
