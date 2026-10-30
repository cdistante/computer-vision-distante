# Recupero L 16

HTML e JSON leggibili senza kernel. Predictor iniziale/finale e predizioni dalla corsa CPU. Il notebook di recupero esegue solo una nuova inferenza, richiedendo dataset e pesi ufficiali. Non contiene le soluzioni dei tre TODO. I tempi del JSON rimangono riferiti alla CPU della corsa originale.

Il recupero non sostituisce i completamenti C07/C12/C19 e le risposte P01/P02 e N01–N08. Inserire nome/cognome in C01 e sintetizzare F01–F05 in C36 entro 500 parole, senza PDF separato, per il 19 novembre. Dichiarare i passaggi personali e gli output importati; P01/P02 sono esercizi scritti e non richiedono la nuova inferenza di recupero. Nello studio l’HTML già eseguito è il riferimento predefinito per gli output dipendenti dai TODO.


## Risultato di predizione e limite del ripristino

`predictions.pt` conserva soltanto due gruppi, `epoch0` ed `epoch3`, ciascuno con74 immagini: per ogni immagine ci sono `boxes` float32[N,4], `labels` int64[N] e `scores` float32[N]. Non contiene parametri del modello. Il file e la sua provenienza appartengono ai risultati studenti originali e si leggono con `torch.load(..., weights_only=True)`. Le teste `predictor_epoch0.pt` e `predictor_epoch3.pt` non sono distribuite: il recupero che le richiede dipende quindi dagli artefatti dell'archivio originale. Le tabelle e le figure restano consultabili senza nuova inferenza.