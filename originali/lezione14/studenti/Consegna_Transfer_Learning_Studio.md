# Model comparison card · Lezione 14

**Computer Vision · elaborato individuale · 29 ottobre.**


Sono richieste **cinque attività guidate**: i tre TODO di codice **C11, C19, C29** e i richiami testuali **P01 in C13** (forme e parametri della testa LP), **P02 in C17** (residuo e validità della cache). Restano **39 celle, di cui 21 di codice**; P01/P02 non richiedono nuovo codice o training. Il carico stimato è **4–6 ore complessive**, comprendendo circa **20** per i richiami: è una stima organizzativa, non una misura dei tempi di esecuzione.

Le risposte **N01–N06 sono brevi, 3–5 righe o una piccola tabella**. N06 richiede di leggere la diagnosi C34 già fornita, senza sviluppare un'altra challenge. Scrivere **F01–F05 direttamente in C36, entro 500 parole complessive**, richiamando risposte e output senza ricopiarli; **non serve una relazione separata**. D01–D08, discussione conclusiva, repliche e nuovi esperimenti sono autostudio facoltativo e non valutato. Eseguire solo LP20/FT3 epoche per ciascuno dei due modelli, come già previsto, senza aggiungere corse o consultare il test.

## Percorso richiesto

1. Inserire nome e cognome in C01 e preparare il pacchetto completo. CPU, dati, pesi e supporto sono già quelli del laboratorio; seguire il [README](../README.md) e l'[avvio Colab](../Avvio_Colab.md).
2. Completare C11 (parametri trainabili/modalità), C19 (gruppi Adam) e C29 (delta in punti percentuali). Eseguire il percorso in ordine: LP di 20 epoche e FT parziale di 3 per ciascun modello, senza corse supplementari.
3. Rispondere a P01/P02 e N01–N06 nei blocchi pertinenti. Non occorre riscrivere formule, tabelle o output già presenti: indicare la cella che sostiene la risposta.
4. Completare F01–F05 in C36 entro 500 parole complessive e salvare notebook e cartella OUT. La sintesi richiama gli esercizi già svolti, senza una seconda versione delle risposte.

## Due richiami alla lezione 13

**P01 — Testa LP (C13, circa 10).** Con B immagini, 10 classi e feature di dimensione D, scrivere le forme H/W/b/S e contare pesi e bias per D=512 (ResNet18) e D=192 (DeiT Tiny). Spiegare perché il token CLS non aggiunge una classe e distinguere la testa finale dalle teste di attenzione. Usare i dati già stampati da C09, senza un altro forward.

**P02 — Residuo e cache (C17, circa 10).** Nel caso analitico senza ReLU x=(1, 2), h=x+αx, calcolare h con α=−0.2 e poi α=−0.1. La vecchia cache rappresenta il nuovo stato? Classificare i cambiamenti a sola testa finale, pesi del suffisso, preprocessing e buffer BN pertinenti all'estrattore. Una modifica può invalidare la garanzia della cache senza cambiare necessariamente ogni valore su ogni input: spiegare quando servono ricalcolo o verifica. È un calcolo analitico e una risposta, non un nuovo training.

## Sintesi F01–F05 in C36

| Sezione | Contenuto richiesto, con rimandi agli output |
|---|---|
| F01 · Protocollo | Dati/split/hash, checkpoint pretrained, versioni, preprocessing, seed, stadi, budget e selezione |
| F02 · Prestazioni | Quattro righe ResNet/DeiT × LP/FT già prodotte: epoca, accuracy corretti/N, macro-F1, CE e delta FT−LP |
| F03 · Costo | Tempi LP/FT e pipeline totale con confini, parametri totali/trainabili, hardware e latenza batch1 con variabilità |
| F04 · Stress | Alterazioni prefissate, delta in pp, immagini appaiate e validità delle etichette |
| F05 · Decisione | Requisito dichiarato, conclusione condizionata, limite e un controllo o esperimento futuro soltanto proposto |

La scheda di confronto è il report del protocollo: ft_delta conserva valori finali dei tensori adattati, da sostituire per nome, mentre il delta di accuracy è una differenza di metriche. Non presentare la memoria dei soli parametri come picco RAM né lo stress su cifre note come test di classi sconosciute. Il test precedente resta escluso dalla selezione e dalla consegna.

N06 è una diagnosi breve della dimostrazione già pronta in C34 su copia: non richiede un’altra implementazione o varianti della challenge. Le repliche menzionate in N03/F05 restano proposte di metodo, non corse da svolgere. D01–D08 e discussione conclusiva sono facoltativi, non valutati e non vanno aggiunti alla consegna.

## File da 

Consegnare il notebook individuale con nome e cognome (nome suggerito `ComputerVision_Lezione14.ipynb`) e la cartella `output_lezione14/` compressa, contenente i JSON/CSV e i checkpoint delta prodotti. Non serve un PDF separato. Conservare hash e provenienza dei pesi originali. Se si usano risultati di recupero, dichiarare quali operazioni sono personali e quali misure sono fornite nell’archivio originale; non attribuire al proprio computer i tempi importati.

Valutazione: correttezza dei cinque esercizi guidati, del protocollo e delle unità, interpretazione critica e possibilità di riproduzione. Un risultato inferiore o inconcludente resta valido se documentato; non è richiesta una soglia minima di accuracy. Le attività facoltative non aggiungono obblighi né vantaggi nella valutazione.