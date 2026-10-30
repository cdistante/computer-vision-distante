# Consegna — PyTorch Studio, 15 ottobre 2026

**Computer Vision · Lezione 10.** L’attività si svolge in coppia; le cinque risposte dell’discussione conclusiva sono individuali. Completare C08, C12 e C21, eseguire il notebook e rispondere a N01–N12. C36 esporta gli output tecnici; salvare anche il notebook completato e la diagnosi di C35.

Costruire la baseline lineare assegnata, rispettando dati, split e criterio di selezione. Il test già mostrato nella lezione 9 viene riusato come verifica didattica, non come una nuova prova indipendente di generalizzazione. Non è richiesto cercare l’accuracy migliore o provare altri modelli.

Il **report finale** è il riepilogo del protocollo e dei risultati del modello ripristinato, composto dai materiali F01–F05; non è un nuovo addestramento. Il **checkpoint** è la copia indipendente dello stato scelta sulla validation e ricaricata prima della misura sul test. Il training completa il budget di 30 **epoche**, ciascuna un passaggio sui batch del training, e non usa il test per la scelta. Le definizioni operative sono in C01/C18/C28 e nel README del pacchetto.

Negli elaborati usare $s$ per gli score (*logits* nel codice), $S$ per gli score del batch e $p$ per le probabilità, come nella lezione 9. La scala di standardizzazione $a_j$ è distinta dallo score $s_j$.

## Materiali da consegnare

| ID | Elemento | Che cosa deve consentire di verificare |
|---|---|---|
| F01 | Notebook con output e risposte N01–N12 | Dataset, batch, aggiornamento e valutazione corretti; nessuna cella lasciata in errore |
| F02 | Curve di apprendimento | Loss train_eval e validation sullo stesso stato, macro-F1 validation ed epoca selezionata |
| F03 | Confusion matrix e metriche | Righe vere, colonne predette, conteggi/supporti e valori del modello ricaricato |
| F04 | Configurazione e checkpoint | Dati/split, preprocessing, seed, versioni, modello, ottimizzatore e criterio della scelta |
| F05 | Diagnosi di massimo 150 parole | Un errore o la challenge scelta, evidenza, spiegazione e limite |

F02 deve distinguere la loss online durante gli aggiornamenti dalla loss del training valutata a fine epoca. F03 deve indicare la convenzione adottata per denominatori nulli. F04 deve permettere la ricostruzione del modello e del preprocessing per la valutazione; non si richiede il salvataggio completo per riprendere l’addestramento a metà corsa.

Per F05 usare questa traccia: «Ho osservato …; il controllo che lo sostiene è …; l’ipotesi o il contratto violato è …; la correzione proposta è …; il limite che rimane è …». Non inventare risultati in caso di una corsa incompleta: indicare il punto raggiunto e il controllo che non passa.

## Challenge in coppia

Svolgere una sola prova assegnata:

- **A:** due backward senza azzeramento dei gradienti, in un esperimento controllato a pesi fissi;
- **B:** media semplice delle medie dei batch invece della media pesata per esempio;
- **C:** checkpoint conservato mediante una semplice assegnazione di state_dict, senza copia indipendente.

Annotare prima la previsione, poi il controllo osservato. Un controllo corretto non garantisce da solo una migliore accuracy. Il test non serve per scegliere la correzione.

## Discussione conclusiva individuale

- **E01:** Che cosa restituisce un Dataset e che cosa aggiunge un DataLoader?
- **E02:** Quali operazioni azzerano, accumulano e usano i gradienti?
- **E03:** Come si aggrega una loss media quando l’ultimo batch è più piccolo?
- **E04:** Come si sceglie e conserva il checkpoint finale?
- **E05:** Quale limite rimane dopo aver fissato un seed e riusato il test della lezione 9?

## Criteri di feedback

Il docente verifica coerenza del protocollo, correttezza dei conteggi, distinzione fra osservazione e ipotesi e capacità di spiegare il risultato. Per le domande aperte sono accettabili diagnosi diverse se sostenute da controlli pertinenti. La consegna è formativa e non introduce una nuova componente nel voto d’esame.