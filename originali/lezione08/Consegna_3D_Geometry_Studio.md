# 3D Geometry Studio — Consegna di laboratorio

**Lezione 08**

| Autore della consegna | Nome | Cognome |
|---|---|---|
| Studente | … | … |

Compilare gli stessi campi all’inizio del notebook. Identificare con Nome e Cognome anche l’discussione conclusiva individuale.

## Obiettivo

Ricostruire una scena da due immagini e spiegare quali evidenze sostengono il risultato. Il laboratorio e tutti gli esercizi si svolgono individualmente, con un discussione conclusiva personale. Questa traccia accompagnerà il notebook del laboratorio.

## Il percorso

| Blocco | Cosa devi ottenere | Domanda da risolvere |
|---|---|---|
| 1 — Corrispondenze | Match, inlier e distribuzione nell'immagine | Abbiamo associazioni sufficienti e ben distribuite? |
| 2 — Rettifica | Confronto prima/dopo e residui verticali | Gli stessi punti corrispondono sulla stessa riga? |
| 3 — Profondità | Disparità, profondità e maschera valida | Dove abbiamo una misura e con quali limiti? |
| 4 — Ricostruzione sparsa | Camere e punti triangolati selezionati | Quali punti possiamo conservare e in quale scala? |
| Challenge | Una modifica e confronto con la baseline | In quale passaggio nasce l'errore? |

La coppia A è un dataset reale già rettificato, con calibrazione e disparità di riferimento. La coppia B è ottenuta da A con rotazioni virtuali note: serve soltanto all'esperimento di rettifica. Per disparità e profondità si torna alla coppia A.

## Prima di iniziare

Annota identificativo dei dati, risoluzione, immagine di riferimento, unità della baseline e impostazioni comuni. Mantieni lo stesso input e gli stessi parametri durante il confronto, salvo il fattore assegnato nella challenge.

## Che cosa consegnare

1. **Notebook eseguito**, con Nome e Cognome dello studente, le risposte ai checkpoint e ai tre esercizi **E01–E03**, impostazioni e commenti che permettano di ricostruire il procedimento.
2. **Quattro figure**, esportate o chiaramente identificate nel notebook: match/inlier; rettifica prima/dopo; disparità/profondità con maschera; camere e ricostruzione sparsa.
3. **Una tabella baseline/perturbazione**, con almeno due misure pertinenti, conteggi e copertura quando si scartano campioni.
4. **Una diagnosi individuale di 80–120 parole**, seguita dalla discussione conclusiva individuale.

Ogni figura deve avere una didascalia che spieghi che cosa mostra e una conclusione sostenuta dai dati. Per immagini confrontate usare scale colore e regioni confrontabili; per la nuvola dichiarare sistema di riferimento e scala.

## Esercizi nel notebook

Completare i tre esercizi nelle sezioni in cui sono proposti. In C06, C10 e C21 una sola risposta completa può coprire sia il checkpoint sia l’esercizio; non ripetere lo stesso testo. E02 ed E03 forniscono codice di partenza con una sola riga da completare ciascuno, da copiare in nuove celle dopo la sezione indicata. Riportare calcoli o tabella e la breve interpretazione richiesta; non occorre una relazione separata. Conservare la baseline, riutilizzata nei confronti; gli esercizi non sostituiscono la challenge assegnata.

| Esercizio | Cella | Argomento |
|---|---|---|
| E01 | C06 | Calibrazione originale, centri pixel e invarianza della profondità |
| E02 | C10 | Soglie F 0.5/2 px rispetto alla baseline 1; codice guidato e conclusione CP1 |
| E03 | C21 | Blocchi 3/7 rispetto alla baseline 5; pixel comuni, un bordo e conclusione CP3–4 |

## Scheda della challenge

Ogni studente svolge una sola challenge.

**Prova assegnata:** A — sfocatura / B — disallineamento verticale / C — focale errata nella conversione in profondità.

| Campo | Da compilare |
|---|---|
| Unico fattore modificato e intensità | … |
| Impostazioni rimaste fisse | … |
| Previsione prima dell'esperimento | … |
| Regione e popolazione di valutazione | … |

| Misura e unità | Baseline | Dopo la modifica | Numero di campioni / copertura |
|---|---:|---:|---|
| Prima misura pertinente | … | … | … |
| Seconda misura pertinente | … | … | … |
| Errore su campioni validi comuni, se applicabile | … | … | … |

Se un metodo non produce un risultato utilizzabile, scrivi «non valutabile», riporta il numero di campioni e spiega dove si è interrotta la pipeline. Il fallimento documentato è un risultato da analizzare.

## Come scrivere la diagnosi

Collega il fattore modificato allo stadio della pipeline che ne risente. Cita due evidenze osservate, spiega un limite della tua interpretazione e proponi una sola modifica o nuova acquisizione che permetterebbe di verificarla. Evita conclusioni basate soltanto sull'aspetto della nuvola o sul numero di punti.

## Discussione conclusiva individuale

Compilare un solo discussione conclusiva individuale, lo stesso proposto in C30 del notebook. Rispondi in cinque righe:

1. Un’evidenza che distingue coerenza da accuratezza.
2. La provenienza della scala metrica.
3. Il significato di un pixel invalido.
4. La causa dominante osservata nella challenge.
5. Un dato da migliorare nella prossima acquisizione.

## Riferimenti e feedback

I riferimenti teorici sono le dispense 3D Geometry Studio e Due o più viste. Il dataset proviene da [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/).

Valutare riproducibilità, coerenza geometrica, qualità delle misure e solidità della diagnosi.