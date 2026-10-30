# Lezione 12 — CNN Training Studio

**Elaborato individuale · assegnato il 22 ottobre 2026 · consegna .**

Inserite **nome e cognome in C01** del notebook. Il lavoro va completato da una sola persona durante la settimana. Le attività presentate nello studio servono come orientamento e dimostrazione.

**Computer Vision · CNN Training Studio.** Notebook, dati, checkpoint e log reali di recupero sono nel pacchetto. Questo documento definisce le attività; le risposte restano nei materiali docente. Il [README del notebook](../README.md) spiega installazione e struttura.

L'obiettivo è costruire una piccola CNN, verificare che dati, gradienti e modalità funzionino come previsto e confrontare una baseline con una sola variante. La consegna comprende il **notebook compilato con risposte e output**, la sintesi **F01–F05 in C40 entro 500 parole complessive** e la cartella degli output. Non è richiesta una relazione PDF separata. Il feedback riguarda la qualità del metodo: non è richiesta una soglia minima di accuracy e non si introduce una nuova componente di voto.

## Organizzazione del lavoro

Prevedete **4–6 ore complessive** fra il 22 e il 29 ottobre: è una stima organizzativa, non una misura dei tempi di calcolo. Le tre attività aggiunte P01–P03 occupano circa 30 inclusi nel totale. Il percorso richiesto mantiene solo le due corse A/B e il piccolo fit di controllo già fornito; non occorrono nuove prove sul test.


| Attività | Prodotto |
| --- | --- |
| Obiettivo, protocollo, richiamo e demo | Domanda sperimentale e controlli da eseguire |
| Lab 1 — CNN, forme e receptive field | Checkpoint 1 |
| Lab 2 — BatchNorm e stato | Checkpoint 2 |
| Lab 3 — Baseline e diagnostica | Checkpoint 3 |
| Lab 4 — Ablation e interpretazione | Checkpoint 4 |
| Challenge | Approfondimento facoltativo, non valutato |
| Discussione e sintesi | F01–F05 in C40 |
| Autoverifica e raccordo | E01–E05, non da consegnare |

Il notebook fornisce loader, valutatore, grafici e salvataggio. Completate **sei esercizi guidati**: i tre blocchi di codice C10 (architettura), C13 (RF), C17 (modalità) e le attività brevi P01–P03 descritte sotto, senza riscrivere integralmente il loop della L10. Rispondete a **N01–N15 in 3–5 righe per punto**, accanto al blocco pertinente; per quantità già calcolate rimandate al vostro output invece di ripetere lo svolgimento. I calcoli analitici di kernel e BN vanno svolti una sola volta nel rispettivo blocco. I cinque completamenti di codice sono distribuiti nelle quattro celle **C08/C10/C13/C17**: C08 include P01, C17 include modalità BN e P02. P03 richiede soltanto testo. Conservate i 43 ID delle celle.

## Tre attività brevi nel notebook

- **P01 — Neurone, backward e uno step SGD (C07/C08, circa 15 minuti).** Prima del kernel, usate x=2, w=2, b=−1, target y=1 e h=ReLU(wx+b), con L=0.5·(h−y)². Calcolate forward e derivate rispetto a w, b e x; controllatele con autograd. Eseguite un solo aggiornamento simultaneo di w e b con eta=0.1 e ricalcolate la loss. Distinguete backward e aggiornamento; non applicate lo step a x.
- **P02 — Accumulo dei gradienti (C16/C17, circa 10 minuti).** Su un parametro scalare separato theta=2 usate L=0.5·theta². Confrontate due backward senza azzeramento con un nuovo backward dopo zero_grad. Ricostruite la loss prima di ogni backward. Registrate theta e il suo gradiente; non eseguite optimizer.step e non usate le reti A/B.
- **P03 — Selezione del checkpoint (C27, circa 10 minuti).** Per le epoche 1, 2, 3, 4 sono date CE validation (0.90, 0.70, 0.60, 0.50) e macro-F1 validation (0.70, 0.75, 0.75, 0.74). Con budget fisso di quattro epoche, massimo macro-F1 e prima epoca a parità, indicate quale checkpoint conservare e quante epoche eseguire. Spiegate se minima CE o test intervengono nella scelta. È una tabella sintetica, non una misura Digits; non richiede training.

P03 è l’attività richiesta sulla selezione: D05 rimane un approfondimento facoltativo, senza secondo svolgimento obbligatorio.

## Lessico operativo

Prima del blocco pertinente leggere le introduzioni del notebook: **C02/C04** per CNN, stato, forme e tipi numerici; **C09/C12** per pooling, Flatten, stride, padding, dilation e campo recettivo; **C14/C16** per statistiche BN, parametri, buffer e modalità; **C19/C23/C24** per appaiamento, sonde, norme e curve. Sono spiegazioni di supporto ai sei esercizi previsti, senza risposte aggiuntive da consegnare. NCHW indica batch, canali, altezza e larghezza; uno score s è il valore prodotto dalla testa prima della trasformazione in probabilità. Fit significa addestramento.

## Protocollo fissato prima delle corse

Usate [Digits locale](../dati/digits_cv2026.npz), senza download o nuovo split: 1077 immagini training, 360 validation e 360 test. Gli input grezzi sono immagini 8×8 con intensità 0–16 e dieci classi. Il test resta fuori dalla selezione e dalla consegna principale; è già stato discusso nelle lezioni precedenti.

Convertite in float32, dividete per 16 e stimate una **sola media e una sola deviazione standard di canale**, con varianza popolazione, su tutti i pixel delle sole immagini training, accumulando le statistiche in float64. Riusate quei due scalari per trasformare tutti gli split in float32. Il preprocessing differisce da quello per pixel della L10: oggi il confronto controllato è A/B, non il confronto numerico con la baseline L10. Se la deviazione standard risultasse zero, interrompete e controllate i dati.

Il contratto degli input è `(B, 1, 8, 8)` float32 e target `(B,)` long. CPU, batch 64, `num_workers=0`, `drop_last=False`; shuffle solo training. Nessuna augmentation, dropout, scheduler o weight decay in questa ablation.

**A — baseline:** `Conv2d(1, 8, 3, padding=1, bias=False) → BatchNorm2d(8) → ReLU → MaxPool2d(2, 2) → Conv2d(8, 16, 3, padding=1, bias=False) → BatchNorm2d(16) → ReLU → MaxPool2d(2, 2) → Flatten(1) → Linear(64, 10)`.

Le Conv usano stride=1, dilation=1, groups=1 e padding a zero; il pooling usa padding=0 e ceil_mode=False. Le BN hanno eps=1e−5, momentum=0.1, parametri affine e running statistics abilitate.

**B — unica modifica:** sostituite entrambe le BN con Identity, senza aggiungere bias alle Conv o cambiare altri fattori. L'intervento riguarda l'intero componente BN, compresi i suoi parametri affine.

Inizializzazione Conv: He normale fan-in/ReLU; Linear: Xavier uniforme gain=1 e bias zero. In A, gamma BN=1, beta=0, running_mean=0, running_var=1 e contatori zero. Copiate e confrontate esplicitamente i tensori iniziali condivisi da A e B. Seed base **20261022**, generatore dedicato del loader reinizializzato per ogni corsa, nuovo ottimizzatore ogni volta.

**Adam:** lr=0.001, betas=(0.9, 0.999), eps=1e−8, weight_decay=0. Cross-entropy media sui logits, senza pesi di classe e con target di classe non smussati (label smoothing disattivato). Budget **30 epoche per corsa**; non cambiatelo in base ai risultati. Il budget è stato verificato su CPU; i log reali di recupero sono inclusi e ne va dichiarata la provenienza quando usati.

Prima di eseguire B, scrivete un'ipotesi sulla misura primaria. Potete prevedere, per esempio, che A con BN raggiunga un macro-F1 validation superiore a B nel budget fissato; dichiarate quale segno di B−A vi aspettate e quale risultato non sosterrebbe la previsione. Il segno non è garantito dal protocollo.

Selezionate per ciascuna corsa il checkpoint con **massimo macro-F1 validation, prima epoca in caso di parità**; completate comunque le 30 epoche e registrate anche l'ultima epoca. Misura primaria: **macro-F1 validation di B meno quello di A**, ciascuno al proprio checkpoint selezionato. Il salvataggio del migliore checkpoint non è early stopping.

## Lab 1 — Kernel, forme e receptive field

### N01 — Dati

Verificate conteggi, indici disgiunti, dtype, range e forma. Registrate come e su quali dati avete stimato le statistiche di canale, con accumulo in float64 e successiva trasformazione degli input in float32. Spiegate la differenza rispetto alla standardizzazione per pixel della L10. Quale range vi aspettate prima e dopo la standardizzazione, senza confondere i due passaggi?

### N02 — Kernel appreso

Usate `X=[[1, 2, 0], [0, 1, 3], [2, 1, 0]]`, `K=[[1, 0], [0, -1]]`, cross-correlazione valid, stride 1 e nessun bias. Target T=0 con forma 2×2; `L=0.5·sum((Y−T)²)`, senza media. Calcolate forward, gradiente condiviso e un passo SGD con eta=0.1. Verificate tutti i quattro elementi del gradiente con differenze finite e con l'autograd del notebook. Spiegate la provenienza della somma dei contributi. L'esempio algebrico è distinto dal training della CNN con Adam e CE.

### N03 — CNN

Completate i layer indicati, stampate le forme principali e conteggiate parametri e buffer separatamente in A e B. Annotate quali quantità dipendono dalla dimensione del batch. La testa deve produrre logits per dieci classi.

### N04 — Receptive field

Compilate r e j dopo ciascuna Conv e Pool, iniziando da r=j=1. Inserite nel notebook uno schizzo o una breve descrizione di una griglia 8×8 riferita allo stesso esempio training e il riquadro nominale di una posizione della feature map. Indicate posizione, bordo dell’immagine e padding; non è richiesta una nuova routine grafica. Dichiarate la modalità della BN e spiegate che cosa cambia nell'interpretazione del supporto tra train ed eval.

**Checkpoint 1:** mostrate contratto dei dati, forme, conteggi e griglia RF; spiegate il ruolo della condivisione dei pesi. Tenete distinti kernel, feature map e receptive field.

## Lab 2 — Normalizzazione e stato

### N05 — BatchNorm a quattro valori

Un canale con N=2, H=1, W=2 contiene i valori (1, 3, 5, 7). Gamma=1, beta=0, eps=1e−5, momentum=0.1, running_mean iniziale 0, running_var iniziale 1. Calcolate media, varianza del forward, output train e buffer aggiornati. Distinguete la varianza usata nel forward da quella usata per aggiornare running_var. Poi valutate gli stessi valori con i buffer aggiornati.

### N06 — Modalità e stato

Su copie inizialmente identiche confrontate train+no_grad ed eval+no_grad. Registrate parametri, buffer e output prima/dopo. Spiegate perché il confronto deve specificare lo stato iniziale di ciascun ramo. Ripristinate completamente le copie e non contaminate lo stato ufficiale delle corse.

**Checkpoint 2:** motivate gli assi della normalizzazione, distinguete parametri e buffer e spiegate che cosa occorre salvare per ricaricare la stessa funzione eval. Documentate se e dove lo stato cambia durante il forward.

## Lab 3 — Baseline e diagnostica

### N07 — Pairing A/B

Costruite gli stati A e B dal medesimo insieme di tensori comuni; verificate l'uguaglianza. Controllate l'ordine degli ID nei batch per epoca senza consumare il generatore ufficiale. Annotate la creazione di un Adam nuovo per ogni corsa.

### N08 — Sanity check

Su una copia di B scegliete 20 immagini training, due per classe con minori ID. Usate batch intero e 200 step dell'Adam di progetto; osservate loss e accuracy. Documentate che cosa avete controllato e che cosa il risultato non dimostra. Non riusate questa rete o il suo ottimizzatore nelle corse principali. Non sostituite il risultato osservato con un valore desiderato.

### N09 — Sonde

Raccogliete attivazioni Conv/ReLU, frazione di zeri, gradienti dopo backward e aggiornamenti effettivi dopo step. Dichiarate layer, modalità, batch e istante. Usate frequenza uguale in A/B e il meccanismo fornito dal notebook, evitando forward aggiuntivi in train sulla rete ufficiale. Spiegate perché attivazione nulla, gradiente nullo e parametro invariato sono fatti distinti.

### N10 — Baseline A

Eseguite il budget fissato. Conservate separatamente loss online, train_eval e validation; annotate modalità, stato e aggregazione usati. Dopo la valutazione controllate parametri e buffer e ripristinate train.

**Checkpoint 3:** mostrate il controllo dello stato iniziale e del valutatore, un esito del sanity check, una sonda contestualizzata e la legenda delle curve. Se un controllo fallisce, indicate il punto da risolvere prima di interpretare la prestazione.

## Lab 4 — Ablation e interpretazione

### N11 — Variante B

Se avete già visto i risultati nelle figure o nei log di riferimento, dichiaratelo in C30: la replica didattica non richiede di presentare una spiegazione retrospettiva come previsione indipendente.

Prima di avviare B registrate ipotesi, variabili fisse, unica modifica e misura primaria. Eseguite B da stato nuovo e appaiato, mantenendo il protocollo. Annotate ogni deviazione accidentale invece di ignorarla.

### N12 — Checkpoint

Selezionate, copiate e ricaricate lo stato migliore di ciascuna corsa, inclusi i buffer necessari. Verificate le predizioni della copia ricaricata rispetto al checkpoint originale. Documentate anche le metriche dell'ultima epoca.

### N13 — Confronto

Calcolate B−A, definite il gap usato e interpretate loss e macro-F1 ai checkpoint selezionati e all'ultima epoca. Formulate un controllo che possa distinguere una possibile spiegazione da un errore della procedura. Dichiarate i limiti del singolo seed e del test già noto.

### N14 — Figure ed errori

Sullo stesso esempio training mostrate kernel e feature map prima/dopo, con layer, modalità e scale dichiarati. Descrivete ciò che osservate senza attribuire automaticamente una funzione causale a un kernel. Esaminate tre errori validation nell'ordine stabile degli ID; se sono meno di tre, dichiaratelo. Associate osservazione e ipotesi controllabile.

### N15 — Scheda

Completate F01–F05 e la differenza primaria. Identificate separatamente output propri e log di recupero dell’archivio originale. Nessuna misura può essere inventata e non si cerca il seed più favorevole.

**Checkpoint 4:** consegnate una conclusione valida anche se B peggiora, la differenza è piccola oppure il confronto è rimasto incompleto. Collegate ogni affermazione a una misura o a un controllo documentato.

### N16 — Estensione facoltativa, non valutata

Eseguite tre repliche appaiate con seed 20261022, 20261023 e 20261024. Riportate tutte le differenze, la loro media e deviazione standard campionaria. Spiegate quali conclusioni non sono giustificate da sole tre coppie di corse. Questa estensione non è richiesta né nello studio né nella consegna individuale. Lasciate `RUN_REPLICATES=False` in C39 se non scegliete di approfondirla.

## Esercizi di consolidamento D01–D08

D01–D08 sono **facoltativi e non valutati**: non occorre consegnarne lo svolgimento. D01–D04 riprendono calcoli già presenti in N02–N05 e non richiedono una seconda soluzione. Per il checkpoint è sufficiente P03; D05 resta studio ulteriore.

| ID | Consegna |
|---|---|
| D01 | Compilate forme e parametri della CNN A/B, distinguendo pesi dei kernel, canali, attivazioni, bias e buffer. |
| D02 | Calcolate r e j lungo la CNN; discutete bordi, RF nominale, testa finale e modalità BN. |
| D03 | Risolvete integralmente l'esempio X/K di N02: forward, loss, gradiente condiviso, step, nuova loss e differenze finite per ogni peso. |
| D04 | Risolvete N05 e confrontate train/eval: due varianze, running statistics, output e significato di no_grad. |
| D05 | Interpretate la tabella sintetica seguente: checkpoint, andamento, due controlli e limite della diagnosi. |
| D06 | Un gruppo cambia BN, lr, augmentation e seed, poi sceglie la configurazione dal test. Individuate i problemi e riscrivete un protocollo A/B. |
| D07 | Un tensore ha norma 2 e aggiornamento effettivo di norma 0.004; un bias ha norma iniziale zero e aggiornamento non nullo. Calcolate e interpretate il rapporto quando definito; spiegate come documentare il secondo caso e perché un gradiente piccolo non dimostra inutilità del layer. |
| D08 | Input 15×15, tre conv k=3, padding=0, dilation 1, 2, 1 e stride tutti 1. Calcolate shape e RF; confrontate i pesi di un 3×3 dilatato con quelli di un denso 5×5 a canali uguali. Ripetete con stride 2 soltanto nella seconda conv. Distinguete supporto, campionamento e numero dei pesi. |

**D05 — dati sintetici per la discussione, non risultati Digits.** Le CE sono valutate in eval a stato fissato. Questa tabella non può sostituire le misure delle corse A/B nella consegna.

| Epoca | train_eval CE | validation CE | validation macro-F1 |
|---:|---:|---:|---:|
| 1 | 0.90 | 0.95 | 0.70 |
| 5 | 0.30 | 0.40 | 0.86 |
| 10 | 0.10 | 0.55 | 0.84 |
| 15 | 0.03 | 0.80 | 0.81 |

## Challenge facoltative, non valutate

Se desiderate un approfondimento, scegliete **al massimo una** challenge C01/C02/C03 in C37; potete anche saltare C36–C37. Per la challenge scelta annotate: **contratto violato, controllo, correzione e limite della conclusione**. Le altre restano disponibili per studio.

- **C01 — Stato che cambia in validation.** Il valutatore racchiude il forward in no_grad, ma lascia la rete in train. Confrontate stato prima/dopo e definite il recupero corretto.
- **C02 — Confronto non appaiato.** A e B hanno pesi iniziali diversi e condividono un loader il cui generatore non viene reinizializzato. Come verificate e correggete il confronto?
- **C03 — Curve non confrontabili.** Una relazione interpreta la differenza fra loss online di training e loss validation come prova di overfitting. Quali misure e controlli occorrono?

## Autoverifica E01–E05

Queste domande aiutano a controllare la comprensione e **non richiedono una risposta nella consegna**.

- **E01:** come il gradiente di un peso convoluzionale incorpora più posizioni dell'immagine?
- **E02:** quali sono r e j dopo Conv 3×3 stride 1 e Pool 2×2 stride 2, partendo da r=j=1?
- **E03:** perché no_grad non basta per una validation con BN?
- **E04:** quali tre elementi rendono controllato il confronto A/B di oggi?
- **E05:** se Δmacro-F1 è vicino a zero in un solo seed, quale conclusione è giustificata?

## Consegna F01–F05

Nella cella **C40**, scrivete una sintesi **entro 500 parole complessive** articolata in F01–F05. Rimandate alle tabelle e alle figure già presenti nel notebook, senza ricopiare gli output e senza preparare un PDF separato:

- **F01 — Ipotesi e protocollo:** previsione registrata prima di B, unica modifica, fattori fissi, misura primaria e criterio di selezione.
- **F02 — Architettura e controlli:** forme e conteggi essenziali, RF e condizioni di validità, esiti dei controlli dei dati e dello stato.
- **F03 — Misure e curve:** A/B ai checkpoint selezionati e all'ultima epoca, epoca, CE train_eval/validation, macro-F1 e accuracy validation, gap definito, tempo e step; ΔB−A. Distinguite le serie online/eval.
- **F04 — Diagnosi:** osservazione documentata, interpretazione, controllo proposto e limite; eventuale lettura di figure ed errori.
- **F05 — Riproducibilità:** configurazione, ambiente, indici/hash, preprocessing, seed, checkpoint e verifica della ricarica; origine dei log e delle figure.

Consegnate il notebook individuale con nome e cognome, sei esercizi guidati, risposte N01–N15, sintesi C40 e output; allegate l’intera cartella `output_lezione12/` con configurazione e log. Non includete nel confronto principale test, nuove augmentation o prove scelte perché favorevoli. In caso di recupero eseguite almeno i controlli, un aggiornamento e un breve tratto; analizzate poi i log reali predisposti nell’archivio originale, dichiarando quali parti non avete eseguito personalmente. Se un risultato manca, riportate che manca.