# Come impara una rete — scheda di lettura

**Computer Vision · Lezione 11 · percorso di due ore.**

La Lezione 11 usa un notebook dimostrativo completo: non richiede un elaborato da consegnare, risposte da inviare o codice da completare. Questa scheda raccoglie esempi svolti da discutere durante lo studio o consultare per il ripasso. F01–F04 ed E01–E05 sono riferimenti per orientarsi nelle figure e nella guida; non sono parti di un compito. Il lavoro individuale è contenuto nel notebook della Lezione 12,  , secondo le sue istruzioni.

## F01 — Un grafo spiegato

Con `x=2`, `w=3`, `b=−1`, `y=1` e `a=wx+b; h=ReLU(a); L=(h−y)²/2`, il forward dà `a=5`, `h=5`, residuo 4 e `L=8`. Le derivate locali sono `dL/dh=4`, `dh/da=1`, `da/dw=2`, `da/db=1` e `da/dx=3`: moltiplicandole lungo i percorsi si ottengono `dL/dw=8`, `dL/db=4` e `dL/dx=12`. I parametri sono `w` e `b`; `x` è un dato e `y` il target.

Nel caso `q=x*x+x`, a `x=2` il valore è 6 e la derivata è `2+2+1=5`: si sommano i contributi dei due ingressi del prodotto e del ramo diretto. Durante il backward si calcolano i gradienti; i parametri non cambiano fino allo step dell’ottimizzatore. Le celle C05 e C07 mostrano entrambi i casi già svolti.

## F02 — Dal gradiente all’aggiornamento

Per `L(theta)=theta²/2` e `theta=2`, la loss iniziale è 2 e il gradiente è 2. Un passo SGD con `eta=0.5` produce `theta=1`, loss `0.5`; con `eta=2.5` produce `theta=−3`, loss `4.5`. Una derivata corretta non garantisce una diminuzione della loss con qualunque learning rate. Il confronto è già calcolato in C15.

Nel primo passo Adam di C17, il gradiente assegnato è 2, i momenti iniziali sono nulli, `beta1=0.9`, `beta2=0.999`, `eta=0.1` ed `epsilon=1e−8`. Si ottengono `m1=0.2`, `v1=0.004`, `m_hat1=2` e `v_hat1=4`; con epsilon fuori dalla radice nel denominatore, `theta1≈1.9000000005`. È un caso analitico, non un risultato di classificazione su immagini.

## F03 — Leggere un confronto controllato

Il protocollo stampato in C27 prepara la lettura dello Studio 12. L’unico fattore del confronto comune è BatchNorm rispetto a Identity: non occorre proporre o avviare un secondo esperimento nella Lezione 11.

| Campo | Esempio già compilato |
|---|---|
| Domanda | Quale effetto ha sostituire BatchNorm con Identity nel protocollo fissato? |
| Baseline | La CNN dello Studio 12 con BatchNorm. |
| Una modifica | Sostituire BatchNorm con Identity, mantenendo la struttura comune. |
| Ipotesi | La normalizzazione può cambiare l’andamento dell’ottimizzazione; un eventuale vantaggio va misurato, non presunto. |
| Invarianti | Dati e split, pesi comuni iniziali, ordine dei batch, Adam, budget di 30 epoche e criteri di validation. |
| Controllo | Verificare il solo fattore variato, le modalità train/eval e che i parametri previsti ricevano gradienti e aggiornamenti. |
| Validation | Applicare il medesimo criterio di scelta del checkpoint e la medesima regola di pareggio, dichiarati nel notebook dello Studio. |
| Misure | Leggere curve train_eval e validation nello stesso stato, conteggi, configurazione e durata effettiva. |
| Risultato | Da misurare nello Studio. Il testo di un’ipotesi non sostituisce una misura. |

Non si usa il test per scegliere la modifica. Il test Digits già discusso nella Lezione 9 non torna indipendente se viene riutilizzato. Una sola corsa può verificare il funzionamento, ma non dimostra un vantaggio stabile fra configurazioni.

## F04 — Esempio di nota di metodo

> Voglio evitare di confrontare una loss online con dropout e una validation deterministica come se fossero la stessa misura. Ricalcolerei train_eval e validation sul medesimo checkpoint, in eval e senza augmentation casuale, separando data loss e penalità. I passi SGD e Adam qui discussi sono calcoli analitici; le curve e l’effetto dell’ablation richiedono invece un’esecuzione del protocollo. Una sola corsa non dimostra un vantaggio stabile.

La nota distingue un errore, un controllo e un limite. È un esempio già svolto da leggere e discutere, non un testo da consegnare.

## E01–E05 — Ripasso guidato

- **E01 — Come si combinano i contributi di un nodo riutilizzato?** Si moltiplicano le derivate lungo ogni percorso e si sommano i contributi che tornano allo stesso nodo.
- **E02 — Che differenza c’è fra backpropagation e SGD/Adam?** Il backward calcola i gradienti; l’ottimizzatore li usa insieme al proprio stato e agli iperparametri per aggiornare i pesi.
- **E03 — Perché unità nascoste identiche possono essere un problema, mentre i bias zero possono essere corretti?** La simmetria può mantenerle indistinguibili. Pesi distinti possono rompere la simmetria anche con bias zero; un classificatore lineare inizialmente nullo è un caso diverso da una MLP ReLU interamente nulla.
- **E04 — Quale differenza c’è fra L2, dropout ed early stopping?** L2 aggiunge una penalità all’obiettivo; dropout azzera temporaneamente attivazioni del forward di training con maschere casuali, senza eliminare neuroni o pesi; early stopping interrompe il training secondo una regola di validation. Selezionare il best checkpoint dopo un budget fisso non è early stopping.
- **E05 — Quando una trasformazione è un’augmentation valida?** Quando il target resta difendibile e immagine, maschera o box ricevono il trattamento coerente con il task. L’utilità si misura con un confronto controllato su validation; non è garantita dalla sola plausibilità visiva.