# Computer Vision · Sintesi svolta della Lezione 13

**Dalle CNN ai transformer.**

La Lezione 13 usa un notebook dimostrativo completo. Non occorre preparare una pagina, completare esercizi nel notebook o inviare file al docente. I riferimenti F01–F05 orientano questa sintesi già svolta e i collegamenti con le slide e la guida.

## F01 — Gradiente residuale, N01

Per F(x)=−0.2x e x=(1, 2), y=x+F(x)=(0.8, 1.6), L=1.6 e gradiente rispetto a x=(0.64, 1.28). Il contributo diretto è (0.8, 1.6), quello attraverso F è (−0.16, −0.32): si sommano perché gli stessi ingressi alimentano due percorsi. Il ramo puro produce un’altra uscita e un’altra loss; non dimostra che una shortcut garantisca sempre un gradiente maggiore. C04–C05 contengono spiegazione e codice completo.

## F02 — Una riga di attenzione, N06

La prima riga di A è circa (0.401112, 0.197776, 0.401112). I tre contributi dei value sono circa (0.401112, 0), (0, 0.395552) e (1.203336, 0.401112), con output circa (1.604448, 0.796664). La softmax normalizza lungo le key della singola query: le righe sommano a 1, le colonne non necessariamente. A può non essere simmetrica anche quando QKᵀ lo è. C12–C15 mostrano i passaggi senza training su immagini.

## F03 — Coefficienti e memoria, N11

Con input 224×224, CLS e 12 head, patch 16×16 danno T=197 e 465708 coefficienti di A; patch 8×8 danno T=785 e 7394700 coefficienti. Se A viene materializzata in float32, sono circa 1.776535 e 28.208542 MiB. Queste quantità non comprendono QKV, gradienti, altre attivazioni o l’intero modello; un kernel efficiente può evitare di conservare A esplicita. C24–C25 separano memoria e MAC.

## F04 — Interpretazione e limite

Dimezzare il lato delle patch quadruplica il loro numero e fa crescere fortemente il termine quadratico dell’attenzione. Il calcolo non determina da solo la latenza: proiezioni, MLP, implementazione e dispositivo contano. Una attention map mostra pesi del modello, senza diventare automaticamente una segmentazione o una prova causale della decisione. C26 discute un possibile controllo separato e i suoi limiti; non assegna un esperimento aggiuntivo. C28–C35 mostrano un esempio reale di CLS-verso-patch su Chelsea con DeiT Tiny: media dei pesi grezzi delle tre head dell’ultimo blocco, rinormalizzazione sulle 196 patch, griglia 14×14 e sovrapposizione. Il peso massimo circa 0.095411 è una quota di attenzione, non una probabilità di classe.

## F05 — Conservare la dimostrazione

C27 salva `risultati_lezione13.json` con i blocchi analitici effettivamente eseguiti; C35 aggiorna lo stesso file con `cls_patch`. C33 salva `output_cls/cls_attention_chelsea.png` e C35 salva `output_cls/cls_attention_misure.npz`. Chi desidera una copia personale completa può estrarre tutto il pacchetto, rieseguire le 17 celle di codice in ordine da un kernel nuovo e salvare notebook e risultati. La sola fotografia riparte con C29, C31, C33, C35 e conserva soltanto i suoi risultati, senza attribuirle blocchi non eseguiti. Non è richiesta la consegna di questi file. Le indicazioni per Colab sono in [Avvio Colab](Avvio_Colab.md).

Gli esercizi D01–D08 della dispensa restano facoltativi per studio e ripasso, con soluzioni nella guida docente. Il lavoro individuale della **Lezione 14** è distinto: si introduce il **29 ottobre 2026** e  secondo le istruzioni del relativo notebook.