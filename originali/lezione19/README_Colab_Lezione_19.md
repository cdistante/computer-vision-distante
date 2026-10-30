# Avvio della Lezione 19

1. Scaricare ed estrarre il pacchetto in locale. In Colab aprire `Visione_multimodale_generativa_studenti.ipynb` e caricare il pacchetto ZIP nello stesso runtime tramite C03 se richiesto.
2. CPU è sufficiente. C03 verifica i materiali. C11 usa il checkpoint ufficiale OpenAI CLIP ViT-B/32. In locale installare `requirements_lezione19.txt` nel proprio ambiente.
3. Modalità live: C11 installa CLIP se manca e scarica circa 338 MiB di pesi. Completare prima dell’esecuzione. Il codice controlla l’hash. CPU è il default, CUDA è selezionabile con la variabile CV19_DEVICE e richiede disponibilità effettiva.
4. Modalità riferimento: impostare `MODE="riferimento"` in C03. C11 legge feature CPU preesistenti senza nuova inferenza. I calcoli successivi vengono rieseguiti. Non modificare l’ordine dei prompt della cache.
5. Seguire le indicazioni Cxx nel notebook. C31 esporta il ZIP degli output e in Colab ne avvia il download. C33 mostra risultati precedenti come recupero esplicitamente dichiarato.

I numeri Cxx includono celle di testo e di codice. Dopo un riavvio eseguire C03. Per CLIP: C11, C13, C15. Per ricostruzione: C17, C19, C21. C23, C25, C27 sono indipendenti dalla rete e dipendono da C03. La prova verificata è locale CPU, non una sessione Google Colab. Non consegnare pesi.