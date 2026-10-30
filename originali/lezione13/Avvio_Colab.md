# Computer Vision · Lezione 13 in Colab

Il notebook è una dimostrazione completa di 35 celle, 17 di codice. Non si richiedono codice da completare, una relazione o file da inviare per la Lezione 13. I download descritti qui servono soltanto a conservare una copia personale.

## Caricare tutti i materiali nello stesso runtime

1. Aprire [Google Colab](https://colab.research.google.com/) e usare **File → Carica notebook** per `Dalle_CNN_ai_transformer_studenti.ipynb`. Non caricare lo ZIP come notebook.
2. Scegliere un runtime Python con **nessun acceleratore (CPU)**. Gli esempi analitici C03–C27 usano float64; la fotografia C29–C35 usa float32. Non installare il requirements locale sopra i pacchetti di Colab. Se manca una dipendenza, installare soltanto quella indicata dal messaggio della cella.
3. Caricare nel pannello **File** di Colab anche `Dalle_CNN_ai_transformer_studenti.zip`. Estrarlo nello stesso runtime: il notebook deve poter trovare `supporto/cls_attention.py`, `input/chelsea.png` e `pesi/deit_tiny_patch16_224-a1311bcf.pth`. Seguire le istruzioni delle celle C01–C02. Il solo `.ipynb` basta per i piccoli esempi iniziali, ma non per la fotografia. Il checkpoint è già nel pacchetto: non occorre scaricarlo durante la dimostrazione.

Se serve estrarre manualmente il pacchetto, eseguire questo codice in una cella temporanea **in fondo al notebook**, dopo C35, prima di avviare le demo:

Per caricare il pacchetto completo e scegliere la cartella corrente seguire le istruzioni Colab nel README di questo laboratorio. Le celle scientifiche Cxx rimangono invariate.

Non è necessario spostare a mano i file delle sottocartelle. C29 individua le risorse nel pacchetto estratto. Dopo l’estrazione eliminare la cella temporanea per conservare i 35 riferimenti C01–C35; non inserire celle in mezzo al percorso.

## Avvio, ripartenze e risultati

Prima della lezione eseguire C03 e C29. C03 prepara i piccoli esempi e registra le versioni; C29 carica su CPU il modello fotografico con tutti i suoi import. Cxx conta anche le celle di testo; `In[]` dipende dall’ordine di esecuzione. La scelta del browser non cambia questi riferimenti.

- Residui: C03, C05, C07.
- Patch: C03, C09, C11.
- Attenzione assegnata: C03, C13, C15, C17.
- Permutazioni: C03, C13, C15, C21, C23.
- Sola fotografia, anche da kernel nuovo: **C29, C31, C33, C35**.
- Tutto il notebook: **Esegui tutto** da un kernel nuovo.

Preparare C29, eseguire C31 e C33 e leggere fotografia, mappa e sovrapposizione. C35 serve al salvataggio personale e può essere eseguita dopo la lettura dei risultati. Non si effettua training né una valutazione sul test set.

C27 salva `risultati_lezione13.json`; C35 aggiorna lo stesso file con `cls_patch`, includendo soltanto i blocchi eseguiti. La figura è in `output_cls/cls_attention_chelsea.png` e le misure in `output_cls/cls_attention_misure.npz`. Prima di disconnettere scaricare il notebook dal menu **File** e i file desiderati dal pannello **File**. Per scaricare il JSON con codice, aggiungere soltanto in fondo, dopo C35:

```python
from google.colab import files
files.download('risultati_lezione13.json')
```

Dopo il download eliminare questa cella temporanea. Non occorre inviare i file al docente. Salvare una copia del notebook in Drive conserva testo, codice e output, ma non conserva automaticamente lo stato Python o i file della macchina virtuale. Dopo un riavvio caricare nuovamente il pacchetto se manca, estrarlo e rieseguire le dipendenze; gli output visibili non ricreano le variabili.

**Verifica disponibile:** esecuzione locale su CPU; una sessione Google Colab autenticata non è stata provata e non sono disponibili tempi cloud misurati. Le regole di persistenza e caricamento sono documentate nelle [FAQ ufficiali Colab](https://research.google.com/colaboratory/faq.html) e nel [notebook ufficiale di gestione file](https://colab.research.google.com/notebooks/io.ipynb).