# Computer Vision · Lezione 14 in Colab

Il notebook è individuale: inserire nome e cognome in C01, completare C11/C19/C29 e i richiami testuali P01/C13 e P02/C17, poi . Le 39 celle e le 21 celle di codice restano invariate. Sintesi F01–F05 in C36 entro 500 parole, senza relazione separata.

## Avvio prima di iniziare

1. Scaricare sul computer il pacchetto `Transfer_Learning_Studio_CV2026_studenti.zip` e il notebook `studenti/Transfer_Learning_Studio_CV2026_studenti.ipynb` (si può estrarre dal pacchetto).
2. Aprire [Google Colab](https://colab.research.google.com/), **File → Carica notebook**, e scegliere il file `.ipynb`. Selezionare runtime Python con **nessun acceleratore (CPU)**. Tutti i tensori e i timer di questa lezione usano CPU: selezionare GPU nel menu non sposta automaticamente il lavoro sulla GPU.
3. Eseguire **C03**, prima del percorso operativo. Al primo avvio la cella chiede il pacchetto ZIP; scegliere quello completo, con dati, pesi, supporto e recupero. C03 estrae in `/content/ComputerVision_Lezione14/Transfer_Learning_Studio_studenti` e lavora nella sua cartella `studenti`. Il notebook rimane lo stesso: non occorrono un secondo launcher o due runtime condivisi.
4. C03 conserva la coppia torch/torchvision già presente in Colab. Se manca `timm`, installa la versione 1.0.30 vincolando torch/torchvision alle versioni del runtime. L’installazione iniziale richiede rete; dati e pesi sono già nello ZIP. **Non eseguire `pip install -r requirements.txt` in Colab**: quel file descrive l’ambiente locale di riferimento. Se la coppia preinstallata non si importa, ripartire da un runtime pulito invece di aggiornarla alla cieca.
5. Eseguire **C05, C07, C09** e verificare hash, forme, caricamento strict e ambiente stampato. C05 controlla gli hash contro il manifest. Completare C11, C19, C29 nella propria copia durante la settimana; P01/P02 richiedono solo risposta testuale nelle celle originali. Le etichette Cxx e i passaggi delle figure restano invariati.

Il riferimento pubblicato usa CPU Mac arm64, Python 3.12/PyTorch 2.14, un thread e float32. Una corsa Colab è **una nuova esecuzione**, da identificare con il proprio ambiente e i propri tempi. Seed uguale non garantisce risultati identici fra piattaforme/versioni. Il protocollo non cambia: LP 20 epoche, FT parziale 3, batch 32, stessi ID e preprocessing, nessun uso del test. Latenza resta preprocessing+forward CPU batch 1. Non confrontare un tempo CPU di riferimento con una misura GPU sotto lo stesso nome.

## Salvataggio e download

C14/C15 scrivono le teste LP e le cache; C20/C21 scrivono i delta FT; C32 la figura; **C37** scrive JSON e CSV in `OUT`, stampato dal notebook. Prima di chiudere, scaricare il notebook dal menu File e la cartella output. Per creare e scaricare un archivio, eseguire questo blocco in una cella **temporanea in fondo, dopo C39**; rimuoverla prima della consegna per conservare le 39 celle originali:

```python
from google.colab import files
import shutil
archive = shutil.make_archive(str(OUT.parent / 'Consegna_Lezione14'),
                              'zip', root_dir=OUT)
files.download(archive)
```

Prima del download controllare identità in C01, risposte P01/P02 e N01–N06, sintesi C36 e output. Salvare in Drive il notebook conserva testo, risposte e output incorporati; **non conserva automaticamente** file `.pt`, JSON/CSV e variabili del runtime. Scaricare i risultati anche alla pausa, usando lo stesso blocco: in quel momento l’archivio è parziale e va indicato. Una disconnessione può comportare la perdita della macchina virtuale; non affidarsi alla persistenza di `/content`.

## Recupero dopo riavvio o su macchina lenta

- Se il runtime è nuovo, C03 richiede nuovamente lo ZIP. Se i file sono ancora presenti, li riusa. In entrambi i casi le variabili vanno ricostruite. Non rieseguire C09 in mezzo al FT: ricrea teste casuali.
- Per una corsa completa ripartire dall’inizio, con i tre TODO salvati. Per il solo FT seguire C38: C03, C05, C09, C11, C19, poi caricare le teste LP e dichiarare i risultati LP importati, quindi C20–C22. Un FT interrotto non si riprende esattamente con i soli delta: non contengono lo stato Adam/RNG.
- Per leggere subito usare `recupero/Risultati_riferimento.html`, scaricabile/apribile sul computer. Per eseguire il recupero aprire in Colab `recupero/Ripristino_risultati.ipynb`: R02 ha lo stesso avvio ZIP, ma sceglie la cartella `recupero`; seguire R01–R08. Questo secondo notebook ha il **proprio runtime**: non presumere che erediti variabili o file del primo.
- Dopo R08, per stress/challenge copiare C29–C35 nel notebook di recupero, completare C29 e procedere nello stesso kernel. I delta sono copiati nella sua cartella `output_recupero`. Non ripartire da C09. Se si misura C25, etichettarla come nuova latenza locale; i tempi LP/FT letti rimangono quelli del riferimento.
- Se R05 non conferma le predizioni del riferimento, conservare l’errore e l’ambiente e consultare i risultati salvati; non modificare metriche o tolleranze per far passare il controllo. Un controllo fallito va diagnosticato, non presentato come riproduzione riuscita.

**Stato della verifica:** notebook e ripartenze eseguiti realmente in kernel locali CPU, da ZIP estratti; ramo di upload/percorso Colab verificato con simulazione locale. Non è stata effettuata una sessione Google Colab autenticata né una corsa GPU; non sono riportati tempi cloud. Fonti: [FAQ Colab](https://research.google.com/colaboratory/faq.html), [gestione file Colab](https://colab.research.google.com/notebooks/io.ipynb), [riproducibilità PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html), consultate il 25 settembre 2026.