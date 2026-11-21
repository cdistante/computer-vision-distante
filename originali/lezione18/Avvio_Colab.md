# Avvio in Colab · Foundation Vision Studio

1. Aprire il [notebook del capitolo 13 in Colab](https://colab.research.google.com/github/cdistante/computer-vision-distante/blob/main/originali/lezione18/studenti/Foundation_Vision_Studio_CV2026_studenti.ipynb). Per la dimostrazione facoltativa, leggere C39a ed eseguire C39b su CPU con NumPy e Matplotlib.
2. Per il laboratorio con SAM 2.1, preparare i materiali nello stesso runtime con la cella indicata nelle [istruzioni del repository](README.md#google-colab).
3. Scegliere CPU o una GPU CUDA compatibile. Eseguire C03 per verificare il dispositivo. Per forzare CPU, impostare la variabile di ambiente CV_DEVICE a cpu prima di eseguire C03.
4. Eseguire C05 prima dell’esecuzione. Scarica circa 149 MiB SAM, 12.5 MiB LRASPP e l'archivio PennFudan. Le corse successive riusano la cache finché il runtime resta attivo. Codice SAM fissato al commit del manifest; Torch/Torchvision compatibili del runtime vengono conservati.
5. Completare i TODO C09, C23 e C35 prima delle celle che li usano. Seguire i passaggi Cxx nelle figure. Usare le etichette Cxx per trovare le celle di testo e codice.
6. Dopo i risultati semantici e C12/C38, C40 salva la corsa completa, C42 scarica lo ZIP. Queste celle rifiutano uno stato incompleto: non si usano per un export parziale durante la pausa. Salvare anche il notebook dal menu File. Il disco della VM non è permanente.

## DINOv3 autorizzato

Il riferimento consegnato non ha eseguito DINOv3. Per attivarlo ottenere personalmente l'accesso dal [sito ufficiale](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) e leggere la licenza. Caricare nella VM il proprio checkpoint ViT-S/16 e avere una copia del [repository ufficiale](https://github.com/facebookresearch/dinov3). In C12 assegnare direttamente i percorsi, per esempio `DINO_CHECKPOINT='/content/mio_checkpoint.pth'` e `DINO_REPO='/content/dinov3'`, poi C13/C14. La forma attesa dei patch token è (256, 384) con input 256. Conservare hash e revisione del repository nel proprio report. Non inserire token o URL personali nel notebook distribuito.

Se manca l'autorizzazione, C13/C14 mostrano uno stato esplicito. Restano eseguibili i calcoli analitici e il percorso SAM/LRASPP. Non sostituire modelli senza modificare dichiaratamente il protocollo.

## Recupero senza inferenza

Aprire `recupero/Ripristino_risultati.ipynb`, eseguire R02–R05 e caricare lo ZIP nello stesso runtime se richiesto. Il manifest deve appartenere alla lezione 18, anche se in `/content` sono già presenti altre lezioni. Il percorso legge JSON/maschere e controlla hash. Non richiede pesi o Torch e non completa i TODO. L'HTML può essere letto offline. I tempi rimangono quelli della corsa CPU originale.

La versione consegnata è stata eseguita localmente su CPU. Upload dei byte, scelta della cartella, riuso e download sono stati simulati localmente con le celle distribuite; nessuna prova autenticata nel servizio Google o su una sua GPU è dichiarata.