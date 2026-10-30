# Fonti · Lezione 13

## Metodi e definizioni

- AlexNet: Krizhevsky et al., [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), NeurIPS 2012.
- ResNet: He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), CVPR 2016: somma residuale, shortcut e proiezione. Il caso y=x+F(x) è distinto dalla ReLU dopo la somma del blocco originale.
- ConvNeXt: Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), CVPR 2022: depthwise, pointwise, normalizzazione e struttura del blocco.
- GELU: Hendrycks e Gimpel, [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415), 2016: attivazione deterministica GELU(a)=aΦ(a).
- Attenzione: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017: query, key, value, scala e più head.
- ViT: Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), ICLR 2021: patch embedding, posizione, CLS ed encoder pre-LN.
- DeiT: Touvron et al., [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a.html), ICML 2021. La variante Tiny non distillata è usata nell’esempio fotografico della slide 56, della sezione 13.18 e delle celle C28–C35 del notebook dimostrativo, e nella lezione 14.
- Swin Transformer: Liu et al., [Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows](https://arxiv.org/abs/2103.14030), **ICCV 2021**, pp. 10012–10022: finestre locali alternate a finestre sfalsate per collegare regioni adiacenti nei blocchi successivi.
- MaxViT: Tu et al., [MaxViT: Multi-axis Vision Transformer](https://arxiv.org/abs/2204.01697), **ECCV 2022**, pp. 459–479, [DOI 10.1007/978-3-031-20053-3_27](https://doi.org/10.1007/978-3-031-20053-3_27): attenzione locale per blocchi e attenzione su griglie sparse di token distribuiti nell’immagine.
- PyTorch: [MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html), [LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html) e [scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

- CLS-verso-patch: Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html), **ICCV 2021**, pp. 9650–9660, §4.2.2 e figura 3; [codice ufficiale di visualizzazione DINO](https://github.com/facebookresearch/dino/blob/main/visualize_attention.py). Il metodo di estrazione della riga CLS non implica che DeiT supervisionato abbia le stesse proprietà sperimentali di DINO.
- Lettura su più layer: Abnar e Zuidema, [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/), **ACL 2020**, pp. 4190–4197. La media delle head di un solo blocco è distinta dall’attention rollout.

Il manoscritto docente **DeepLearning_Book**, sezioni sulle attivazioni e sui transformer, costituisce un ulteriore riferimento per il percorso GELU, query/key/value, aggregazione e più head. Le definizioni del corso distinguono l'encoder ViT pre-LN dall'encoder NLP post-LN.

## Figure ed esempi

Gli schemi del corso mostrano percorsi di calcolo e piccoli esempi assegnati. L’esempio fotografico della sezione 13.18 è invece una mappa misurata con pesi preaddestrati, descritta separatamente sotto. Le fonti sopra attribuiscono i metodi; le figure guidano la lettura di feature, somma residuale, blocco ConvNeXt, patch, attenzione e encoder.

| Schema | Slide | Lettura e riferimento concettuale |
|---|---|---|
| Pipeline CNN | 3 | Immagine, backbone, feature, testa e score s: ruoli del calcolo e forme, senza attribuire feature misurate allo schema |
| Shortcut con proiezione | 14 | Due rami con forma B×128×14×14 prima della somma; P(x) non è identità; ResNet di He et al. |
| Depthwise e pointwise | 17 | Collegamenti spaziali per canale e combinazioni dei canali per posizione; ConvNeXt di Liu et al. |
| Conteggio dei pesi | 18 | 200704, 3136, 32768 e 35904 soli pesi con C=64; conti analitici, senza equivalenza funzionale o misura di latenza |
| Ordine delle patch | 21 | Griglia, ordine raster e vettore dei valori dentro una patch; ViT di Dosovitskiy et al. |
| Proiezione condivisa | 23 | La stessa matrice E e lo stesso bias b per tutte le patch; numero dei parametri indipendente da N |
| Da mappa a sequenza | 24 | Conv2d con kernel/stride P, appiattimento degli assi spaziali e trasposizione in B×N×D |
| Posizione e CLS | 28 | CLS anteposto e somma E_pos riga per riga; le feature restano D |
| Ruolo di CLS | 29 | Embedding iniziale condiviso, interazione con le patch e rappresentazione finale per la testa |
| Confronti query/key | 34 | Una query fissa confrontata con più key; V non entra nel calcolo degli score |
| Varianza e scaling | 35 | Varianza d_k del prodotto non scalato e 1 dopo divisione per √d_k nelle ipotesi dichiarate; grafico teorico |
| Softmax per query | 36 | Barre impilate dei tre pesi per query nell'esempio delle S38–40; ogni barra somma a 1 |
| Somma pesata dei value | 37 | Peso scalare, vettore value e somma componente per componente; percorso di calcolo di AV |
| Asimmetria dei pesi | 42 | A13≈0.401112 e A31≈0.248255: denominatori per riga diversi nell'esempio assegnato |
| Concatenazione delle head | 46 | T righe conservate; tre blocchi di 4 feature affiancati, poi W_O; esempio con D=12 |
| Patch riordinate | 51 | Equivarianza senza posizione/maschere/dropout, CLS fisso e output CLS invariato; non invarianza a ogni permutazione di pixel |
| MAC del blocco | 52 | Termini 4TD², 2T²D e 2rTD² per D=192, r=4, T=197 e T=785; conteggi teorici, non tempi misurati |
| Memoria del solo A | 53 | Batch 1, 12 head, float32: 1.777 e 28.209 MiB arrotondati; esclusi altri tensori e gradienti |
| Località e interazioni globali | 54 | Dipendenze architetturali CNN/ViT, non attenzione misurata o graduatoria di accuratezza |
| Campo ricettivo nominale | 5 | Una, due e tre convoluzioni 3×3, stride 1 e dilation 1: supporti 3×3, 5×5 e 7×7; dipendenze possibili, non contributi misurati, §13.2 |
| 01 — Forward residuale | 10 | x nei due percorsi, correzione e somma; ResNet di He et al. |
| 02 — Backward residuale | 11 | Contributo diretto e del ramo F, poi somma delle derivate; regola della catena |
| 03 — ConvNeXt | 19 | Depthwise, LayerNorm, pointwise, GELU e residuo; Liu et al. |
| 04 — Pipeline ViT | 20 | Quattro patch, proiezione condivisa, cinque token con CLS, posizione, encoder e testa; Dosovitskiy et al. |
| 05 — Patch 4×4 | 25 | Quattro patch 2×2, ordine interno e ordine delle patch; esempio assegnato |
| 06 — Proiezione della patch | 26 | Da (0, 1, 4, 5) a (0.1, 1.2, 9.3), poi gli altri token; stesso esempio del notebook |
| 07 — Una head QKV | 32 | Z→Q/K/V, score scalati, softmax per riga e AV; Vaswani et al. |
| 08 — Output della query 1 | 41 | Tre pesi moltiplicano tre value e i contributi si sommano; esempio analitico |
| 09 — Più head | 45 | Tre output 5×4 concatenati in 5×12, poi W_O; Vaswani et al. |
| 10 — Encoder pre-LN | 48 | Normalizzazione prima del ramo e due somme residuali; encoder ViT |
| 11 — LayerNorm | 16 | Media 4, varianza 5, epsilon 0.00001 e valori a tre decimali su un solo token |
| GELU e ReLU | 49 | Grafici delle funzioni, dominio e valori a ±1; Hendrycks–Gimpel |
| Residuo con ReLU finale | 9 | Il punto dopo la somma e l'attivazione sono distinti; ResNet di He et al. |
| CLS-verso-patch, schema 2×2 | 55 | Riga CLS, esclusione di CLS-verso-CLS, ordine spaziale e rinormalizzazione; esempio assegnato, §13.18 |
| CLS-verso-patch, fotografia Chelsea | 56 | Ultimo blocco DeiT Tiny, media di tre head, griglia 14×14 e sovrapposizione; misura descritta sotto, §13.18 |

Il caso residuale usa x=(1, 2), F(x)=−0.2x e loss quadratica. Patch e token usano una matrice 4×4 con valori 0–15. L'attenzione usa tre query, key e value a due componenti, mantenuti identici alle celle del notebook. I risultati sono esempi analitici verificati in float64, senza training o download. La copertina mantiene la fotografia del formato del corso.

## Metodi recenti e collegamento alla dispensa

- Ali Hatamizadeh et al., [FasterViT: Fast Vision Transformers with Hierarchical Attention](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7e49642375e3315467fa33120813547f-Abstract-Conference.html), **ICLR 2024**.
- Dai Shi, [TransNeXt: Robust Foveal Visual Perception for Vision Transformers](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_TransNeXt_Robust_Foveal_Visual_Perception_for_Vision_Transformers_CVPR_2024_paper.html), **CVPR 2024**.
- Cheng Shi, Yizhou Yu, Sibei Yang, [Vision Transformers Need More Than Registers](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Vision_Transformers_Need_More_Than_Registers_CVPR_2026_paper.html), **CVPR 2026**.

Le fonti recenti sono collegate ai passaggi metodologici discussi nella dispensa. Le prestazioni degli articoli restano distinte dai risultati del notebook e dai suoi esempi analitici.

## Esempio fotografico CLS-verso-patch · Slide 56, §13.18 e C28–C35

- Fotografia **Chelsea**: Stefan van der Walt, **CC0**, distribuita da [scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.chelsea). La figura mostra il ritaglio realmente fornito al modello.
- Modello: **DeiT Tiny patch16_224 non distillato**, pesi ImageNet originali; [architettura ufficiale](https://github.com/facebookresearch/deit/blob/main/models.py) e [checkpoint ufficiale](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth). Input 224×224, 12 blocchi, 3 head, embedding 192.
- Protocollo: lato corto 256 con interpolazione bicubica, ritaglio centrale 224×224 e normalizzazione ImageNet. Inferenza CPU float32 in eval, senza addestramento. Dall’ultimo blocco si prende la riga CLS e si escludono le colonne dei token speciali; qui solo CLS. Media sulle tre head, poi rinormalizzazione sulle 196 patch; griglia 14×14 e sovrapposizione al vicino più prossimo.
- La figura è una misura di pesi interni del modello. Non è una segmentazione, una probabilità di classe, un risultato DINO o un attention rollout. Lo schema 2×2 precedente usa invece valori assegnati e mostra il calcolo a mano.

Il pacchetto notebook include la foto originale in `input/`, il checkpoint ufficiale in `pesi/` e il modulo PyTorch in `supporto/`. C31 esegue il forward con questi pesi; C33 visualizza il ritaglio effettivo, la griglia dei pesi e la sovrapposizione. Gli esempi analitici precedenti restano distinti da questa inferenza CPU float32. C35 conserva misure e protocollo insieme ai blocchi effettivamente eseguiti.