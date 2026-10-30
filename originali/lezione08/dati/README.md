# Dati del 3D Geometry Studio

Fonte: [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/), scena [Adirondack-perfect](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-perfect/). Scharstein et al., *High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth*, GCPR 2014.

Nei materiali del corso si usa il **punto decimale** (0.5) e la **virgola seguita da uno spazio** fra componenti di vettori e argomenti, per esempio (1.2, 1, −0.3). I conteggi interi non usano separatori delle migliaia.

## Contenuti

- `im0.png`, `im1.png`: coppia A RGB già rettificata, 720×497.
- `calibrazione.json`: K sinistra/destra, doffs, baseline in metri, trasformazione S e parametri di lavoro.
- `riferimento.npz`: disparità GT sinistra e destra (`disp0`, `disp1`) alla risoluzione di lavoro. I due riferimenti distribuiti hanno disparità positiva in modulo.
- `provenienza.json`: sorgenti dei dati e descrizione del ricampionamento.
- `coppia_B.npz`: immagini reali con trasformazione geometrica controllata (roll ±3°), G0/G1 e posa R/t. La funzione del notebook le rigenera dagli stessi dati A.
- `checkpoint_baseline.npz`: archivio di recupero non distribuito qui; il notebook principale ricalcola la pipeline.

## Trasformazione e unità

Originale 2880×1988; fattore s=1/4. Immagini ridotte con INTER_AREA. Coordinate dei centri pixel: u′=s(u+0.5)−0.5, v′=s(v+0.5)−0.5. S ha scala0.25 e traslazione−0.375; K′=SK. Disparità e doffs si moltiplicano per0.25; la baseline fisica non cambia.

Per la GT si seleziona il campione originale vicino al centro del pixel di destinazione, indice floor((u+0.5)/s), e si moltiplica la disparità per s. Non si mediano valori attraverso bordi di profondità. Il lettore PFM usa il segno dell'intestazione per l'endianness e ribalta verticalmente i campioni: i valori Middlebury sono già disparità in pixel, senza moltiplicazione per il valore assoluto del campo scale.

Valori di lavoro: f=1040.30525px, doffs=52.26475px, B=0.176252m. Convenzione d=uL−uR, doffs=cxR−cxL. Profondità Z=fB/(d+doffs). Coppia nota: P^(R)=P^(L)+(−B, 0, 0), C_L=(0, 0, 0), C_R=(B, 0, 0).

## Maschere e valutazione

Il notebook costruisce separatamente maschere di campionamento, GT valutabile e predizione valida. La valutazione usa GT finita positiva, u≥88, u<width−8, 8≤v<height−8. Per la challenge B si aggiungono 4px di margine verticale in entrambe le versioni. EPE e Bad-2 usano l'intersezione con la predizione valida; la copertura conserva il denominatore GT. Il confronto aggiunge una seconda lettura sugli stessi pixel validi in baseline e perturbazione.

La griglia di controllo GT (passo24) rimane fuori dalle stime: visibilità destra e consistenza delle GT entro0.75px. Match, F, E, posa e selezione dei punti 3D non usano la GT per scegliere le stime. Le metriche sono didattiche alla risoluzione ridotta, non risultati ufficiali del benchmark.

Le immagini e il riferimento provengono dagli autori del dataset. Conservare attribuzione e link alla fonte insieme a ogni ridistribuzione didattica. Le trasformazioni, i checkpoint e le maschere sono prodotti di questo laboratorio.