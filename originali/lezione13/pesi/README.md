# Pesi della dimostrazione CLS-verso-patch

DeiT Tiny patch16_224 **non distillato**, preaddestrato su ImageNet-1k. I pesi sono congelati: la lezione esegue soltanto inferenza su CPU, senza addestramento o accesso a un test set.

- [Checkpoint originale](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth), circa 22 MB.
- [Progetto DeiT](https://github.com/facebookresearch/deit), licenza Apache-2.0 del progetto.
- Touvron et al., [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a.html), ICML 2021.
- SHA-256: `a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97`.

Il modulo di supporto controlla integrità e corrispondenza di tutte le chiavi del checkpoint. Non sostituire questo file con la variante distillata, che contiene un token speciale aggiuntivo.