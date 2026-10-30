# Pesi preaddestrati ufficiali

I file originali vengono caricati con strict=True prima di sostituire la testa. Conservare attribuzione e provenienza. Il copyright e le condizioni dei rispettivi progetti restano dei titolari.

- `resnet18-f37072fd.pth` SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- `deit_tiny_patch16_224-a1311bcf.pth` SHA256 `a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97`.

ResNet18: https://download.pytorch.org/models/resnet18-f37072fd.pth, torchvision https://github.com/pytorch/vision, licenza BSD-3-Clause del codice.

DeiT Tiny non distillato: https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth, https://github.com/facebookresearch/deit, licenza Apache-2.0 del progetto.

L’architettura DeiT viene istanziata con timm1.0.30, https://github.com/huggingface/pytorch-image-models. La scelta di preprocessing target è documentata e differisce dal crop di valutazione ImageNet.