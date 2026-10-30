# Computer Vision — Model card del progetto

Compilare usando modello, risultati e protocollo effettivi. Non copiare numeri analitici della Lezione 21. Non inserire nomi, email, credenziali o altri dati identificativi aggiuntivi. Non è un nuovo assignment.

| Campo | Descrizione da compilare |
|---|---|
| Modello e versione | Baseline/architettura, librerie, checkpoint e hash/revisione. |
| Uso previsto | Input, output, unità di decisione, scenario e destinatario della misura. |
| Usi non verificati | Domini/condizioni/decisioni che i risultati non sostengono. |
| Dati e split | Collegamento alla data card e indici/manifest effettivi. |
| Scelte di sviluppo | Parametri, criterio di checkpoint, validation; evitare selezioni sul test. |
| Metriche e riduzioni | Formula/convenzione, supporti, aggregazioni, soglie, unità. |
| Qualità per condizioni | Prestazioni per gruppi pertinenti con numerosità; “non misurato” se assente. |
| Probabilità e astensione | Se pertinenti: calibrazione separata, soglia congelata, coverage per gruppo e fallback. Non obbligatorie per ogni progetto. |
| Efficienza | Dispositivo, ambiente, batch/shape/dtype, warmup, ripetizioni e confine temporizzato. Non dedurre latenza dai soli FLOPs. |
| Riproducibilità | Comando/notebook, dipendenze, seed, pesi, trasformazioni, salvataggi; distinguere un manifesto da una prova in ambiente nuovo. |
| Limiti ed errori | Un caso documentato, supporto e conseguenza sull’uso. |
| Affermazione sostenuta | Una frase proporzionata alle evidenze e un collegamento al risultato. |
| Prossimo passo | Una revisione o una prova breve, senza nuovi benchmark obbligatori. |

Riferimento: [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993). La scheda descrive il comportamento verificato, non garantisce ogni utilizzo futuro.