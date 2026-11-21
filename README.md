# Mech Royale

Versión de referencia: **3.33 oficial reportado por el usuario**, con la política
publicada en `c0e261dde754a55c51ca5ed63a2e9339cfe2802f`.
La pistola puede disparar simultáneamente al láser o los rayos cuando su aim ya
está alineado. La estrategia anterior obtuvo 3.17.

El resultado real confirma la mejora aunque las pruebas locales anteriores
fueran inconclusas. Historial: [REVIEW.txt](REVIEW.txt), [EVALUATION.json](EVALUATION.json).

Con Python y Docker (kit incluido):

```sh
python tools/retest.py --docker --check-only
```

El checker valida protocolo y latencia; no demuestra placement. No envía al leaderboard.
