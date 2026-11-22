# Mech Royale

Referencia conservada: **3.33 oficial reportado por el usuario**, política de
`c0e261dde754a55c51ca5ed63a2e9339cfe2802f`.

Se probaron seis cambios adicionales. Ninguno superó las comprobaciones de
placement necesarias para sustituir esa estrategia. Este commit actualiza
informes y evaluación; el comportamiento del bot conserva la versión de 3.33.

Resultados y reproducción: [REVIEW.txt](REVIEW.txt), [EVALUATION.json](EVALUATION.json).
Las puntuaciones locales no equivalen a las de las semillas privadas oficiales.

```sh
python tools/retest.py --docker --check-only
```

Con Python, Docker y el kit incluido. Comprueba protocolo y latencia; no entrega al leaderboard.
