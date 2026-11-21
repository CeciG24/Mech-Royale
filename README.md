# Mech Royale

Estado de esta rama: **experimento sin mejora validada; main conserva el 3.17**.
La referencia `59d32bc` obtuvo **3.17 oficial según el usuario**.
Este candidato añade pistola simultánea cuando su aim ya está alineado,
conservando el plan de láser/rayos. No tiene puntuación oficial nueva.

Se probaron seis variantes y dos candidatos en lotes de validación distintos.
La corrección de celdas ganó entrenamiento y perdió en validación; se descartó.
Disparo simultáneo en validación nueva: 2.688 frente a
2.156 de la referencia. Las cifras locales no equivalen a las de la página.

Resultados, commits, límites y reproducción: [REVIEW.txt](REVIEW.txt), [EVALUATION.json](EVALUATION.json).

Con Python y Docker (kit incluido):

```sh
python tools/retest.py --docker --check-only
```

Comprueba protocolo, latencia, arranque y fallback con el perfil del juez.
No envía al leaderboard.
