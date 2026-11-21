# Mech Royale

Candidato de práctica basado en `808e115286c5dfdde8dcca00dddc74e8af92e22f`.
El usuario reporta 3.17 en la plataforma para esa referencia; este paquete conserva
su estrategia. El experimental anterior obtuvo 2.37. Cinco variantes nuevas no
superaron la referencia en 24 semillas locales y se descartaron.

Resultados y límites: [REVIEW.txt](REVIEW.txt), [EVALUATION.json](EVALUATION.json).
Las puntuaciones locales no equivalen a las de las 48 semillas oficiales privadas.

Con Python y el kit oficial disponible en `mechfight-kit/`:

```sh
python tools/retest.py --docker --check-only
python tools/retest.py --docker
```

El primer comando comprueba arranque, fallback y protocolo con los límites del juez.
El segundo añade la evaluación local de práctica. No envía al leaderboard.
El checker mide protocolo y latencia; no demuestra mejor placement.
