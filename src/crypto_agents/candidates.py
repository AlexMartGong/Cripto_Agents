"""Qué modelos de Zen se han propuesto para un rol que todavía no tiene modelo elegido.

Son listas, no decisiones: decir quién *podría* llevar `structure`, `volume` o `bull` es de quien
encarga la medición. Viven aparte del sondeo que los mide (`zen_probe.py`) porque quien presupuesta
la corrida (`estimate.py`) necesita la misma lista y no puede importar un módulo que llama a
modelos: con la lista dentro del sondeo, el estimador leía los candidatos de los archivos que
encontraba en un directorio, y un candidato ya descartado seguía teniendo su línea.

No importa nada del paquete: es una tabla de nombres.
"""

from __future__ import annotations

from typing import Final, NamedTuple

__all__ = ["BULL_CANDIDATES", "CANDIDATES", "Candidate"]


class Candidate(NamedTuple):
    """Un modelo de pago de Zen, servido por `/chat/completions`, que podría llevar un rol."""

    model: str
    family: str
    """Declarada a mano, como `ModelChoice.family`: deducirla del nombre es lo que se evita."""


CANDIDATES: Final = (
    Candidate("deepseek-v4.1-flash", "deepseek"),
    Candidate("deepseek-v4-pro", "deepseek"),
    Candidate("glm-5.3-flash", "zhipu"),
    Candidate("minimax-m2.7", "minimax"),
    Candidate("kimi-k2.7-code", "moonshot"),
    Candidate("qwen3.8-max", "qwen"),
)
"""Los candidatos a `structure` y `volume`."""

BULL_CANDIDATES: Final = (
    Candidate("kimi-k3", "moonshot"),
    Candidate("qwen3.8-max", "qwen"),
)
"""Los modelos que podrían llevar `bull`. Familia declarada a mano, como en `ModelChoice`.

`kimi-k2.6`, el que `.env` declara hoy, no está: Zen lo lista y contesta 410 de forma sostenida, y
sondearlo otra vez es pagar la misma respuesta. Vuelve si soporte dice que no fue retirado.

`deepseek-v4-pro` salió en el bloque T5: es el modelo de `momentum`, y un candidato a bull no
puede ser el productor de la evidencia sobre la que argumenta. En `20261005T233053Z` lo fue en 12
de 12 activaciones."""
