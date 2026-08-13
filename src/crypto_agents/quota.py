"""Contador de cuota por ventana deslizante.

El presupuesto es una restricción de diseño de primer orden: los modelos
disponibles varían por un factor enorme en solicitudes por ventana, y algunos
cobran doble por llamada. Esto lo hace medible.

El reloj se inyecta. Ninguna función de este módulo llama a `datetime.now()`:
de otro modo las pruebas de expiración de ventana dependerían del reloj real.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from datetime import datetime

    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import AgentRole, LLMCall

__all__ = ["Clock", "QuotaExhaustedError", "QuotaLedger"]

type Clock = Callable[[], datetime]
"""Fuente de tiempo inyectada. En producción, `lambda: datetime.now(UTC)`.

Alias perezoso (PEP 695) para que `Callable` y `datetime` puedan quedarse en el
bloque TYPE_CHECKING sin romper la importación en runtime.
"""


class QuotaExhaustedError(RuntimeError):
    """Ni el modelo primario ni el respaldo caben en la ventana actual."""

    def __init__(self, role: AgentRole, tried: tuple[str, ...]) -> None:
        self.role = role
        self.tried = tried
        super().__init__(
            f"cuota agotada para el rol {role.value}; modelos probados: {', '.join(tried)}"
        )


class QuotaLedger:
    """Consumo por (rol, modelo) dentro de una ventana deslizante.

    Vive en el `context` del grafo, no en `TradingState`: es mutable y no debe
    serializarse en cada checkpoint.
    """

    def __init__(self, settings: Settings, clock: Clock) -> None:
        self._settings = settings
        self._clock = clock
        self._entries: deque[LLMCall] = deque()

    # ── Consulta ──────────────────────────────────────────────────────────────

    def used(self, role: AgentRole, model: str) -> float:
        """Cuota consumida por ese par dentro de la ventana que termina ahora."""
        self._purge()
        return sum(
            entry.quota_weight
            for entry in self._entries
            if entry.role is role and entry.model == model
        )

    def remaining(self, role: AgentRole, choice: ModelChoice) -> float:
        """Cuota libre para ese modelo en ese rol. Puede ser negativa si se sobregiró."""
        return choice.quota_per_window - self.used(role, choice.model)

    def fits(self, role: AgentRole, choice: ModelChoice) -> bool:
        """Si una llamada más cabe entera en lo que queda de ventana."""
        return self.remaining(role, choice) >= choice.quota_weight

    def resolve(self, role: AgentRole) -> ModelChoice:
        """Modelo a usar ahora: primario, respaldo si el primario no cabe.

        Degradación de un solo salto; si el respaldo tampoco cabe, `QuotaExhaustedError`.
        """
        config = self._settings.role_config(role)
        candidates = [config.primary]
        if config.fallback is not None:
            candidates.append(config.fallback)
        for candidate in candidates:
            if self.fits(role, candidate):
                return candidate
        raise QuotaExhaustedError(role, tuple(candidate.model for candidate in candidates))

    # ── Registro ──────────────────────────────────────────────────────────────

    def record(self, call: LLMCall) -> None:
        """Anota una llamada. Un cache hit no llegó al proveedor: no consume."""
        if call.cache_hit:
            return
        self._entries.append(call)
        self._purge()

    def extend(self, calls: Iterable[LLMCall]) -> None:
        """Rehidrata el contador desde `TradingState.calls`, para replay.

        Ordena por `at` porque `_purge` asume una cola cronológica y las llamadas
        paralelas llegan al estado en orden de resolución, no de emisión.
        """
        for call in sorted(calls, key=lambda call: call.at):
            self.record(call)

    # ── Interno ───────────────────────────────────────────────────────────────

    def _purge(self) -> None:
        """Descarta lo que quedó fuera de la ventana. Las entradas llegan ordenadas."""
        cutoff = self._clock() - self._settings.quota_window
        while self._entries and self._entries[0].at <= cutoff:
            self._entries.popleft()
