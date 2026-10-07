"""Tope de gasto en dólares: lo único que frena una corrida con pago por uso.

OpenCode Zen no publica límite de peticiones, así que con `billing = payg` la cuota declarada no
frena un rol remoto (lo decide `ModelRouter`) y el contador de cuota deja de ser un freno. Lo que
queda es el saldo, y agotarlo es una corrida inválida. Este módulo es el freno que se declara antes:
un tope en USD que los sondeos, la ablación y el runner comparten.

Hay **una** clase y vive aquí. Nació dentro de `zen_probe.py`, y la ablación no puede importar el
sondeo —es el sondeo quien importa la ablación—, así que la alternativa a moverla era escribir otra:
dos guardas que suman el gasto cada uno a su manera acaban discrepando sobre cuánto se gastó.

No llama a nadie: mide lo que ya se gastó con `consumption.consume` y los precios de
`settings.py`. Por eso no importa el router.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crypto_agents.consumption import consume
from crypto_agents.quota import LOCAL_BACKENDS
from crypto_agents.settings import DEFAULT_PRICING, ConfigError
from crypto_agents.state import AgentRole, Billing

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from crypto_agents.settings import PriceTable, Settings
    from crypto_agents.state import LLMCall

__all__ = ["SpendGuard", "spend_guard", "unpriced_roles"]


class SpendGuard:
    """Tope duro de gasto: no abre una invocación nueva cuando lo gastado lo alcanza.

    No existe una cota de coste rigurosa *antes* de llamar: el repo no fija `max_tokens`, de modo
    que la salida —el razonamiento incluido— no tiene techo, y el prompt de las mesas depende de
    veredictos que aún no existen. Estimar tokens rompe la regla del repo. Lo que sí se puede es
    declarar un tope y pararse en él: el gasto es el **extremo alto** de lo que el proveedor
    declaró (`Consumption.cost_upper_usd` más la cota de las llamadas con `cached_tokens` en
    `null`), y se mira antes de cada invocación.

    Dos límites que los informes repiten: la invocación que cruza el tope termina, igual que las
    que ya estaban en vuelo al alcanzarlo, y se suman después (el exceso es, como mucho, una
    invocación de hasta dos intentos por trabajo concurrente), y las llamadas sin tokens no tienen
    cifra que sumar, así que no cuentan para el tope aunque se facturen.

    Los precios son los de pago por uso: con la suscripción no hay tope en dólares que declarar, y
    quien construye el guarda —la ablación, el runner— se niega antes.
    """

    def __init__(self, cap_usd: float) -> None:
        if cap_usd <= 0:
            raise ValueError("el tope de gasto debe ser positivo")
        self.cap_usd = cap_usd
        self.refused: list[str] = []
        self._calls: list[LLMCall] = []

    def add(self, calls: Iterable[LLMCall]) -> None:
        """Anota lo que el proveedor acaba de ver: se llama con cada intento, válido o no."""
        self._calls.extend(calls)

    @property
    def spent_usd(self) -> float:
        """Lo gastado hasta ahora, en el extremo alto."""
        total = consume(self._calls, DEFAULT_PRICING, Billing.PAYG)
        return total.cost_upper_usd + total.unreported_ceiling_usd

    @property
    def unpriced(self) -> int:
        """Llamadas respondidas sin tokens: se facturaron y no suman al gasto del tope."""
        return consume(self._calls, DEFAULT_PRICING, Billing.PAYG).no_tokens

    def allow(self, label: str) -> bool:
        """Si cabe abrir otra invocación; si no, deja anotado qué se dejó sin hacer."""
        if self.spent_usd < self.cap_usd:
            return True
        self.refused.append(label)
        return False


def unpriced_roles(
    settings: Settings,
    at: datetime,
    roles: Iterable[AgentRole] | None = None,
    prices: PriceTable = DEFAULT_PRICING,
) -> tuple[tuple[AgentRole, str], ...]:
    """Los roles remotos cuyo modelo no tiene precio de pago por uso: el tope no los vería.

    Una llamada de un modelo sin fila de precio no suma nada al gasto (`CostGap.NO_PRICE`), así que
    un tope declarado sobre ese mapa de roles dejaría pasar todo lo que ese rol gastara. Quien
    arranca con tope pregunta esto antes y se niega nombrando rol e id. `roles` son los que la
    corrida va a usar —todos, si no se dice—; un primario local no gasta y no se mira.
    """
    missing: list[tuple[AgentRole, str]] = []
    for role in AgentRole if roles is None else roles:
        config = settings.roles.get(role)
        if config is None or config.primary.backend in LOCAL_BACKENDS:
            continue
        if prices.price_for(config.primary.model, Billing.PAYG, at) is None:
            missing.append((role, config.primary.model))
    return tuple(missing)


def spend_guard(
    settings: Settings,
    cap_usd: float | None,
    missing: Iterable[tuple[AgentRole, str]],
) -> SpendGuard | None:
    """El guarda de una corrida que puede gastar, o `None` si no se declaró tope.

    Es la única puerta por la que la ablación y el runner lo construyen, para que las dos
    negativas no se puedan olvidar en uno de los dos sitios:

    - **Solo con pago por uso.** Con la suscripción lo que frena es la cuota de la ventana y
      nada cambia; un tope en dólares ahí mediría el gasto con los precios de otra forma de pago.
    - **Ningún rol remoto sin precio.** El tope no vería lo que ese rol gastara, y un tope que
      deja pasar un rol entero no es un tope.

    `missing` lo trae quien llama, de `unpriced_roles`, y no tiene valor por defecto: la ablación
    lo calcula brazo a brazo —cada uno con su mapa de roles y solo los roles que usa— y el runner
    sobre el mapa entero. Con un valor por defecto, olvidarlo sería construir un tope ciego.
    """
    if cap_usd is None:
        return None
    if settings.billing is not Billing.PAYG:
        raise ConfigError(
            f"el tope en USD es del pago por uso: CA_BILLING={settings.billing.value}, y con la "
            "suscripción lo que frena es la cuota de la ventana"
        )
    if cap_usd <= 0:
        raise ConfigError("el tope de gasto debe ser positivo")
    unpriced = sorted(set(missing), key=lambda pair: (pair[0].value, pair[1]))
    if unpriced:
        detail = ", ".join(f"{role.value} → {model}" for role, model in unpriced)
        raise ConfigError(
            f"sin precio de pago por uso para {detail}: el tope no vería lo que ese rol gaste. "
            "Falta la fila en settings.py o el id no es de este proveedor"
        )
    return SpendGuard(cap_usd)
