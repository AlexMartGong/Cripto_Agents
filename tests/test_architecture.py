"""Pruebas de las reglas permanentes del sistema.

Estas reglas se cumplían hoy por costumbre, no por construcción. Aquí quedan
impuestas: si alguien las rompe, falla el ciclo de verificación en vez de
descubrirse en producción cuando un modelo devuelva un RSI inventado o una orden
salte el gate de riesgo.

Cada test corresponde a una regla permanente y la cita en su docstring.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import get_args, get_type_hints

import annotated_types
import pytest

import crypto_agents.graph  # importa el paquete entero: registra las subclases
import crypto_agents.journal
from crypto_agents import ablation, execution
from crypto_agents.market import CcxtMarketClient, CcxtTradingClient
from crypto_agents.quota import QuotaLedger
from crypto_agents.state import Claim, LLMCall, LLMOutput, Observation

SOURCE_DIR = Path(crypto_agents.graph.__file__).parent
PROVIDER_MODULES = {"langchain_openai", "langchain_core", "ollama"}
ROUTER_MODULE = "llm.py"

UNBOUNDED_NUMERIC_ALLOWLIST = {
    ("Proposal", "invalidation_price"),
    ("Decision", "invalidation_price"),
}
"""Único número sin acotar que un modelo puede emitir.

Un precio de invalidación es un juicio sobre dónde se rompe la tesis, no una
medición del mercado. Cualquier añadido a esta lista tiene que defenderse igual.

Las dos entradas son el mismo campo: `Decision` hereda de `Proposal`, y el test
recorre `model_fields`, que incluye los heredados. No es una excepción nueva, es
la misma vista desde las dos clases. Que haya que escribirla dos veces es
deliberado: si alguien mueve el campo a una tercera subclase, esta lista se lo
recuerda.
"""


def source_files() -> list[Path]:
    """Módulos del paquete."""
    return sorted(SOURCE_DIR.rglob("*.py"))


def imported_roots(path: Path) -> set[str]:
    """Módulos raíz importados, incluidos los importados dentro de funciones."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def llm_output_models() -> list[type[LLMOutput]]:
    """Todas las clases que un modelo puede llegar a producir."""

    def walk(cls: type[LLMOutput]) -> list[type[LLMOutput]]:
        found: list[type[LLMOutput]] = []
        for subclass in cls.__subclasses__():
            found.append(subclass)
            found.extend(walk(subclass))
        return found

    return walk(LLMOutput)


# ───────────────────── Regla 1: un LLM nunca calcula un número ────────────────────────


def test_llm_outputs_declare_no_unbounded_numbers() -> None:
    """Un modelo no puede devolver una magnitud calculada.

    Todo campo numérico de una salida de LLM está acotado en [0,1]: eso lo hace
    un juicio (confianza, convicción, fracción de tamaño) y no una medición. Un
    float sin acotar sería un número que el modelo tuvo que calcular, y si un
    modelo devuelve un RSI es alucinación.
    """
    offenders: list[str] = []
    for model in llm_output_models():
        for name, field in model.model_fields.items():
            if not _is_numeric(field.annotation):
                continue
            if (model.__name__, name) in UNBOUNDED_NUMERIC_ALLOWLIST:
                continue
            if not _is_bounded_unit_interval(field.metadata):
                offenders.append(f"{model.__name__}.{name}")
    assert offenders == [], (
        f"campos numéricos sin acotar en salidas de LLM: {', '.join(offenders)}. "
        "Un número que el modelo tiene que calcular es una alucinación esperando ocurrir."
    )


def test_technical_verdicts_reference_indicators_only_by_name() -> None:
    """El agente técnico cita indicadores; no los reproduce.

    `cites` son nombres, no valores: el agente no tiene forma de devolver un
    número de indicador ni de contradecir el que se calculó.
    """
    assert Observation.model_fields["cites"].annotation == list[str]


def test_the_package_computes_indicators_outside_the_llm_path() -> None:
    """pandas-ta solo se usa en el módulo de indicadores."""
    users = {
        path.name
        for path in source_files()
        if "pandas_ta" in imported_roots(path) or "pandas" in imported_roots(path)
    }
    assert users <= {"indicators.py", "market.py", "activation.py"}


# ────────────── Regla 2: un LLM nunca es lo último antes de una orden ─────────────────


def test_build_order_never_reads_the_decision_size() -> None:
    """El decisor propone; el gate de riesgo dispone.

    Se inspecciona el AST y no el texto: un `decision.size_fraction` dentro de un
    docstring explicando la regla no debe hacer fallar el test, y uno dentro del
    código sí.
    """
    tree = ast.parse(inspect.getsource(execution.build_order))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "decision"
        and node.attr == "size_fraction"
    ]
    assert reads == [], "build_order lee el tamaño propuesto por el modelo"


def test_order_size_can_only_come_from_the_risk_verdict() -> None:
    """El único atributo de tamaño que `build_order` lee es el del veredicto."""
    tree = ast.parse(inspect.getsource(execution.build_order))
    size_reads = {
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and "size" in node.attr
    }
    assert size_reads == {"verdict.final_size_fraction"}


# ────────── Regla 3: toda afirmación de un LLM cita evidencia por id ─────────────────


def test_claims_cannot_exist_without_grounding() -> None:
    """Sin `grounded_in` el debate deriva a opinión libre en pocas iteraciones."""
    assert _min_length(Claim.model_fields["grounded_in"].metadata) >= 1
    assert _min_length(Observation.model_fields["cites"].metadata) >= 1


# ────────── Regla 4: toda llamada queda registrada con su digest de prompt ───────────


def test_a_call_cannot_be_recorded_without_its_prompt_digest() -> None:
    """Sin digest no hay backtesting ni auditoría de por qué se compró algo."""
    field = LLMCall.model_fields["prompt_digest"]
    assert field.is_required()
    patterns = [item for item in field.metadata if getattr(item, "pattern", None) is not None]
    assert patterns, "prompt_digest debe estar restringido a un digest hexadecimal"


PROVIDER_CLIENTS = {"ChatOpenAI", "AsyncOpenAI", "AsyncClient"}
"""Clientes concretos que el router construye. Los tres salen a la red."""

RETRYING_CLIENTS = {"ChatOpenAI", "AsyncOpenAI"}
"""Los que reintentan por su cuenta si no se les dice que no."""


def provider_client_calls() -> list[ast.Call]:
    """Cada construcción de un cliente de proveedor dentro del router."""
    tree = ast.parse((SOURCE_DIR / ROUTER_MODULE).read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PROVIDER_CLIENTS
    ]


def test_no_provider_client_is_built_without_an_explicit_timeout() -> None:
    """Un cliente sin timeout hereda el del SDK, y los dos heredados son inservibles.

    El de OpenAI son 600 s de lectura contra una evaluación que cuesta ~90 s
    completa. El de Ollama no existe: `Timeout(None)`, es decir, esperar
    indefinidamente a un servidor local que se colgó. Ninguno de los dos es una
    decisión que nadie de este repositorio haya tomado, y esa es la objeción.
    """
    offenders = [
        node.func.id  # type: ignore[attr-defined]
        for node in provider_client_calls()
        if "timeout" not in {keyword.arg for keyword in node.keywords}
    ]
    assert offenders == [], f"clientes construidos sin timeout explícito: {offenders}"


def test_no_provider_client_retries_behind_the_router() -> None:
    """Regla 4: un reintento que el router no ve es una llamada que el journal no tiene.

    `max_retries` por defecto es 2 en el SDK de OpenAI, así que un `complete()`
    puede ser tres peticiones facturadas y un solo `LLMCall`. Reintentar es del
    router, que adjunta el error al prompt y escribe una fila por intento.
    """
    offenders = [
        node.func.id  # type: ignore[attr-defined]
        for node in provider_client_calls()
        if node.func.id in RETRYING_CLIENTS  # type: ignore[attr-defined]
        and "max_retries" not in {keyword.arg for keyword in node.keywords}
    ]
    assert offenders == [], f"clientes que reintentan sin que el router lo sepa: {offenders}"


def test_only_the_router_module_talks_to_a_provider() -> None:
    """Todas las llamadas pasan por el router, que es quien las registra.

    Si un nodo pudiera instanciar un cliente por su cuenta, gastaría cuota sin
    dejar `LLMCall`, y el registro dejaría de ser completo.
    """
    offenders = {
        path.name
        for path in source_files()
        if path.name != ROUTER_MODULE and imported_roots(path) & PROVIDER_MODULES
    }
    assert offenders == set(), f"módulos que hablan con un proveedor: {sorted(offenders)}"


# ────────────────── El origen de datos no puede autenticarse ─────────────────────────


def test_the_market_client_has_no_way_to_receive_credentials() -> None:
    """Leer mercado y operar son dos papeles, y solo uno quiere claves.

    La garantía es la firma, no la disciplina: `CcxtMarketClient` recibe un
    `exchange_id` y no un `ExchangeSettings`, así que no existe parámetro por el
    que una credencial pueda entrar ni `sandbox` que aplicar. Ampliarla para
    aceptar la configuración —y prometer ignorar tres de sus campos— es
    exactamente la regresión que este test tiene que ver.

    Las dos consecuencias que lo motivan están medidas: con las claves puestas
    ccxt firma también los endpoints públicos y binance responde `-2008 Invalid
    Api-Key ID`; y con `sandbox=true` la fuente pasaba a ser testnet, que da 58
    velas de 4h contra un preset que exige 400.
    """
    parameters = inspect.signature(CcxtMarketClient.__init__).parameters
    assert list(parameters) == ["self", "exchange_id"]
    assert get_type_hints(CcxtMarketClient.__init__)["exchange_id"] is str


def test_the_activation_sweep_cannot_call_a_model() -> None:
    """El barrido del gate se repite cuantas veces haga falta porque es gratis.

    Un import del router bastaría para que una versión futura metiera una llamada
    «solo para comprobar algo» y la tabla pasara a costar dinero por barrido: son
    153 000 velas. Que no pueda es lo que la hace repetible.
    """
    imports = {
        node.module
        for node in ast.walk(ast.parse((SOURCE_DIR / "activation_sweep.py").read_text("utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "crypto_agents.llm" not in imports
    assert "crypto_agents.graph" not in imports


@pytest.mark.parametrize("module", ["baselines.py", "stops.py", "dispersion.py"])
def test_a_baseline_cannot_call_a_model(module: str) -> None:
    """Una línea base cuesta cero porque no hay por dónde llamar, no porque nadie lo haga.

    Es lo que permite correrlas en modo solo-caché sin mirar la factura. Un import del
    router, del grafo o del replay bastaría para que una versión futura metiera una
    llamada «solo para comparar» y el brazo dejara de ser una línea base. `dispersion.py` entra
    por lo mismo: mide las líneas base, y no puede tener a mano ni un proveedor ni la caché.
    """
    imports = {
        node.module
        for node in ast.walk(ast.parse((SOURCE_DIR / module).read_text("utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not imports & {
        "crypto_agents.llm",
        "crypto_agents.graph",
        "crypto_agents.nodes",
        "crypto_agents.replay",
        "crypto_agents.quota",
        "crypto_agents.cache",
    }


def test_only_the_stops_module_knows_the_common_multiple() -> None:
    """Un solo lugar fabrica el stop común: nadie más escribe `2 * atr` por su cuenta.

    Las líneas base lo declaran y la puntuación lo reconstruye. Si cualquiera de las dos
    citara el múltiplo, tendrían dos fórmulas y la comparación entre un brazo y una línea
    base mediría la diferencia entre ellas.
    """
    offenders = []
    for path in source_files():
        if path.name == "stops.py":
            continue
        tree = ast.parse(path.read_text("utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        if "COMMON_STOP_ATR_MULTIPLE" in names:
            offenders.append(path.name)
    assert offenders == []


def test_building_a_selection_cannot_call_a_model() -> None:
    """Elegir qué velas se evalúan tiene que ser gratis y repetible.

    `selection.py` confirma cada candidato pasándolo por el gate con la ventana
    del replay, que son ~4 400 velas por símbolo. Un import del router —o de
    `replay.py`, que lo trae— bastaría para que una versión futura metiera ahí una
    llamada «solo para comprobar algo».
    """
    imports = {
        node.module
        for node in ast.walk(ast.parse((SOURCE_DIR / "selection.py").read_text("utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "crypto_agents.llm" not in imports
    assert "crypto_agents.replay" not in imports
    assert "crypto_agents.nodes" not in imports


def test_no_arm_of_the_ablation_can_own_its_quota_ledger() -> None:
    """Un contador por brazo son seis presupuestos donde el proveedor ve uno.

    El defecto ya se corrigió una vez entre símbolos del runner y volvió por otra
    puerta: la construcción dentro de `run_arm`. Se cierran las dos que quedan —el
    contador entra por la firma y no se puede fabricar dentro—, porque el síntoma
    solo aparece con el histórico grande, a mitad de una corrida de horas.
    """
    parameters = inspect.signature(ablation.run_arm).parameters
    assert "ledger" in parameters, "run_arm ya no recibe el contador compartido"
    assert parameters["ledger"].default is inspect.Parameter.empty, (
        "con valor por defecto, olvidarlo vuelve a ser silencioso"
    )

    tree = ast.parse(inspect.getsource(ablation.run_arm))
    built = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QuotaLedger"
    ]
    assert built == [], "run_arm construye su propio contador"


def test_the_ablation_has_no_way_to_keep_its_journal_in_memory() -> None:
    """Regla 5 aplicada a la ablación: una corrida que no deja registro no se puede probar.

    La primera tabla completa salió de journals en memoria. `run_arm` tenía uno por
    defecto y el comando no pasaba otro, así que 840 evaluaciones dejaron una tabla
    agregada y ninguna forma de preguntar por qué 31 no decidieron. Se cierran las
    tres puertas: el journal entra por la firma sin valor por defecto, el módulo no
    nombra el journal en memoria, y el contexto que `run_arm` construye lo recibe
    explícito —`AgentContext` trae uno en memoria por defecto, así que omitirlo
    volvería a ser silencioso.
    """
    parameters = inspect.signature(ablation.run_arm).parameters
    assert "journal" in parameters, "run_arm ya no recibe el journal"
    assert parameters["journal"].default is inspect.Parameter.empty, (
        "con valor por defecto, olvidarlo vuelve a ser silencioso"
    )

    module = ast.parse((SOURCE_DIR / "ablation.py").read_text(encoding="utf-8"))
    named = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "InMemoryJournal" not in named | imported, "la ablación puede volver a no persistir"

    contexts = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(ablation.run_arm)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AgentContext"
    ]
    assert contexts, "run_arm ya no construye el contexto: revisar dónde entra el journal"
    for call in contexts:
        assert "journal" in {keyword.arg for keyword in call.keywords}, (
            "un AgentContext sin journal usa el de memoria que trae por defecto"
        )


@pytest.mark.parametrize("module", ["metrics.py", "audit.py"])
def test_auditing_a_run_cannot_call_a_model(module: str) -> None:
    """Leer lo que pasó tiene que ser gratis, o nadie lo hará dos veces.

    Ni el router ni nada que lo traiga: `replay.py`, `graph.py` y `nodes.py` lo
    importan, y `ablation.py` además construye los backends. Un import de
    cualquiera bastaría para que una versión futura «solo comprobara algo» contra
    un proveedor desde el comando que se supone que solo lee archivos.
    """
    imports = {
        node.module
        for node in ast.walk(ast.parse((SOURCE_DIR / module).read_text("utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {
        "crypto_agents.llm",
        "crypto_agents.replay",
        "crypto_agents.graph",
        "crypto_agents.nodes",
        "crypto_agents.ablation",
    }
    assert not imports & forbidden


def test_only_the_quota_ledger_caller_knows_which_models_a_role_may_use() -> None:
    """El contador no tiene `Settings`, y esa ausencia es lo que permite compartirlo.

    Con la configuración dentro, un contador compartido resolvería los roles con el
    mapa de otro brazo: el que pide un rol en local recibiría el modelo remoto y
    dejaría de medir lo que dice medir, sin que nada fallara.
    """
    parameters = inspect.signature(QuotaLedger.__init__).parameters
    assert list(parameters) == ["self", "window", "clock"]
    assert QuotaLedger.__init__.__annotations__["window"] == "timedelta"
    assert "choices" in inspect.signature(QuotaLedger.resolve).parameters

    names = {
        node.id
        for node in ast.walk(ast.parse((SOURCE_DIR / "quota.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.Name)
    }
    assert "Settings" not in names, "el contador volvió a mirar la configuración"


def test_only_the_trading_client_knows_how_to_authenticate() -> None:
    """`verify_credentials` vive donde viven las claves, y no en el lector."""
    assert hasattr(CcxtTradingClient, "verify_credentials")
    assert not hasattr(CcxtMarketClient, "verify_credentials")


# ──────────────────────────── Capas del paquete ──────────────────────────────────────


def test_the_contract_is_the_root_of_the_dependency_graph() -> None:
    """`state.py` no importa nada del paquete: todo lo demás depende de él."""
    assert "crypto_agents" not in imported_roots(SOURCE_DIR / "state.py")


def test_the_risk_gate_depends_on_no_model_machinery() -> None:
    """El gate de riesgo es código determinista: no conoce routers ni backends."""
    imports = imported_roots(SOURCE_DIR / "risk.py")
    assert imports & PROVIDER_MODULES == set()
    tree = ast.parse((SOURCE_DIR / "risk.py").read_text(encoding="utf-8"))
    modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "crypto_agents.llm" not in modules


# ────────────────────────────────── Auxiliares ───────────────────────────────────────


def _is_numeric(annotation: object) -> bool:
    """Si la anotación admite un número, incluidas las opcionales y anotadas."""
    if annotation in (int, float):
        return True
    return any(_is_numeric(arg) for arg in get_args(annotation))


def _is_bounded_unit_interval(metadata: list[object]) -> bool:
    """Si el campo está restringido a [0,1], que es lo que hace de él un juicio."""
    has_floor = any(
        isinstance(item, annotated_types.Ge) and float(item.ge) >= 0.0  # type: ignore[arg-type]
        for item in metadata
    )
    has_ceiling = any(
        isinstance(item, annotated_types.Le) and float(item.le) <= 1.0  # type: ignore[arg-type]
        for item in metadata
    )
    return has_floor and has_ceiling


def _min_length(metadata: list[object]) -> int:
    """Longitud mínima declarada, o 0 si no hay ninguna."""
    for item in metadata:
        if isinstance(item, annotated_types.MinLen):
            return item.min_length
    return 0
