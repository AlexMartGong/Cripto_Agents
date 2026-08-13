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
from typing import get_args

import annotated_types

import crypto_agents.graph  # importa el paquete entero: registra las subclases
import crypto_agents.journal
from crypto_agents import execution
from crypto_agents.state import Claim, LLMCall, LLMOutput, Observation

SOURCE_DIR = Path(crypto_agents.graph.__file__).parent
PROVIDER_MODULES = {"langchain_openai", "langchain_core", "ollama"}
ROUTER_MODULE = "llm.py"

UNBOUNDED_NUMERIC_ALLOWLIST = {
    ("Decision", "invalidation_price"),
}
"""Único número sin acotar que un modelo puede emitir.

Un precio de invalidación es un juicio sobre dónde se rompe la tesis, no una
medición del mercado. Cualquier añadido a esta lista tiene que defenderse igual.
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
