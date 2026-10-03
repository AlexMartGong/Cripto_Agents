"""Pruebas de configuración.

El criterio central: si falta una variable requerida, el arranque falla con un
mensaje que la nombra. Un `KeyError` a mitad de una evaluación no cuenta.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import crypto_agents.settings
from crypto_agents.ablation import ARMS
from crypto_agents.settings import (
    DEFAULT_ENV_FILE,
    DEFAULT_PRICING,
    Backend,
    ConfigError,
    ExchangeSettings,
    ModelChoice,
    PriceRow,
    PriceTable,
    RoleConfig,
    load_settings,
)
from crypto_agents.state import AgentRole, Billing, StructuredOutputMode
from tests.conftest import CHEAP, role_map

EXPENSIVE = ModelChoice(
    backend=Backend.OPENAI,
    model="gpt-x",
    family="gpt",
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    quota_weight=2.0,
    quota_per_window=120,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla del entorno real del desarrollador; el `.env` se desactiva por kwarg."""
    for key in list(os.environ):
        if key.startswith("CA_"):
            monkeypatch.delenv(key, raising=False)


def base_kwargs(**overrides: object) -> dict[str, object]:
    """Configuración mínima válida; usa Ollama para no exigir clave de OpenAI."""
    kwargs: dict[str, object] = {
        "roles": role_map(),
        "ollama": {"host": "http://localhost:11434"},
    }
    kwargs.update(overrides)
    return kwargs


# ─────────────────────────────────────────── Casos válidos ────────────────────────────────────────


def test_minimal_configuration_loads() -> None:
    """Ventana por defecto de 5 horas y todos los roles mapeados."""
    settings = load_settings(**base_kwargs())
    assert settings.quota_window == timedelta(hours=5)
    assert set(settings.roles) == set(AgentRole)


def test_choices_includes_primaries_and_fallbacks() -> None:
    """El router necesita ver todos los modelos declarados, no solo los primarios.

    Uno por rol más un respaldo para cada rol salvo el decisor, que no admite.
    """
    settings = load_settings(**base_kwargs(roles=role_map(primary=CHEAP, fallback=CHEAP)))
    assert len(settings.choices()) == 2 * len(AgentRole) - 1


def test_settings_are_frozen() -> None:
    """La configuración no cambia a mitad de una ejecución."""
    settings = load_settings(**base_kwargs())
    with pytest.raises(ValidationError):
        settings.quota_window = timedelta(hours=1)


def test_exchange_without_credentials_is_valid() -> None:
    """Leer OHLCV no necesita claves."""
    assert load_settings(**base_kwargs()).exchange == ExchangeSettings()


def test_roles_load_from_nested_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """El reparto rol → modelo es un dato del entorno, no una constante del código."""
    for role in AgentRole:
        prefix = f"CA_ROLES__{role.value.upper()}__PRIMARY__"
        monkeypatch.setenv(f"{prefix}BACKEND", "ollama")
        monkeypatch.setenv(f"{prefix}MODEL", "qwen3:8b")
        monkeypatch.setenv(f"{prefix}FAMILY", "qwen" if role is not AgentRole.BEAR else "llama")
        monkeypatch.setenv(f"{prefix}STRUCTURED_OUTPUT", "json_schema")
        monkeypatch.setenv(f"{prefix}QUOTA_PER_WINDOW", "63000")
    monkeypatch.setenv("CA_OLLAMA__HOST", "http://localhost:11434")

    settings = load_settings()
    assert settings.role_config(AgentRole.DECIDER).primary.model == "qwen3:8b"


def test_quota_weight_is_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los modelos de doble consumo se declaran como dato, con peso 2.0."""
    for role in AgentRole:
        prefix = f"CA_ROLES__{role.value.upper()}__PRIMARY__"
        monkeypatch.setenv(f"{prefix}BACKEND", "openai")
        monkeypatch.setenv(f"{prefix}MODEL", "gpt-x")
        monkeypatch.setenv(f"{prefix}FAMILY", "gpt" if role is not AgentRole.BEAR else "claude")
        monkeypatch.setenv(f"{prefix}STRUCTURED_OUTPUT", "function_calling")
        monkeypatch.setenv(f"{prefix}QUOTA_WEIGHT", "2.0")
        monkeypatch.setenv(f"{prefix}QUOTA_PER_WINDOW", "120")
    monkeypatch.setenv("CA_OPENAI__API_KEY", "sk-test")

    settings = load_settings()
    assert settings.role_config(AgentRole.BULL).primary.quota_weight == 2.0


# ─────────────────────────────────────────── Archivo .env ─────────────────────────────────────────


def _write_env_file(directory: Path) -> Path:
    """Un `.env` mínimo y válido, con los seis roles en local."""
    roles = {
        role.value: {
            "primary": {
                "backend": "ollama",
                "model": "qwen3:8b",
                "family": "qwen" if role is not AgentRole.BEAR else "llama",
                "structured_output": "json_schema",
                "quota_per_window": 63000,
            }
        }
        for role in AgentRole
    }
    path = directory / ".env"
    path.write_text(
        f"CA_ROLES={json.dumps(roles)}\nCA_OLLAMA__HOST=http://localhost:11434\n",
        encoding="utf-8",
    )
    return path


def test_a_env_file_in_the_working_directory_is_not_read_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Estar parado junto a un `.env` no es pedirlo.

    Cuando `Settings` lo declaraba por defecto, cualquier llamada a
    `load_settings()` leía el archivo del desarrollador: la suite entera pasaba a
    depender de en qué máquina corría, y en la que opera —la única que tiene
    `.env`— dos pruebas de la CLI fallaban, una de ellas colgándose hasta el
    siguiente cierre de vela.
    """
    _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match=r"falta CA_ROLES"):
        load_settings()


def test_the_env_file_loads_when_it_is_asked_for_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Y con la ruta explícita sí se lee, que es como lo piden los dos comandos."""
    path = _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)

    settings = load_settings(path)
    assert settings.role_config(AgentRole.DECIDER).primary.model == "qwen3:8b"
    assert load_settings(DEFAULT_ENV_FILE).roles == settings.roles


# ────────────────────────────────────────── Fallos de arranque ────────────────────────────────────


def test_missing_roles_names_the_variable() -> None:
    """Sin mapa de roles, el mensaje nombra la variable que falta."""
    with pytest.raises(ConfigError, match=r"falta CA_ROLES"):
        load_settings(ollama={"host": "http://x"})


def test_incomplete_role_map_names_the_missing_roles() -> None:
    """Un rol sin modelo revienta al arrancar, no a mitad de una evaluación."""
    partial = role_map()
    del partial[AgentRole.DECIDER]
    del partial[AgentRole.BULL]
    with pytest.raises(ConfigError, match="bull, decider"):
        load_settings(**base_kwargs(roles=partial))


def test_openai_backend_without_credentials_fails_at_startup() -> None:
    """Usar un backend sin credenciales debe fallar antes de la primera llamada."""
    with pytest.raises(ConfigError, match=r"CA_OPENAI__API_KEY"):
        load_settings(roles=role_map(primary=EXPENSIVE))


def test_fallback_backend_also_requires_credentials() -> None:
    """El respaldo se usa de verdad: sus credenciales cuentan igual que las del primario."""
    with pytest.raises(ConfigError, match=r"CA_OPENAI__API_KEY"):
        load_settings(**base_kwargs(roles=role_map(primary=CHEAP, fallback=EXPENSIVE)))


def test_ollama_only_accepts_the_mode_it_actually_implements() -> None:
    """Declarar `function_calling` sobre el respaldo local sería una configuración que miente.

    `OllamaBackend` restringe la generación pasando el esquema en `format`, que es
    `json_schema`, y no expone herramientas. Aceptar otra cosa y luego ignorarla
    dejaría al journal registrando un modo que nunca se pidió.
    """
    with pytest.raises(ValidationError, match="ollama solo admite"):
        ModelChoice(
            backend=Backend.OLLAMA,
            model="qwen3:8b",
            family="qwen",
            structured_output=StructuredOutputMode.FUNCTION_CALLING,
            quota_per_window=100,
        )


def test_a_model_without_a_declared_mode_does_not_load() -> None:
    """El modo no tiene default: un valor por omisión sería la constante implícita de vuelta.

    Y el fallo nombra la variable, como el resto de la configuración que falta.
    """
    roles = {
        role.value: {
            "primary": {
                "backend": "ollama",
                "model": "qwen3:8b",
                "family": "qwen" if role is not AgentRole.BEAR else "llama",
                "quota_per_window": 100,
            }
        }
        for role in AgentRole
    }
    with pytest.raises(ConfigError, match="STRUCTURED_OUTPUT"):
        load_settings(roles=roles, ollama={"host": "http://localhost:11434"})


def test_the_decider_cannot_declare_a_fallback() -> None:
    """Degradar al decisor cambiaría quién decide sin que quede dicho en ninguna parte.

    Los demás roles sí degradan: una lectura técnica más pobre sigue siendo una
    lectura y el journal registra con qué backend se produjo. La decisión final no
    admite ese trato, así que la configuración que lo intente no llega a cargar.
    """
    roles = role_map()
    roles[AgentRole.DECIDER] = RoleConfig(primary=CHEAP, fallback=CHEAP)
    with pytest.raises(ConfigError, match="decider no admite fallback"):
        load_settings(**base_kwargs(roles=roles))


def test_debate_desks_sharing_a_family_are_rejected() -> None:
    """Con la misma familia en ambas mesas, sus errores están correlacionados.

    Dos mesas sobre el mismo modelo pasan por alto lo mismo: el decisor recibiría
    dos versiones del mismo sesgo creyendo que son puntos de vista independientes.
    """
    roles = role_map()
    roles[AgentRole.BEAR] = RoleConfig(primary=CHEAP)
    with pytest.raises(ConfigError, match="comparten familia: qwen"):
        load_settings(**base_kwargs(roles=roles))


def test_debate_desks_sharing_a_family_through_a_fallback_are_rejected() -> None:
    """El respaldo cuenta: la correlación aparece igual cuando una mesa degrada."""
    roles = role_map()
    roles[AgentRole.BULL] = RoleConfig(
        primary=CHEAP.model_copy(update={"family": "gpt"}), fallback=CHEAP
    )
    roles[AgentRole.BEAR] = RoleConfig(
        primary=CHEAP.model_copy(update={"family": "claude"}), fallback=CHEAP
    )
    with pytest.raises(ConfigError, match="comparten familia: qwen"):
        load_settings(**base_kwargs(roles=roles))


def test_debate_desks_in_different_families_are_accepted() -> None:
    """Familias distintas en ambas mesas es la configuración válida."""
    settings = load_settings(**base_kwargs())
    bull = {choice.family for choice in settings.role_choices(AgentRole.BULL)}
    bear = {choice.family for choice in settings.role_choices(AgentRole.BEAR)}
    assert not bull & bear


def test_half_exchange_credentials_are_rejected() -> None:
    """Media credencial es configuración a medio hacer, no modo de solo lectura."""
    with pytest.raises(ConfigError, match="juntas o ninguna"):
        load_settings(**base_kwargs(exchange={"api_key": "solo-la-clave"}))


def test_unknown_field_is_rejected() -> None:
    """`extra="forbid"`: una variable mal escrita no se ignora en silencio."""
    with pytest.raises(ConfigError):
        load_settings(**base_kwargs(qouta_window=timedelta(hours=1)))


def test_zero_quota_weight_is_rejected() -> None:
    """Un peso de cero haría que un modelo pareciera gratis."""
    raw = {
        role.value: {
            "primary": {
                "backend": "ollama",
                "model": "qwen3:8b",
                "family": "qwen" if role is not AgentRole.BEAR else "llama",
                "quota_weight": 0.0,
                "quota_per_window": 10,
            }
        }
        for role in AgentRole
    }
    with pytest.raises(ConfigError, match="QUOTA_WEIGHT"):
        load_settings(**base_kwargs(roles=raw))


# ─────────────────────────── La plantilla que se reparte ──────────────────────────────────────────


TEMPLATE = Path(crypto_agents.settings.__file__).parents[2] / ".env.example"


def test_the_shipped_template_declares_a_fallback_for_every_local_arm() -> None:
    """Un brazo que pide un rol en local necesita que la plantilla lo declare.

    `local_bull` existía como brazo mientras `bull` no declaraba respaldo, así que
    la ablación moría en `arm_settings()` — pero solo al llegar al sexto brazo,
    con cinco ya pagados. La plantilla es el único sitio donde eso se puede ver
    antes de gastar nada.
    """
    settings = load_settings(TEMPLATE)
    wanted = {role for arm in ARMS for role in arm.local_roles}

    assert wanted, "ningún brazo pide roles en local: esta prueba dejó de medir algo"
    for role in sorted(wanted):
        assert settings.role_config(role).fallback is not None, (
            f"{role.value} se pide en local y la plantilla no le declara respaldo"
        )


def test_the_shipped_template_keeps_the_two_desks_in_different_families() -> None:
    """La restricción dura del sistema, comprobada sobre lo que se reparte.

    El respaldo de `bull` es qwen y su primario ya no lo es, así que la
    comprobación tiene que mirar todos los modelos que el rol puede llegar a usar
    y no solo el primario.
    """
    settings = load_settings(TEMPLATE)
    bull = {choice.family for choice in settings.role_choices(AgentRole.BULL)}
    bear = {choice.family for choice in settings.role_choices(AgentRole.BEAR)}

    assert bull & bear == set()


# ───────────────────────────────────── Facturación y precios ──────────────────────────────────────
#
# La tabla es de la página de OpenCode Go, copiada a mano: cada fila se fija aquí con los números
# que se dieron, para que cambiar uno sea un diff visible y no un valor que se movió solo.

GO, PAYG = Billing.GO, Billing.PAYG

PAGE_ROWS = [
    # modelo, billing, pico, entrada, cacheados, salida, límite mensual
    ("mimo-v2.5", GO, None, 0.14, 0.0028, 0.28, 60.0),
    ("hy3", GO, None, 0.14, 0.035, 0.58, 60.0),
    ("kimi-k2.6", GO, None, 0.95, 0.16, 4.00, 60.0),
    ("minimax-m3", GO, None, 0.30, 0.06, 1.20, 60.0),
    ("glm-5.2", GO, None, 1.40, 0.26, 4.40, 60.0),
    ("deepseek-v4-flash", GO, False, 0.15, 0.003, 0.60, 30.0),
    ("deepseek-v4-flash", GO, True, 0.30, 0.006, 1.20, 30.0),
    ("kimi-k2.6", PAYG, None, 0.95, 0.16, 4.00, None),
    ("minimax-m3", PAYG, None, 0.30, 0.06, 1.20, None),
    ("glm-5.2", PAYG, None, 1.40, 0.26, 4.40, None),
    ("deepseek-v4-flash", PAYG, None, 0.14, 0.028, 0.28, None),
]


def test_the_price_table_is_exactly_the_one_from_the_page() -> None:
    shipped = [
        (
            row.model,
            row.billing,
            row.peak,
            row.input_per_mtok,
            row.cached_per_mtok,
            row.output_per_mtok,
            row.monthly_limit_usd,
        )
        for row in DEFAULT_PRICING.rows
    ]
    assert shipped == PAGE_ROWS


def test_the_table_says_when_its_prices_were_copied() -> None:
    assert DEFAULT_PRICING.prices_as_of == date(2026, 10, 2)


def test_two_models_do_not_exist_under_pay_as_you_go() -> None:
    """MiMo-V2.5 y Hy3 solo se sirven con la suscripción: sin precio, no hay coste."""
    payg = {row.model for row in DEFAULT_PRICING.rows if row.billing is PAYG}
    assert {"mimo-v2.5", "hy3"}.isdisjoint(payg)


def test_the_page_estimates_are_per_five_hours_for_each_remote_model() -> None:
    assert DEFAULT_PRICING.page_estimates == {
        "mimo-v2.5": 30_100,
        "deepseek-v4-flash": 13_000,
        "hy3": 4_300,
        "kimi-k2.6": 1_150,
        "minimax-m3": 3_200,
        "glm-5.2": 880,
    }


def test_five_hours_is_a_fifth_of_the_monthly_pool() -> None:
    assert DEFAULT_PRICING.five_hour_share == pytest.approx(0.20)


def test_billing_defaults_to_the_subscription() -> None:
    settings = load_settings(**base_kwargs())
    assert settings.billing is Billing.GO
    assert settings.pricing == DEFAULT_PRICING


def test_the_shipped_template_declares_the_billing_it_assumes() -> None:
    assert "CA_BILLING=go" in TEMPLATE.read_text("utf-8").splitlines()
    assert load_settings(TEMPLATE).billing is Billing.GO


def test_billing_can_be_chosen_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CA_BILLING", "payg")
    assert load_settings(**base_kwargs()).billing is Billing.PAYG


def test_an_unknown_billing_is_rejected_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CA_BILLING", "gratis")
    with pytest.raises(ConfigError, match="CA_BILLING"):
        load_settings(**base_kwargs())


def moment(day: int, hour: int, minute: int = 0) -> datetime:
    """Un instante de octubre de 2026. El 5 es lunes: el 3 y el 4 son sábado y domingo."""
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


@pytest.mark.parametrize(
    ("at", "peak"),
    [
        (moment(6, 2), True),  # martes 02:00 UTC: dentro de 01:00-04:00
        (moment(3, 2), False),  # sábado 02:00 UTC: mismas horas, fuera de los días
        (moment(4, 2), False),  # domingo
        (moment(5, 1), True),  # lunes 01:00: el inicio es inclusivo
        (moment(5, 0, 59), False),
        (moment(5, 3, 59), True),
        (moment(5, 4), False),  # el final es exclusivo
        (moment(5, 5), False),  # entre las dos franjas
        (moment(5, 6), True),
        (moment(9, 9, 59), True),  # viernes
        (moment(9, 10), False),
        (moment(9, 23), False),
    ],
)
def test_the_peak_is_decided_by_the_utc_hour_and_weekday(at: datetime, peak: bool) -> None:
    assert DEFAULT_PRICING.is_peak(at) is peak


def test_the_peak_reads_utc_whatever_zone_the_instant_carries() -> None:
    """02:00 UTC del martes es lunes por la noche en Ciudad de México: sigue siendo pico."""
    from zoneinfo import ZoneInfo

    local = moment(6, 2).astimezone(ZoneInfo("America/Mexico_City"))
    assert local.weekday() == 0
    assert DEFAULT_PRICING.is_peak(local) is True


def test_deepseek_has_one_price_in_the_peak_and_another_outside_it() -> None:
    peak = DEFAULT_PRICING.price_for("deepseek-v4-flash", GO, moment(6, 2))
    off = DEFAULT_PRICING.price_for("deepseek-v4-flash", GO, moment(3, 2))
    assert peak is not None
    assert off is not None
    assert (peak.input_per_mtok, off.input_per_mtok) == (0.30, 0.15)


def test_a_model_without_a_peak_has_the_same_price_all_week() -> None:
    glm = [DEFAULT_PRICING.price_for("glm-5.2", GO, at) for at in (moment(6, 2), moment(3, 2))]
    assert glm[0] is not None
    assert glm[0] == glm[1]


def test_pay_as_you_go_deepseek_ignores_the_peak() -> None:
    peak = DEFAULT_PRICING.price_for("deepseek-v4-flash", PAYG, moment(6, 2))
    off = DEFAULT_PRICING.price_for("deepseek-v4-flash", PAYG, moment(3, 2))
    assert peak == off
    assert peak is not None
    assert peak.input_per_mtok == 0.14


def test_a_model_the_table_does_not_know_has_no_price() -> None:
    assert DEFAULT_PRICING.price_for("gpt-x", GO, moment(6, 2)) is None
    assert DEFAULT_PRICING.price_for("mimo-v2.5", PAYG, moment(6, 2)) is None


def row(**overrides: object) -> PriceRow:
    fields: dict[str, object] = {
        "model": "m",
        "billing": GO,
        "input_per_mtok": 1.0,
        "cached_per_mtok": 0.1,
        "output_per_mtok": 2.0,
        "monthly_limit_usd": 60.0,
    }
    fields.update(overrides)
    return PriceRow.model_validate(fields)


def test_a_subscription_price_needs_its_monthly_limit_and_pay_as_you_go_has_none() -> None:
    with pytest.raises(ValidationError, match="límite mensual"):
        row(monthly_limit_usd=None)
    with pytest.raises(ValidationError, match="no tiene pool"):
        row(billing=PAYG, monthly_limit_usd=60.0)


def test_a_price_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        row(input_per_mtok=-0.1)


def table(*rows: PriceRow) -> PriceTable:
    return PriceTable(rows=rows, prices_as_of=date(2026, 10, 2), page_estimates={})


def test_a_model_cannot_have_two_prices_for_the_same_case() -> None:
    with pytest.raises(ValidationError, match="repetida"):
        table(row(), row())


def test_a_peak_price_needs_its_off_peak_twin_and_cannot_mix_with_a_flat_one() -> None:
    with pytest.raises(ValidationError, match="pico"):
        table(row(peak=True))
    with pytest.raises(ValidationError, match="pico"):
        table(row(peak=True), row(peak=None))


def test_the_peak_hours_are_whole_hours_in_a_day() -> None:
    with pytest.raises(ValidationError, match="horas"):
        PriceTable(
            rows=(row(),),
            prices_as_of=date(2026, 10, 2),
            page_estimates={},
            peak_hours_utc=((4, 1),),
        )
