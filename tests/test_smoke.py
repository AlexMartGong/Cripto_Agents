"""Prueba mínima que valida el ciclo de verificación: el paquete se instala y se importa."""

from __future__ import annotations

import crypto_agents


def test_package_is_importable() -> None:
    """Con layout src/, esto solo pasa si el paquete está instalado en el entorno."""
    assert crypto_agents.__version__ == "0.1.0"
