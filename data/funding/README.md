# Funding de perpetuos USDT — Binance USDⓈ-M

Una serie por símbolo del universo, de `binanceusdm`, para descontar el funding del retorno
**neto** de `outcomes.py`. Es una medida descriptiva: no entra en los criterios de la enmienda 2,
que siguen sobre el retorno bruto.

## Procedencia

Descargadas con `perp_probe` (no hay otra descarga): lectura pública, sin claves, sin pedir
órdenes, sin llamar a ningún modelo.

```bash
uv run python -m crypto_agents.perp_probe --exchange binanceusdm \
    --start 2024-08-16 --end 2026-08-16T00:00:00
```

| | |
| --- | --- |
| Origen | `fapi.binance.com/fapi/v1/fundingRate`, vía ccxt (`fetch_funding_rate_history`) |
| Máquina | `alex-mg` (escritorio) |
| Descargado | 2026-10-04 03:19 UTC |
| Rango pedido | `2024-08-16T00:00:00Z` → `2026-08-16T00:00:00Z`, los dos extremos incluidos |
| Commit del código | `d1f6fb3` con cambios sin confirmar (el propio `--start/--end` de `perp_probe`) |
| Filas por serie | 2191 = 730 días × 3 liquidaciones + 1 |
| Intervalo | 8 h en las 2190 diferencias de cada serie; ninguna fuera de rejilla |

El rango es el de `data/history/` (velas de 4 h, 2024-08-16 → 2026-08-15) **más la liquidación de
`2026-08-16T00:00`**, que es el cierre de la última vela. Sin ella, una orden que salga en esa vela
no tendría la liquidación de su instante de salida y su neto quedaría «no determinado».
`data/history/` no tiene README propio: su procedencia es la tabla de `docs/activation.md`.

Cada archivo son los bytes que escribió el sondeo, sin tocar: `timestamp_ms,funding_rate`, la tasa
como fracción del nominal por liquidación (`0.0001` = 1 punto básico), el `repr` de cada float para
no perder un bit. Una tasa positiva la **paga** el largo y la cobra el corto.

## Digests

```
c0a49ea47cad58ac40cb5e19a2efd15381e2e6e8070e597763f487db9e7733e8  adausdt.csv
510211daa2d56b2279b8533f0632d73cab019734689f93c0d1f54533f9173fbb  bnbusdt.csv
3ae0e01844e4b9f17281da6ab8205804ff064c44107558615a0739be48cc84ac  btcusdt.csv
f50ec881651eac0eead303ad34fe03eb92667614eebd2ba3ad0852b8ffd22c08  dogeusdt.csv
02f6e0c5cbccfc6b5eb2e3e35e88d2beebad43553ec210950ff56bca6b07f888  ethusdt.csv
bd5b061ec461882f58679769417fb6c9da6f953a474c3bd219061e5190268fc0  solusdt.csv
d0972913d4bdfa49eb3809605221dadede1bb815cf7d0b6a8a1cf68dcbb33329  xrpusdt.csv
```

Se verifican con `sha256sum -c` y los comprueba `tests/test_funding.py`: editar una serie a mano
haría dos puntuaciones netas incomparables sin que nada lo avise.

## Lo que hay que saber antes de usarlas

- **Las marcas de tiempo traen jitter de milisegundos.** Entre 0 y 16 ms, siempre *después* de la
  hora exacta, en 6 419 de las 15 337 filas, el 42% (`…08:00:00.013`). La primera descarga de
  prueba, otra ventana, llegó a 26 ms. Comparadas tal cual con una entrada a las
  `08:00:00.000`, esa liquidación contaría como «estrictamente posterior» a la entrada. `funding.py`
  ajusta cada marca al minuto más cercano antes de comparar.
- **Un registro que falte dentro de un tramo de 4 h parece un hueco válido de 8 h** y no se puede
  detectar. En estas series no hay tramos de 4 h ni de 1 h, así que no aplica hoy.
- **El funding real se calcula sobre el nominal al precio de marca de cada liquidación.** Aquí se
  suma la tasa sin reescalar por lo que se haya movido el precio desde la entrada: es lo que pide
  la puntuación y el error es de segundo orden frente al nominal de entrada.
