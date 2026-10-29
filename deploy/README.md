# El temporizador de los lotes del estudio

Desde el bloque 2 (que empieza el 2026-10-29 y cuyo lote se abre el 2026-11-06T00:00Z), un
temporizador de systemd de usuario corre cada día a las 19:00 locales los lotes pendientes del
estudio hacia adelante de `solo` (`docs/forward.md`). El lote del bloque 1 lo corre el operador a
mano.

| archivo | qué es |
| --- | --- |
| `crypto-agents-forward.service` | servicio `oneshot`: `python -m crypto_agents.forward_service` en el repositorio |
| `crypto-agents-forward.timer` | `OnCalendar=*-*-* 19:00:00`, `Persistent=true` |
| `forward.env` | el entorno de git del servicio: la ruta de la clave de despliegue y el `pushurl` SSH, sin secretos |
| `50-crypto-agents-inhibit.rules` | regla de polkit: el inhibidor de suspensión sin sesión abierta |

## Qué hace cada ejecución

1. **Antes.**
   - La rama tiene que ser `master`.
   - No puede haber cambios fuera de `data/forward/`.
   - Espera a la red: una conexión TCP a `github.com:443`, el host y puerto de la url de `origin`
     que usa el pull, cada 30 s y durante 300 s como máximo.
   - Hace `git pull --ff-only origin master`.

   Si algo falla, no corre nada: deja una línea en `var/forward/auto.log`, avisa, y la unidad
   termina en `failed`. Los bloques siguen pendientes para el día siguiente.
2. **Durante.** `python -m crypto_agents.forward auto` bajo `systemd-inhibit --what=sleep:idle`.
   - Corre cada bloque pendiente con el mismo código que `batch --block`, sin `--accept-revised`
     y sin `--supersede`.
   - Un 0 pasa al siguiente.
   - Un 3 o un 6 quedan para el día siguiente.
   - Un 1, un 5, un 4 o un 7 detienen todo y avisan. El 4 y el 7 terminan el estudio.
   - Un 2 es un fallo, y se avisa.
   - No corre un bloque retenido: uno cuya última pasada se negó (1) o se detuvo por una vela
     revisada (5). Avisa cada día con su plazo hasta que haya una pasada posterior, la que
     corres tú a mano. Un 1 que no dejó pasada, como un error de red al descargar, se reintenta
     al día siguiente.
   - No corre un bloque inválido o con duplicados: avisa.
   - Un bloque con el plazo vencido, el lote incompleto y sus líneas base sin completar se
     avisa cada día hasta que estén completas (`forward baselines --block …`, a mano).
3. **Después de cada lote completo.**
   - `git add` de exactamente los archivos que el lote listó, todos bajo `data/forward/`;
   - un commit con un mensaje fijo, `forward: batch of block N (inicio), E evaluations, exit code
     0`;
   - `git push origin master`.
4. **Avisos.** `notify-send` y una línea en `var/forward/auto.log`:
   - un lote completo, solo con contadores;
   - todo lo que necesita al operador: una vela revisada, una negativa, un bloque retenido, un
     lote inválido, un cambio de upstream del decisor, un plazo con 3 días o menos, un plazo
     vencido sin las líneas base, la falta de red, un tiempo agotado, un push que falló;
   - el estudio cerrado, o el bloque 52 terminado sin 50 bloques no vacíos: cada día, con el
     bloque y los no vacíos.

   Cada orden de git tiene 120 s como máximo y `notify-send` 10 s; el lote no tiene tiempo
   propio, y la ejecución entera tiene 6 h (`TimeoutStartSec`). Si `notify-send` agota sus 10 s,
   el aviso es la línea del log y la unidad termina en `failed`.

   Sin nada pendiente no escribe ni avisa. Si otra ejecución tiene el cerrojo de la envoltura,
   sale con 0 y deja solo una línea en el log. La salida entera de cada ejecución queda en el
   journal:

   ```bash
   journalctl --user -u crypto-agents-forward
   ```

Siguen siendo manuales:
- aceptar una vela revisada: `batch --block … --accept-revised …`;
- repetir entero un lote inválido: `batch --block … --supersede`;
- completar las líneas base después del plazo: `forward baselines --block …`;
- el análisis: `forward analyze data/forward`.

## Por qué hace falta cada paso

- **El push.** Hoy la laptop sube por HTTPS con `gh auth git-credential`, y el token de `gh` está
  en el llavero de GNOME (`gh auth status` dice «(keyring)»). Un servicio de usuario solo lo
  alcanza mientras la sesión gráfica está abierta. Tras un reinicio sin login, con linger, el
  llavero está cerrado y el push falla. La clave de despliegue no depende de ninguna sesión, de
  ningún agente ni de ninguna terminal.
- **El inhibidor.** Polkit da `inhibit-block-sleep` a una sesión activa o inactiva, pero a un
  proceso sin sesión le pide autenticación de administrador (`allow_any=auth_admin_keep`). Sin la
  regla, tras un reinicio sin login, `systemd-inhibit` se niega, `auto` no llega a correr, y el
  servicio lo registra como fallo.
- **Linger.** Sin él, los servicios de usuario solo existen mientras hay una sesión abierta.

## Instalación (la hace el operador)

1. **La clave de despliegue**, sin frase y solo para este repositorio:

   ```bash
   ssh-keygen -t ed25519 -N '' -C 'crypto-agents laptop: lotes del estudio' -f ~/.ssh/cripto_agents_deploy
   ```

   - En GitHub, en *Settings → Deploy keys → Add deploy key* del repositorio, pega
     `~/.ssh/cripto_agents_deploy.pub` y marca **Allow write access**.
   - La clave privada no sale de la laptop, y nada de este repositorio la lee ni la imprime.
2. **La huella de github.com.** Con `BatchMode=yes` una huella desconocida es un error, no una
   pregunta:

   ```bash
   ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts
   ssh-keygen -lf ~/.ssh/known_hosts -F github.com
   ```

   La huella que imprime tiene que ser la de la página de GitHub *GitHub's SSH key fingerprints*
   (docs.github.com). Si no lo es, borra la línea y no sigas.
3. **La regla de polkit:**

   ```bash
   sudo install -m 644 deploy/50-crypto-agents-inhibit.rules /etc/polkit-1/rules.d/
   ```

   Nombra al usuario `alex`. Si el servicio corre con otro, cámbialo en el archivo.
4. **Linger:**

   ```bash
   loginctl enable-linger "$USER"
   ```
5. **Las unidades.** Se copian, no se enlazan: así una edición del repositorio no cambia el
   servicio instalado sin que lo decidas.

   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/crypto-agents-forward.service deploy/crypto-agents-forward.timer ~/.config/systemd/user/
   systemctl --user daemon-reload
   ```
6. **La comprobación desde systemd, antes de activar nada.** Corre la envoltura con
   `--check` en el mismo entorno que tendrá el servicio. No hace pull, no escribe y no avisa:
   - comprueba la rama y el árbol;
   - comprueba la credencial con `git push --dry-run origin master`;
   - toma y suelta el inhibidor;
   - imprime `auto --dry-run`.

   ```bash
   systemd-run --user --wait --pipe --collect \
     -p WorkingDirectory="$HOME/PycharmProjects/Cripto_Agents" \
     -p EnvironmentFile="$HOME/PycharmProjects/Cripto_Agents/deploy/forward.env" \
     -p Environment=PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
     "$HOME/.local/bin/uv" run --frozen python -m crypto_agents.forward_service --check
   ```

   Tiene que salir con 0 y decir «sí» en las cinco líneas. Para probar el caso sin sesión,
   repítela después de un reinicio, antes de entrar en el escritorio (por SSH).

   Después, un aviso de escritorio de verdad, lanzado desde el gestor de usuario como lo lanzará
   el servicio:

   ```bash
   systemd-run --user --wait --pipe --collect \
     timeout 10 notify-send --app-name=crypto-agents --urgency=critical \
     "Prueba del temporizador" "Si ves esto, los avisos del servicio llegan al escritorio."
   ```

   Con tu sesión abierta tiene que aparecer la notificación y salir con 0. Sin sesión gráfica
   falla o agota los 10 s: entonces los avisos del servicio solo quedan en
   `var/forward/auto.log` y en el journal.
7. **El temporizador:**

   ```bash
   systemctl --user enable --now crypto-agents-forward.timer
   systemctl --user list-timers crypto-agents-forward.timer
   ```

## Límites que conviene saber

- **Cerrar la tapa suspende la laptop aunque el inhibidor esté tomado.** logind ignora los
  inhibidores de `sleep` con la tapa (`LidSwitchIgnoreInhibited=yes` por defecto), y
  `HandleLidSwitch=suspend`. Un lote que corre cuando se cierra la tapa se corta:
  - sus evaluaciones quedan perdidas por el proveedor;
  - se reanudan al día siguiente, y el plazo sigue corriendo.

  Para cerrarla sin suspender hay que cambiar `HandleLidSwitch` en `/etc/systemd/logind.conf`.
- **No corras `batch` ni `baselines` a mano mientras el servicio está activo**
  (`systemctl --user is-active crypto-agents-forward.service`). Comparten un cerrojo
  (`var/forward/lote.lock`): el segundo sale con 1 sin tocar nada, y si es el del temporizador,
  avisa.
- **Lo que versionas a mano lo versionas tú.** Tras un 5 o un 1 resuelto a mano, el commit de
  esas pasadas es tuyo, como antes. El servicio solo versiona los lotes que completa él. Si al
  terminar quedan archivos de `data/forward/` sin versionar que ningún lote pendiente recogerá,
  avisa.
- **Un push que falló deja el commit en local.** Lo sube el siguiente push del servicio, o
  `git push origin master` a mano.
- **Al volver de una suspensión** el temporizador dispara de inmediato si se perdió las 19:00
  (`Persistent=true`). La envoltura espera hasta 5 minutos a que haya red.

## Si el pull no puede avanzar en fast-forward

Pasa cuando `master` local tiene commits del servicio que no subieron (un push que falló) y
entretanto entró algo a `master` en GitHub, como un merge tuyo. El servicio se niega cada día
(«git pull --ff-only no pudo») hasta que lo resuelvas. Nunca hace rebase ni force; a mano:

```bash
cd ~/PycharmProjects/Cripto_Agents
systemctl --user is-active crypto-agents-forward.service   # que no esté corriendo
git fetch origin
git log --oneline origin/master..master   # lo local sin subir: commits «forward: batch of block …»
git log --oneline master..origin/master   # lo que entró en GitHub
git diff --stat origin/master...master    # lo local tiene que tocar solo data/forward/
git rebase origin/master
git push origin master
```

- Si lo local toca algo fuera de `data/forward/`, o el rebase choca, para: `git rebase --abort`
  deja todo como estaba, y hay que mirarlo antes de seguir.
- Los commits del servicio solo añaden archivos de `data/forward/`, así que encima de un merge
  de código no deberían chocar.
- Después del push, la siguiente ejecución del servicio vuelve a correr sola.
