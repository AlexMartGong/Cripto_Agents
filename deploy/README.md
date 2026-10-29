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
| `50-crypto-agents-lid.conf` | drop-in de logind: con el cargador puesto, cerrar la tapa no suspende |
| `crypto-agents-forward-prueba.service` | solo para la prueba de reinicio: `--check` y un `notify-send`, con la salida en `var/forward/prueba-sin-sesion.log` |
| `crypto-agents-forward-prueba.timer` | `OnBootSec=15min`: lanza esa prueba una vez, unos 15 minutos después del arranque |

La laptop queda siempre enchufada y solo la usa el operador. El temporizador tiene que correr
aunque nadie haya entrado tras un reinicio: de ahí la clave de despliegue, linger, la regla de
polkit y las dos piezas contra la suspensión.

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
     bloque y los no vacíos. No es un fallo: es un aviso normal, no urgente, y la unidad no
     termina en `failed`.

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
  ningún agente ni de ninguna terminal. El pull sigue por HTTPS y sin credencial: el repositorio
  es público.
- **El inhibidor.** Polkit da `inhibit-block-sleep` a una sesión activa o inactiva, pero a un
  proceso sin sesión le pide autenticación de administrador (`allow_any=auth_admin_keep`). Sin la
  regla, tras un reinicio sin login, `systemd-inhibit` se niega, `auto` no llega a correr, y el
  servicio lo registra como fallo.
- **Linger.** Sin él, los servicios de usuario solo existen mientras hay una sesión abierta.
- **La tapa.** logind ignora los inhibidores de `sleep` cuando se cierra la tapa
  (`LidSwitchIgnoreInhibited=yes` por defecto). Sin el drop-in, `HandleLidSwitchExternalPower=` no
  tiene valor y logind lo ignora, así que también con el cargador manda `HandleLidSwitch=suspend`.
- **GNOME y el greeter.** Con tu sesión abierta, quien suspende por inactividad es
  gnome-settings-daemon con tus ajustes (`sleep-inactive-ac-type`). Tras un reinicio sin entrar,
  la única sesión gráfica es la del greeter de gdm3, con los suyos: es lo que mira la prueba de
  reinicio, y lo que cambia el paso (h) si hace falta.

## Instalación (la hace el operador)

En este orden. Cada paso dice cómo comprobarlo y qué hacer si falla; no sigas al siguiente con
uno en rojo. Los comandos se lanzan desde `~/PycharmProjects/Cripto_Agents`, en `master`.

### a) La clave de despliegue

Una clave ed25519 sin frase, solo para este repositorio, en la ruta que nombra `forward.env`
(`~/.ssh/cripto_agents_deploy`). La clave privada no sale de la laptop: ningún paso la muestra, la
copia ni la lee, y nada de este repositorio lo hace.

1. Generarla, sin pisar una que ya exista:

   ```bash
   test -e ~/.ssh/cripto_agents_deploy && echo "ya existe: no la regeneres"
   ssh-keygen -t ed25519 -N '' -C 'crypto-agents laptop: lotes del estudio' -f ~/.ssh/cripto_agents_deploy
   ```

   **Comprobación:** `ls -l ~/.ssh/cripto_agents_deploy` da permisos `-rw-------`, y
   `ssh-keygen -lf ~/.ssh/cripto_agents_deploy.pub` imprime `256 SHA256:… (ED25519)`.
   **Si falla:** si `ssh-keygen` pregunta si sobrescribe, contesta `n`: la clave ya existe, y
   basta con que su pública esté en GitHub (paso 2).
2. La clave **pública** en GitHub: en el repositorio, *Settings → Deploy keys → Add deploy key*.
   Pega lo que imprime `cat ~/.ssh/cripto_agents_deploy.pub` (una línea que empieza por
   `ssh-ed25519`) y marca **Allow write access**.

   **Comprobación:** la clave aparece en la lista como *Read/write*, y la huella que muestra
   GitHub es la del `ssh-keygen -lf … .pub` del paso 1.
   **Si falla:** si sale sin escritura, bórrala y añádela otra vez con la casilla marcada; si
   GitHub dice que la clave ya está en uso, está puesta en otra cuenta o repositorio, y hay que
   quitarla de allí o generar otra con otro nombre (y cambiar la ruta en `forward.env`).
3. La clave de host de github.com en `~/.ssh/known_hosts`. Con `BatchMode=yes` un host
   desconocido es un error, no una pregunta. Primero a un archivo aparte, para comparar la huella
   antes de confiar en ella:

   ```bash
   hostkey=$(mktemp)
   ssh-keyscan -t ed25519 github.com > "$hostkey"
   ssh-keygen -lf "$hostkey"
   ```

   Compara la huella `SHA256:…` que imprime con la de la línea ED25519 de la página de GitHub
   *GitHub's SSH key fingerprints*
   (<https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints>;
   si la dirección cambió, búscala por su título en docs.github.com). Copia la huella de la página
   en ese momento: este README no lleva ninguna. Solo si son iguales:

   ```bash
   cat "$hostkey" >> ~/.ssh/known_hosts
   rm "$hostkey"
   ssh -i ~/.ssh/cripto_agents_deploy -o IdentitiesOnly=yes -o BatchMode=yes -T git@github.com
   ```

   **Comprobación:** `ssh-keygen -F github.com -f ~/.ssh/known_hosts` encuentra la línea, y el
   `ssh -T` contesta que te autenticaste nombrando el repositorio, `AlexMartGong/Cripto_Agents`,
   y que GitHub no da shell (sale con 1: es lo normal). El permiso de escritura lo prueba el
   `git push --dry-run` del paso (f).
   **Si falla:**
   - si las huellas no coinciden, `rm "$hostkey"` y para: no añadas nada a `known_hosts`;
   - `Host key verification failed`: falta la línea en `known_hosts`;
   - `Permission denied (publickey)`: la pública no está en el repositorio, o la ruta de `-i` no
     es la del paso 1.

### b) Linger

```bash
loginctl enable-linger "$USER"
loginctl show-user "$USER" --property=Linger
```

**Comprobación:** imprime `Linger=yes`. Desde ahora el gestor de usuario arranca con el sistema y
sigue vivo sin sesión.
**Si falla:** si polkit pide contraseña o dice `Access denied`, `sudo loginctl enable-linger alex`.

### c) La regla de polkit

Se instala tal como está. Nombra al usuario `alex`; si el servicio corre con otro, cámbialo en el
archivo antes.

```bash
sudo install -m 644 deploy/50-crypto-agents-inhibit.rules /etc/polkit-1/rules.d/
sudo ls -l /etc/polkit-1/rules.d/50-crypto-agents-inhibit.rules
journalctl -u polkit.service --since '-5 min' --no-pager
```

**Comprobación:** el archivo está (el directorio es `root:polkitd` con permisos 750, por eso el
`ls` va con sudo), y polkitd volvió a cargar sus reglas solo, sin ninguna línea `Error compiling
script`. Que funcione sin sesión se prueba en (g), en la fila del inhibidor.
**Si falla:** con un error de compilación en el journal, `sudo rm` el archivo instalado y compáralo
con el del repositorio: polkit sigue con las reglas de antes.

### d) El drop-in de logind

```bash
sudo install -D -m 644 deploy/50-crypto-agents-lid.conf /etc/systemd/logind.conf.d/50-crypto-agents-lid.conf
systemd-analyze cat-config systemd/logind.conf | grep -A9 '50-crypto-agents-lid.conf'
```

No reinicies `systemd-logind` (puede cerrar la sesión gráfica): el drop-in se aplica en el
reinicio de la prueba (g).

**Comprobación:** `cat-config` lista `# /etc/systemd/logind.conf.d/50-crypto-agents-lid.conf` con
`HandleLidSwitchExternalPower=ignore`. Después del reinicio de (g), lo confirma logind mismo:

```bash
busctl get-property org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager HandleLidSwitchExternalPower
```

tiene que dar `s "ignore"`; antes del reinicio da `s ""`.
**Si falla:** si `cat-config` no lo lista, revisa la ruta y que el archivo acabe en `.conf`.

### e) GNOME: sin suspensión por inactividad con el cargador

En tu sesión gráfica (gsettings escribe en tu dconf):

```bash
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type
```

**Comprobación:** imprime `'nothing'`. El 2026-10-29 ya lo imprimía en esta laptop. Vale mientras
tu sesión está abierta; sin sesión manda el greeter, paso (h).
**Si falla:** si `gsettings` no puede escribir, lánzalo desde una terminal de tu escritorio, no por
SSH ni desde un servicio.

### f) El servicio y el temporizador

1. Las unidades se copian, no se enlazan: así una edición del repositorio no cambia el servicio
   instalado sin que lo decidas.

   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/crypto-agents-forward.service deploy/crypto-agents-forward.timer ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemd-analyze --user verify ~/.config/systemd/user/crypto-agents-forward.service ~/.config/systemd/user/crypto-agents-forward.timer
   ```

   **Comprobación:** `verify` sale con 0 y no nombra estas unidades. (En esta laptop avisa de
   `spice-vdagent.service`, que no es nuestra.)
   **Si falla:** lo que nombra es la línea a corregir; corrígela en `deploy/`, vuelve a copiar y a
   hacer `daemon-reload`.
2. La comprobación desde systemd, antes de activar nada. Corre la envoltura con `--check` en el
   mismo entorno que tendrá el servicio. No hace pull, no escribe y no avisa:
   - comprueba la rama y el árbol;
   - comprueba la credencial con `git push --dry-run origin master`, que va por SSH con la clave
     de despliegue y necesita permiso de escritura;
   - toma y suelta el inhibidor (`systemd-inhibit --what=sleep:idle … true`);
   - imprime `auto --dry-run`.

   ```bash
   systemd-run --user --wait --pipe --collect \
     -p WorkingDirectory="$HOME/PycharmProjects/Cripto_Agents" \
     -p EnvironmentFile="$HOME/PycharmProjects/Cripto_Agents/deploy/forward.env" \
     -p Environment=PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
     "$HOME/.local/bin/uv" run --frozen python -m crypto_agents.forward_service --check
   ```

   **Comprobación:** sale con 0 y dice «sí» en las cinco líneas.
   **Si falla:** la fila con **NO** dice qué paso revisar:
   - rama o árbol: vuelve a `master` y deja el árbol limpio fuera de `data/forward/`;
   - credencial del push: el paso (a). El detalle es la última línea del error; para verlo entero,
     repite el push a mano con la clave:

     ```bash
     GIT_SSH_COMMAND='ssh -i ~/.ssh/cripto_agents_deploy -o IdentitiesOnly=yes -o BatchMode=yes' \
       git push --dry-run git@github.com:AlexMartGong/Cripto_Agents.git master
     ```

     `marked as read only` es la casilla de escritura del paso (a);
   - inhibidor: el paso (c). Con tu sesión abierta un «sí» todavía no prueba el caso sin sesión,
     que es el de (g);
   - `auto --dry-run`: su salida, debajo de las filas, dice qué no pudo leer.
3. Un aviso de escritorio de verdad, lanzado desde el gestor de usuario como lo lanzará el
   servicio:

   ```bash
   systemd-run --user --wait --pipe --collect \
     timeout 10 notify-send --app-name=crypto-agents --urgency=critical \
     "Prueba del temporizador" "Si ves esto, los avisos del servicio llegan al escritorio."
   ```

   **Comprobación:** con tu sesión abierta aparece la notificación y sale con 0.
   **Si falla:** si sale con otro código o agota los 10 s, los avisos del servicio solo quedan en
   `var/forward/auto.log` y en el journal; el servicio corre igual.
4. El temporizador:

   ```bash
   systemctl --user enable --now crypto-agents-forward.timer
   systemctl --user list-timers crypto-agents-forward.timer
   ```

   **Comprobación:** `list-timers` da la siguiente ejecución a las 19:00 locales.
   **Si falla:** `systemctl --user status crypto-agents-forward.timer` dice por qué no arrancó.

### g) La prueba de reinicio, sin sesión

Una unidad de prueba corre `--check` sola, unos 15 minutos después del arranque, sin que nadie
haya entrado, y deja la salida en `var/forward/prueba-sin-sesion.log`. Se instala solo para esta
prueba y se quita al final.

1. Instalarla. Se activa con `enable` y **sin** `--now`: un `OnBootSec=` que ya pasó dispara en
   cuanto se arranca el temporizador, y la prueba tiene que correr tras el reinicio.

   ```bash
   mkdir -p var/forward
   cp deploy/crypto-agents-forward-prueba.service deploy/crypto-agents-forward-prueba.timer ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemd-analyze --user verify ~/.config/systemd/user/crypto-agents-forward-prueba.service ~/.config/systemd/user/crypto-agents-forward-prueba.timer
   systemctl --user enable crypto-agents-forward-prueba.timer
   systemctl --user is-enabled crypto-agents-forward-prueba.timer
   ```

   **Comprobación:** `verify` sale con 0 sin nombrar estas unidades, e `is-enabled` dice `enabled`.
   **Si falla:** si `is-enabled` no dice `enabled`, comprueba que copiaste las dos unidades y repite
   el `daemon-reload` y el `enable`. Si alguien la arrancó con `--now` y ya dejó líneas en el log,
   esas tenían tu sesión abierta y no cuentan: cuenta la ejecución del reinicio.
2. Con el cargador puesto y la tapa abierta, reinicia y **no inicies sesión durante 40 minutos**.
   La prueba corre hacia el minuto 15; el resto del tiempo es para ver si algo suspende el equipo.
   Si a las 19:00 locales la laptop estaba apagada, el temporizador de verdad (`Persistent=true`)
   corre también al arrancar, y deja su línea en `var/forward/auto.log`.
3. Pasados los 40 minutos, entra y comprueba tres cosas.

   **La salida de la prueba:**

   ```bash
   cat var/forward/prueba-sin-sesion.log
   ```

   - La hora de la primera línea (con su desfase, `-06:00`) es unos 15 minutos posterior a la de
     `uptime --since`, la línea siguiente.
   - En `loginctl list-sessions` tu usuario solo tiene la sesión de clase `manager`, la del gestor
     de usuario; ninguna de clase `user`. La del greeter es de `gdm-greeter`.
   - Las cinco filas de `--check` dicen «sí». Las dos que esta prueba existe para ver: `inhibidor
     de suspensión` (la regla de polkit, sin sesión) y `credencial del push` (la clave de
     despliegue, sin llavero).
   - `- notify-send: código N`. Sin sesión gráfica se espera un código distinto de 0, o 124 si
     agotó los 10 s: entonces los avisos del servicio sin sesión solo quedan en el log y en el
     journal. No es un fallo de la prueba; es lo que hay que saber.

   **Que no hubo ninguna suspensión desde el arranque:**

   ```bash
   journalctl -b -u systemd-suspend.service --no-pager
   ```

   Tiene que decir `-- No entries --`.

   **Que el gestor de usuario arrancó con el sistema:**

   ```bash
   uptime --since
   systemctl show "user@$(id -u).service" --property=ActiveEnterTimestamp
   journalctl -b -u systemd-logind -o short-iso --no-pager | grep "of user '$USER'"
   ```

   `ActiveEnterTimestamp` cae a menos de un minuto del arranque, y en el journal de logind la
   primera sesión de tu usuario es la de clase `manager`, al arrancar; la de clase `user` es la de
   tu entrada, 40 minutos después. (Es el formato de systemd 259 en esta laptop: `New session 'N'
   of user 'alex' with class 'manager' …`.)

   **Si falla:**
   - no hay log: el temporizador de prueba no corrió. `journalctl --user -b -u
     crypto-agents-forward-prueba.timer -u crypto-agents-forward-prueba.service`, y revisa (b);
   - `inhibidor de suspensión: **NO**`: el paso (c);
   - `credencial del push: **NO**`: el detalle dice si fue la clave (paso a) o la red. Sin red
     antes de entrar, la wifi tiene que ser de todos los usuarios y con la contraseña guardada en
     el sistema: `nmcli -g connection.permissions,802-11-wireless-security.psk-flags connection
     show "<red>"` imprime una línea vacía y `0` (el 2026-10-29, «La Pasadita 2» daba eso);
   - hay entradas de `systemd-suspend.service`: el paso (h);
   - `ActiveEnterTimestamp` es la hora de tu entrada: linger no está, el paso (b).
4. Quitar la unidad de prueba:

   ```bash
   systemctl --user disable --now crypto-agents-forward-prueba.timer
   rm ~/.config/systemd/user/crypto-agents-forward-prueba.service ~/.config/systemd/user/crypto-agents-forward-prueba.timer
   systemctl --user daemon-reload
   systemctl --user list-unit-files 'crypto-agents-forward-prueba*'
   ```

   **Comprobación:** `list-unit-files` dice `0 unit files listed.` El log se queda en
   `var/forward/` (no se versiona); bórralo cuando ya no lo necesites.

### h) Si en (g) hubo una suspensión: el greeter

Primero, de dónde vino:

```bash
journalctl -b -u systemd-logind -o short-iso --no-pager | grep -i -E 'lid|suspend|sleep'
```

- Con `Lid closed.` justo antes: fue la tapa, y el drop-in de (d) no se aplicó; el `busctl` de
  (d) tiene que dar `s "ignore"`.
- Sin la tapa: la pidió una sesión, y tras un reinicio sin entrar la única con gestor de energía
  es la del greeter. Lo que sigue la cambia.

Cómo configura gdm3 la energía del greeter, leído en esta laptop el 2026-10-29 (Ubuntu 26.04.1,
gdm3 50.1, gnome-settings-daemon 50.0) sin cambiar nada y sin sudo, cada valor con su fuente:

- **El perfil de dconf del greeter** es `/usr/share/dconf/profile/gdm` (paquete gdm3): `user-db:user`
  y después `file-db:/var/lib/gdm3/greeter-dconf-defaults`.
- **Esa base la compila** `/usr/share/gdm/generate-config`, con `dconf compile
  '/var/lib/gdm3/greeter-dconf-defaults' '/usr/share/gdm/dconf'`. `gdm.service` lo corre en
  `ExecStartPre=` y en `ExecReload=` (`systemctl cat gdm.service`), así que se recompila en cada
  arranque.
- **`/usr/share/gdm/dconf/`** tiene `00-upstream-settings`, que no toca la energía, y
  `90-debian-settings`, un enlace a `/etc/gdm3/greeter.dconf-defaults` (`ls -la
  /usr/share/gdm/dconf`).
- **En `/etc/gdm3/greeter.dconf-defaults`**, la sección `[org/gnome/settings-daemon/plugins/power]`
  (líneas 30 a 40) tiene sus cuatro claves comentadas, entre ellas `# sleep-inactive-ac-timeout=1200`
  y `# sleep-inactive-ac-type='suspend'`. La base compilada no tiene nada bajo esa ruta: con un
  perfil que solo lleva `file-db:/var/lib/gdm3/greeter-dconf-defaults`, `dconf dump
  /org/gnome/settings-daemon/plugins/power/` sale vacío.
- **Queda el esquema:** `sleep-inactive-ac-timeout` 900 y `sleep-inactive-ac-type` `'suspend'`
  (`/usr/share/glib-2.0/schemas/org.gnome.settings-daemon.plugins.power.gschema.xml`), con el
  override de Ubuntu `sleep-inactive-ac-timeout = 0` (línea 52 de
  `/usr/share/glib-2.0/schemas/10_ubuntu-settings.gschema.override`, paquete ubuntu-settings). Ese
  override no lleva calificador de escritorio, así que vale también en el greeter, cuya sesión
  declara `DesktopNames=GNOME-Greeter:GNOME`
  (`/usr/share/gdm/greeter/wayland-sessions/gnome-greeter.desktop`).
- **Resultado:** con ese perfil y `XDG_CURRENT_DESKTOP=GNOME-Greeter:GNOME`, `gsettings get` da
  `sleep-inactive-ac-timeout` `0` y `sleep-inactive-ac-type` `'suspend'`. Con 0 el greeter no
  suspende por inactividad.
- **Lo que no se pudo leer sin sudo:** la `user-db` del propio greeter, que va primero en el perfil
  y por tanto gana. El greeter corre como `gdm-greeter`, un usuario dinámico que solo existe
  mientras corre (sesión `c1`, clase `greeter`, en el journal de logind; `getent passwd
  gdm-greeter` no da nada con tu sesión abierta). `/var/lib/gdm3/seat0/` tiene `config` y `state`
  del uid 60578, con permisos 700.

Así que, por lo que se puede leer, el greeter no debería suspender. Si (g) dice lo contrario, fija
el valor explícitamente en el archivo que gdm3 deja para eso:

1. En `/etc/gdm3/greeter.dconf-defaults`, dentro de `[org/gnome/settings-daemon/plugins/power]`,
   cambia la línea `# sleep-inactive-ac-type='suspend'` por `sleep-inactive-ac-type='nothing'`:

   ```bash
   sudoedit /etc/gdm3/greeter.dconf-defaults
   ```

   Es un archivo de configuración de gdm3 gestionado por ucf (`ucfq
   /etc/gdm3/greeter.dconf-defaults`: gdm3, sin cambios el 2026-10-29): si una actualización de
   gdm3 trae otra versión, ucf pregunta, y hay que quedarse con la local.
2. Se compila en el siguiente arranque (`ExecStartPre=` de `gdm.service`). Repite (g) entera,
   unidad de prueba incluida.
3. **Comprobación**, sin sudo, después de ese reinicio:

   ```bash
   perfil=$(mktemp)
   printf 'file-db:/var/lib/gdm3/greeter-dconf-defaults\n' > "$perfil"
   DCONF_PROFILE="$perfil" dconf dump /org/gnome/settings-daemon/plugins/power/
   rm "$perfil"
   ```

   tiene que imprimir `sleep-inactive-ac-type='nothing'`, y la repetición de (g) ninguna entrada
   de `systemd-suspend.service`.
4. **Si falla:** si la base compilada ya lo lleva y el equipo sigue suspendiendo, la `user-db` del
   greeter lo pisa, y mirarla necesita sudo. Para ahí y míralo antes de cambiar nada más.

## Límites que conviene saber

- **Con batería, cerrar la tapa suspende la laptop aunque el inhibidor esté tomado.** logind ignora
  los inhibidores de `sleep` con la tapa (`LidSwitchIgnoreInhibited=yes` por defecto). Con el
  cargador puesto el drop-in del paso (d) la deja en `ignore`; sin él, y con batería siempre, manda
  `HandleLidSwitch=suspend`. Un lote que corre cuando se suspende se corta:
  - sus evaluaciones quedan perdidas por el proveedor;
  - se reanudan al día siguiente, y el plazo sigue corriendo.
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
