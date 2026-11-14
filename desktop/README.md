# Crotolamo — integración de escritorio (Fedora / GNOME)

Una mini-ventana flotante para **ver si Crotolamo está escuchando y encenderlo o
apagarlo** sin tocar la terminal. Se integra en Fedora como una app más del menú.

```
┌────────────────────────┐
│     C R O T O L A M O   │
│        ●  activo        │   ← verde escuchando · gris apagado · rojo error
│   ┌───────────────┐     │
│   │   APAGAR    ⏻ │     │   ← botón grande encender/apagar
│   └───────────────┘     │
│   Modo  [half-duplex ▾] │   ← cómo escucha la voz
│   Iniciar con sesión  ◉ │   ← arranca solo al entrar
│   Ver registro · Reiniciar
└────────────────────────┘
```

## Cómo funciona

- El **estado "activo"** = el servicio de usuario `crotolamo.service` corriendo
  `python -m crotolamo listen` (la escucha de voz por wake-word).
- La ventana no importa el núcleo de Crotolamo: lo controla por `systemctl --user`
  (encender/apagar/reiniciar/arranque-automático) y refresca el estado sola cada
  1.5 s. Por eso corre con el **python3 del sistema** (el que trae GTK), no con el
  venv del proyecto.
- El **modo de voz** se guarda en `~/.config/crotolamo/listener.env` y el servicio
  lo lee al arrancar:
  - `half-duplex` (altavoces, seguro) → `--no-barge-in`
  - `barge-in` (auriculares, se puede interrumpir) → `--barge-in`
  - `simple` (a prueba de fallos) → `--simple`
- Si el servicio no está instalado el panel lo dice («No instalado — ejecuta
  desktop/install.sh»), y si `systemctl` falla, el motivo aparece bajo el estado.

## Instalar

```bash
./desktop/install.sh            # instala (no enciende)
./desktop/install.sh --start    # instala y enciende ya
```

El instalador:
1. Hace **backup** y apaga el servicio viejo de Crotolamo 1 (`crotolamo-listener.service`).
2. Instala `crotolamo.service` (la escucha) y `crotolamo-hud.service` (el overlay
   que aparece al convocarlo) con la ruta real de **este** repo y su `.venv`, lo
   hayas clonado donde lo hayas clonado. Si ya había uno distinto (editado a mano),
   lo guarda antes como `*.bak-FECHA`.
3. Copia el icono y el lanzador (`Crotolamo` aparece en el menú de apps).
4. Los habilita para iniciar con la sesión (el HUD arranca con la sesión gráfica;
   `--start` enciende la escucha y solo reinicia el HUD si ya corría).

Requisitos: Ollama corriendo, el `.venv` con la extra `[voice]`, y los modelos de
voz en `voices/` (lo normal del proyecto). Si el repo no está en
`~/Documentos/crotolamo2`, apunta `[paths].voces` a su `voices/` en
`config/crotolamo.local.toml`. Para cambiar algo de un servicio sin que el
instalador lo pise: `systemctl --user edit crotolamo.service`.

## Uso

- Abre **«Crotolamo»** desde Actividades, o:
  `/usr/bin/python3 desktop/crotolamo_panel.py`
- A mano: `systemctl --user start|stop|restart crotolamo.service`
- Registro en vivo: `journalctl --user -u crotolamo.service -f`

### Ventanas de Hyprland desde el servicio

Las tools de ventanas (`list_windows`, `app_status`, `focus_window`, `close_window`)
usan `hyprctl`, que necesita `HYPRLAND_INSTANCE_SIGNATURE`. `crotolamo.service`
arranca con `default.target` y solo ve esa variable si la sesión la exporta a
systemd: añade a `~/.config/hypr/hyprland.conf` la línea

```
exec-once = dbus-update-activation-environment --systemd WAYLAND_DISPLAY XDG_CURRENT_DESKTOP HYPRLAND_INSTANCE_SIGNATURE
```

y reinicia el servicio (`systemctl --user restart crotolamo`): un proceso que ya
corría no ve lo que se exporta después. Si falta, Crotolamo contesta «hyprctl se
quejó, patrón: HYPRLAND_INSTANCE_SIGNATURE not set! (is hyprland running?)» y lo
deja en el registro.
(Receta sin validar todavía en la laptop.)

## Desinstalar

```bash
systemctl --user disable --now crotolamo.service crotolamo-hud.service
rm ~/.config/systemd/user/crotolamo.service ~/.config/systemd/user/crotolamo-hud.service
rm ~/.local/share/applications/crotolamo.desktop
rm ~/.local/share/icons/hicolor/scalable/apps/crotolamo.svg
systemctl --user daemon-reload
```
