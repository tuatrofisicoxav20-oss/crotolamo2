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

## Instalar

```bash
./desktop/install.sh            # instala (no enciende)
./desktop/install.sh --start    # instala y enciende ya
```

El instalador:
1. Hace **backup** y apaga el servicio viejo de Crotolamo 1 (`crotolamo-listener.service`).
2. Instala `crotolamo.service` apuntando a **este** repo y su `.venv`.
3. Copia el icono y el lanzador (`Crotolamo` aparece en el menú de apps).
4. Lo habilita para iniciar con la sesión.

Requisitos: Ollama corriendo, el `.venv` con la extra `[voice]`, y los modelos de
voz en `voices/` (lo normal del proyecto).

## Uso

- Abre **«Crotolamo»** desde Actividades, o:
  `/usr/bin/python3 desktop/crotolamo_panel.py`
- A mano: `systemctl --user start|stop|restart crotolamo.service`
- Registro en vivo: `journalctl --user -u crotolamo.service -f`

## Desinstalar

```bash
systemctl --user disable --now crotolamo.service
rm ~/.config/systemd/user/crotolamo.service
rm ~/.local/share/applications/crotolamo.desktop
rm ~/.local/share/icons/hicolor/scalable/apps/crotolamo.svg
systemctl --user daemon-reload
```
