# Auditoría y optimización de Crotolamo 2 — 2026-11-20

Rama `claude/optimistic-ride-iloitn` (así se llama en GitHub; `audit-fixes`
fue solo su nombre de trabajo en el contenedor y no existe en el remoto): 44
commits sobre `f316103` (el merge del PR #1, del 2026-10-04) hasta `b81e1c2`;
uno de ellos, `601f02f`, es la primera versión de este informe. **Los conteos
de tests, commits y diff de este documento son los del commit `b81e1c2`; las
cifras de los benches de la §3 son las medidas al cerrar cada causa raíz y no
se repitieron al final.** Detrás va un commit más, `4d0d7c9`, que solo
corrige un test (§1): no cambia el producto ni el número de tests. El commit
que trae esta versión del informe va detrás y solo toca `docs/`: su run de CI
no consta aquí.

Marcas: **[ARREGLADO]** lleva el commit que lo arregló; **[DECISIÓN]** cambia
algo que oyes o ves y te toca a ti. "ON" / "OFF" = activo o apagado con la
config de fábrica (`config/crotolamo.toml`).

Lo primero que tienes que saber: **nada de esto se probó en tu laptop**. Se hizo
en un contenedor sin micrófono, sin altavoz, sin pantalla (GTK), sin
reproductor MPRIS, sin Ollama, sin API key de GLM y sin la extra de mem0. Todo
lo de esas áreas se ejercitó con dobles, o con los modelos reales (Whisper,
Silero, openWakeWord, Piper) sobre audio sintético. La §8 dice exactamente qué
significa eso y trae la lista de comprobaciones para tu máquina.

## 0. Resumen ejecutivo

- **Qué validación tiene esto.** Nada se probó en tu laptop (§8). Y el CI de
  GitHub salió en rojo en los siete primeros pushes del 20 de noviembre (de
  `2f82ce6` a `601f02f`) sin correr un solo test, y no se miró hasta el
  cierre: en esos commits "suite verde" fue solo la corrida local. No lo causó
  la rama (el runner nuevo no puede importar `sounddevice` sin servidor de
  audio), pero al arreglarlo (`4f64649`) el CI destapó un test de esta
  auditoría que daba por hecho el orden de los archivos de una carpeta.
  Corregido ese test (`4d0d7c9`), el run 59 es **verde**: el primero desde el
  14 de noviembre y el único que valida el código final (§1).
- **Tamaño del trabajo.** Primera pasada: 304 hallazgos; de ellos 198
  arreglados con test, 41 quedaron como decisión tuya y 39 pasaron a rediseño de
  rendimiento. La revisión final del código ya corregido confirmó 101 defectos
  más (todos arreglados). La suite pasó de 769 a 3866 tests y pasa con
  Python 3.11 y 3.12, aquí y, en el código final, en el CI de GitHub. Eso no
  es "probado con micrófono" (§8).
- **Seguridad.** La confirmación de una acción destructiva ahora dice QUÉ va a
  tocar (antes el mismo "sí" valía para una nota que para pisar `~/.bashrc`) y
  ya no toma por un sí una pregunta ("¿Se va a borrar todo?" ejecutaba). El
  corral dejaba pasar `file://` y el doctor podía imprimir tu API key.
  **[ARREGLADO]** (§2). `remember_fact` guardaba contraseñas y las mandaba a la
  nube: **[ARREGLADO]** en lo que el filtro reconoce; una contraseña dictada en
  palabras o un número de tarjeta todavía se guardan (huecos en §2).
- **Fallos silenciosos en tu modo diario** (loop simple, wake difuso): "pon
  música" llegaba como "música"; con el modelo local la tool no corría y oías
  "Listo, patrón"; los hechos que le dictabas desaparecían tras 20 turnos; un
  error del LLM no se oía. **[ARREGLADO]** (§2).
- **Loop concurrente.** Las tres ALTA de septiembre (V1, V2, V4) están cerradas:
  ya no pierde la orden si haces una pausa tras el wake, ya no se queda vivo y
  sordo, y un barge-in ya no deja sonar la frase vieja. Validado solo con audio
  simulado.
- **Rendimiento que ya va encendido.** RAM del servicio tras una orden: 872 →
  683 MB en tu modo y 980 → ~578 MB en el concurrente (ya no carga torch ni
  sklearn). Medido con los Whisper de fábrica (`base` + `tiny`); con tus
  `small` + `base` las cifras absolutas suben (§3). Piper ya no se queda
  ~409 MB tras una respuesta larga. Sin internet, el turno que descubre la
  caída pasa de 497 s a 102 s hasta la primera palabra (90 s son la carga en
  frío de Ollama). Sin nada sonando, 57600 procesos `playerctl` al día pasan a
  unos 288; mientras suena música el sondeo sigue igual que antes (uno cada
  1.5 s). El apretón TCP+TLS con la nube sale del camino del turno (§3).
- **Lo que NO mejoró, dicho claro.** En tu modo diario, con la tele o gente
  hablando, el coste en reposo es el mismo: 12 pasadas de Whisper por minuto
  antes y después. La palanca que lo baja (`[wake].pregate`) va apagada porque
  necesita un modelo entrenado con tu voz (§3.3, D14).
- **Lo más jugoso va apagado.** Las ganancias de latencia grandes cambian lo
  que oyes o no se pueden validar sin tu hardware: micrófono abierto por sesión
  (`audio_session`), hablar antes con GLM (`stream_release`: 1.7 → 0.6 s en el
  bench), pre-síntesis. Son 13 interruptores; cómo probarlos uno a uno, en §8.
- **El propio trabajo metió regresiones y la revisión las cazó.** Dos en el
  wake difuso (tu modo) vivieron entre los commits `124ecf4` y `b09e63d`. Otra,
  en las listas numeradas, la metió el arreglo de voz `2259ba2` y la cazó la
  revisión posterior de ese commit (`22e52d2`, §11). Siete efectos secundarios
  de la auditoría siguen abiertos como decisión tuya (§4): `light_control` dice
  "prendí" con una entidad que no es luz (D40); con el loop concurrente y sin
  micrófono, el servicio se reinicia cada ~5 s saludando (D12); "muéstrame el
  estado del sistema" ya no enruta con el modelo local (D29); con openWakeWord
  0.6 el score sube a ~0.5 tras cada reinicio del detector y un `oww_threshold`
  más bajo abre turnos fantasma (D16); la confirmación por voz lee la ruta
  literal y la `~` no se oye (D1); el apagado por señal, que la auditoría
  alargó hasta ~4 s y que con tres reproductores colgados y la memoria activa
  pasa del tope del unit (D20); y la lista numerada que no empieza en 1, que
  ahora dice sus números (§7.1).
- **Dos decisiones de producto se tomaron por ti.** Un número al empezar una
  frase ahora se dice ("42. ¿Algo más, patrón?" sonaba "¿Algo más, patrón?").
  Precio: una lista numerada que no empieza en 1 dice sus números. Y esos
  números se dicen cada uno como frase corta aparte ("Reinicia el equipo",
  "4.", "Ir al banco a las 10"), con unos 0.3 s más de audio por número. Las
  listas que empiezan en 1 siguen sin decirlos. Deshacerlo es chico (§7.1).
- **Te quedan 50 decisiones numeradas** (§7.3-§7.9), más 13 interruptores
  apagados que encender o no (§7.2), 36 preguntas menores (§7.10) y la de los
  números de las listas (§7.1); casi todas pequeñas. Las que miraría primero,
  porque tocan tu modo diario: las palabras que dejan una orden "a medias" (D6:
  19 de 40 órdenes completas de prueba esperan 2.5 s de más), cómo dice una
  ruta la confirmación por voz (D1) y el coste en reposo con la tele (D14). Y
  si usas `[memoria]`: `olvidar_recuerdo` borra el recuerdo equivocado en 21 de
  42 casos medidos (D22).
- **Qué hacer primero en la laptop:** la lista de la §8 (doctor, smoke, once
  pruebas de voz, apagado). Y revisa la libreta de hechos: si alguna vez le
  dictaste una contraseña, sigue ahí y ahora viaja en cada petición a la nube.
- **Dónde se paró.** Los dos últimos bloques de arreglos (voz y apagado)
  tuvieron una revisión adversarial posterior; lo que confirmó está arreglado,
  y lo que refutó (4 de 10 hallazgos) y dos residuos de gusto quedan como
  están (§11, §7.1). Ahí se cerró: no se abrieron más rondas de revisión.

## 1. Cómo se hizo

**Primera pasada.** Doce frentes (núcleo del agente, LLM, memoria, tools de
archivos y seguridad, tools de red y escritorio, voz en ejecución, captura de
voz, MCP, escritorio, scripts y CI, y dos transversales: cableado de tools y
concurrencia/ciclo de vida) revisados línea por línea. Salieron 304 hallazgos:
21 altos, 99 medios y 184 bajos; 273 reproducidos con un script y 31 por
lectura. Por tipo: 122 bugs, 51 de cableado, 48 de rendimiento, 22 de config
que no coincide con el código, 19 de concurrencia, 11 fugas de recursos, 11 de
código muerto, 10 de seguridad, 5 tests que tapaban un bug y 5 de docs.

**Resolución, test primero.** Cada hallazgo lo volvió a verificar contra el
código alguien distinto de quien lo encontró, y cada arreglo llegó con un test
que fallaba antes. De los 304: 198 arreglados; 41 decisión tuya; 39 a rediseño
de rendimiento; 11 pedían tocar un archivo de otro frente y se aplicaron
después, en commits de integración (`4d2b642`, `fe513ea` y otros); 15 no se
tocaron (13 ya los había arreglado otro frente, 1 se refutó y 1 era código
muerto que entonces se dejó). Veredictos: 251 confirmados, 47 ya arreglados al
llegar, 5 plausibles y 1 refutado.

**Rendimiento por causa raíz.** Aparte, una pasada de medición dio 70
hallazgos de rendimiento: 69 agrupados en nueve causas raíz y uno descartado
por insignificante. Cada causa tuvo diseño, benchmark antes y después, y
revisión adversarial por dos revisores (corrección y comportamiento en
ejecución), casi siempre en dos rondas. En las seis causas de las que queda
registro detallado, esas rondas encontraron 38 problemas. En dos causas la
segunda ronda salió limpia; en las otras cuatro todavía encontró problemas (9
en total), que se arreglaron o se descartaron con su razón sin una tercera
ronda que lo confirmara: el "cero abiertos" con que cerraron las seis es la
cuenta de quien implementó. Ese código lo volvió a mirar después la revisión
final. De las otras tres causas queda lo que dicen sus commits. Regla que se
siguió: lo que cambia lo que oyes, o no se puede validar sin tu hardware,
entra apagado.

**Revisión final, función por función.** Sobre el código ya corregido: 54
unidades (47 de código y 7 transversales). Hay veredicto escrito para 1017
funciones en 45 unidades (819 sin observaciones); de dos unidades (sse/streaming
y tts) no se conserva la ficha función por función, sí sus veredictos y sus
arreglos. Un escéptico por unidad intentó tumbar cada hallazgo.

**Revisión de los últimos arreglos.** Dos unidades más, por commit y no por
función: una revisión adversarial del commit de voz (`2259ba2`) y otra de los
dos de apagado (`09aeedb` y `f460fc0`), cada una hecha por alguien que no los
escribió y con su escéptico (§11).

Entre las 56 unidades: 269 veredictos, de ellos 151 refutados, 101
confirmados y 17 decisiones tuyas (una, la del número de lista pegado, se
aplicó después: §7.1). Los 101 confirmados están arreglados con test. Dos
hallazgos bajos de una unidad se quedaron sin veredicto (§9).

**Revisiones transversales.** Siete: la voz de punta a punta (los dos loops
con audio simulado, comparados con el árbol anterior a la auditoría), config ↔
código ↔ docs, ciclo de vida y apagado, el corral de seguridad, el coste en
reposo, empaquetado e instalación, y las conexiones a la nube.

**Cortes.** El trabajo se interrumpió más de una vez por un límite de gasto de
la cuenta, que cortó lo que estaba corriendo; las revisiones que se perdieron
así (tres transversales y un arreglador) se repitieron desde cero.

**Tests.** `pytest --collect-only -q`: 769 en `f316103` y 3866 en `b81e1c2`
(mismo número con Python 3.11 y 3.12); de 49 a 121 archivos de test. En
`b81e1c2`, aquí: 3864 pasan y 2 se saltan en los dos intérpretes; ruff y mypy
limpios. El diff, sin contar este informe: 188 archivos, +65890 / −1804
líneas, de las que +14656 / −1630 son código (76 archivos `.py` y `.sh`) y
+50675 son tests (99 archivos).

**El CI de GitHub estuvo en rojo y no se miró.** Los siete primeros pushes del
20 de noviembre (runs 51 a 57: `2f82ce6`, `6ac95b2`, `09aeedb`, `2259ba2`,
`f460fc0`, `22e52d2` y `601f02f`) fallaron en los dos jobs, y quien coordinó
no lo vio hasta el cierre. El último verde era el run 50 (`0c6d1d9`, del 14 de
noviembre). Causa, comprobada en los logs de los 14 jobs de esos runs:
`ubuntu-latest` pasó a la imagen ubuntu-26.04, cuyo PortAudio trae la API de
PulseAudio y no inicializa sin servidor de audio. Ahí `import sounddevice`
revienta, y `tests/test_tts_stop.py` (anterior a la auditoría) lo importaba al
cargarse: error de colección, y pytest paraba en 4 a 8 s sin correr ningún
test. No lo causó ningún commit de la rama. La consecuencia sí es de este
trabajo: en esos siete commits, cada "suite verde" de los mensajes de commit
fue solo la corrida local. De ellos el CI solo validó ruff y mypy (pasaron en
los 14 jobs); no corrió ni un test. Se arregló en dos archivos de test
(`4f64649`), sin tocar el producto: con un doble de `sounddevice` que falla
igual al importarse, aquí pasó de 1 error de colección y 0 tests a 3864 que
pasan.

Con eso, el run 58 (`b81e1c2`) corrió la suite entera por primera vez desde el
run 50 y salió rojo otra vez: 3862 pasan, 1 falla y 3 se saltan, en los dos
jobs. Fallaba `test_find_in_project_recorta_una_linea_kilometrica`, un test de
esta auditoría (`075187c`) que daba por hecho el orden en que `grep -r` recorre
una carpeta. Ese orden lo pone el sistema de archivos: en la imagen vieja del
runner y en este contenedor salía de una forma; en ubuntu-26.04, al revés.
`find_in_project` hace lo mismo en los dos sitios; lo que estaba mal era el
test. Lo corrige `4d0d7c9`, que solo toca ese test; antes de subirlo, la suite
entera pasó con un doble de `grep` que devuelve los archivos al revés: ningún
otro test tenía esa suposición.

El run 59 (`4d0d7c9`) es **verde** en los dos jobs: ruff, mypy y pytest, con
3863 que pasan y 3 que se saltan en Python 3.12 y en 3.11. Es el primer verde
desde el run 50 y la única vez que el CI ha corrido la suite entera sobre el
código final. Simular aquí el fallo del runner no bastó para predecir el CI:
el segundo fallo solo lo enseñó el runner de verdad. El juez es el CI, no la
corrida local.

**Cierre.** Los commits `2259ba2` (voz), `09aeedb` y `f460fc0` (apagado)
tuvieron una revisión adversarial posterior al arreglo: lo que confirmó está
arreglado (`22e52d2`, `3993c47`), la decisión que planteó se aplicó
(`b81e1c2`), y cuatro hallazgos refutados y dos residuos de gusto quedan
como están (§11, §7.1). A media revisión preguntaste si lo que se estaba
haciendo estaba bien. La respuesta honesta fue "a medias": lo pedido estaba
cumplido, y revisar los arreglos de los arreglos ya rendía cada vez menos. Se
acordó no abrir rondas nuevas: terminar lo que estaba corriendo y cerrar. Así
se hizo. Por eso los arreglos de `22e52d2` y `3993c47` y los cambios de tests
de `4f64649` y `4d0d7c9` no tienen revisión propia posterior, y el de
`b81e1c2` tiene una sola ronda, de dos verificadores (§11.4).

## 2. Lo más grave que estaba roto (y ya no)

Todo lo de esta sección está **[ARREGLADO]**, con test que fallaba antes.

### Seguridad

- **La confirmación no decía qué iba a tocar.** Oías "La acción 'write file'
  puede ser destructiva, patrón. ¿La confirmo?" igual para una nota que para
  sobrescribir `~/.bashrc` o mover una llave de `~/.ssh` a la zona libre.
  Ahora nombra la ruta real (con enlaces resueltos y el aviso "fuera de la
  zona libre"), la ventana, la combinación de teclas o el comando. `2f82ce6`
- **La confirmación por voz tomaba preguntas por un sí.** "¿Se va a borrar
  todo?" ejecutaba la acción (por el "va"). Ahora cualquier pregunta, el "va a"
  de futuro (`124ecf4`) y el "si" condicional (`4d2b642`) cuentan como no; y la
  respuesta se transcribe con el modelo de órdenes, no con el `tiny` del wake
  (`91f1394`).
- **Huecos en el corral.** Un URI `file:///...` o una ruta con un espacio
  delante pasaban sin clasificar (el guard es la única barrera de las tools
  MCP), `075187c`; también `file:etc/passwd` sin barra, `0354cea`. Las rutas
  relativas de una tool MCP se juzgaban contra la carpeta de Crotolamo y no
  contra el `cwd` de su server, `7fef3a0`. Una raíz relativa o vacía en
  `[paths]` abría el disco entero si se lanzaba desde `/`, `075187c`.
- **Contraseñas a la libreta y a la nube.** "Acuérdate de que la contraseña del
  wifi es…" se guardaba en SQLite y el pre-hook la reenviaba a la nube. El
  filtro de secretos dejaba pasar clave/NIP/PIN, contraseñas cortas y tokens
  tipo `AKIA…` o `ya29.…`. `fd8fe45`. Lo que ya estuviera guardado no se borra
  solo (§8, paso 11). El filtro sigue siendo una heurística con huecos
  conocidos, comprobados en `b81e1c2`: "mi contraseña es gato dos mil
  veinticuatro", "mi contraseña es gato 2024", "la contraseña del wifi es la
  casa azul" o un número de tarjeta con su CVV SÍ se guardan; y como ahora los
  hechos van anclados, lo que se cuele viaja en cada petición a la nube. No le
  dictes secretos (D23).
- **Tu key de OpenAI a un tercero.** Con `[memoria]` activa, sin key de GLM y
  con `OPENAI_API_KEY` en el entorno, mem0 mandaba esa key a la URL de Z.ai en
  cada extracción. `0354cea`
- **El doctor podía imprimir tu API key.** Pegada en dos líneas, salía entera
  en el mensaje de error; y la sonda de la nube seguía redirecciones llevando
  el Bearer a otro host. `b77dae1`
- **SSRF.** `read_page` trataba 100.64.0.0/10 (CGNAT, Tailscale) y fec0::/10
  como internet pública. `18fb058`
- **MCP.** Un typo en `confirm` ("allways") heredaba el valor por defecto,
  quizá más débil; ahora cae a "always". `6508165`

### Pérdida de datos

- **Los hechos se evaporaban.** Lo que guardabas con "acuérdate de que…" se
  inyectaba una sola vez, en el primer mensaje: desaparecía en el turno 21 y
  tras `/reset`, y el daemon de voz nunca reinicia la sesión. Ahora van anclados
  en cada petición. `bdb4e6b`
- **Borrar un acceso directo borraba el archivo real.** `delete_file` y
  `move_file` actuaban sobre el destino del enlace (venía del arreglo de
  septiembre). Ahora actúan sobre el enlace. `0354cea`
- **`create_note` pisaba** sin avisar una nota con el mismo título, y
  `write_file` dejaba el archivo en 0 bytes si el contenido no se podía
  codificar. `075187c`
- **"Olvida lo de X" borraba otra cosa.** Con la memoria semántica caída, el
  respaldo SQLite borraba el hecho más parecido aunque no tuviera que ver.
  `fd8fe45`, afinado en `0354cea`. Con mem0 arriba sigue pasando: es D22.
- **Reiniciar el servicio mataba tus apps.** Lo que abrías por voz vivía en el
  cgroup de `crotolamo.service` y systemd lo mataba al reiniciar. `18fb058`
- **La última extracción de memoria se perdía al salir**: al salir normal
  (`fd8fe45`, `4d2b642`) y al cerrar la ventana de `crotolamo shell` (en una
  pty real: 0 de 8 la conservaban, ahora 24 de 24; `f460fc0`). Ese arreglo dejó
  un caso fuera, que cazó la revisión posterior: cerrar la ventana justo
  después de "salir" (SIGHUP 0.2 s tras el "apagado": extracción perdida;
  ahora, guardada; `3993c47`, §11). Y, ya dentro de la auditoría, al parar el
  servicio durante la pausa de extracción que ella misma añadió (`bdb4e6b`):
  2.0 s perdidos y nada extraído; ahora 0.001 s y extraído (`f460fc0`; ver §4).
- **El panel te borraba `listener.env`.** Cambiar el modo lo reescribía desde
  cero, sin escritura atómica. `e34cf9e`

### Fallos silenciosos

- **"Listo, patrón" sin hacer nada.** Con Ollama en streaming se perdían los
  tool_calls: la tool no corría y el turno cerraba como si sí. `200881d`. Te
  tocaba en el loop simple cuando respondía el modelo local.
- **Respuestas rotas que pasaban por buenas.** Un stream de la nube con un
  error dentro, cortado o redirigido se aceptaba (`200881d`); también un
  `finish_reason` de fallo de inferencia (`0354cea`, `7edde4f`), y un tool-call
  truncado corría con argumentos vacíos (`7edde4f`).
- **Errores que no se oían.** Un fallo del LLM no se hablaba ni quedaba en el
  historial; una respuesta vacía se decía "Listo, patrón". `f87ca9b`
- **"pon música" llegaba como "música"** cuando el wake se reconocía de forma
  aproximada. `124ecf4`
- **Loop concurrente sordo.** Una excepción del micrófono mataba el hilo del
  oído: servicio vivo, sordo, y el panel diciendo "escuchando" (`269196e`,
  `4d2b642`). Una pausa tras el wake cerraba una captura vacía y la orden se
  perdía (`269196e`). Y un wake real disparaba una ráfaga de turnos fantasma: 36
  medidos con el modelo real (`124ecf4`).
- **Sin internet era inusable.** El turno que descubría la caída mandaba las 42
  tools al modelo local (~5000 tokens contra una ventana de 2048, que Ollama
  recortaba en silencio, persona incluida) y cada intento a la nube esperaba 60
  s por dirección. `bdb4e6b`
- **Config que tumbaba todo sin decirlo.** Una API key pegada con comillas
  tipográficas o un espacio invisible reventaba cada turno; `keep_alive = "-1"`,
  que el propio toml sugería, hacía que Ollama rechazara cada petición
  (`7edde4f`). Un typo en cualquier clave de `[memoria]`, aun apagada,
  arrancaba a Crotolamo SIN tools (`0354cea`).
- **Diagnóstico que mentía.** El doctor salía verde sin poder importar Piper
  (asistente mudo) y reventaba fuera de la raíz del repo (`a9a762d`). El HUD
  seguía en HABLANDO si el listener moría (`e34cf9e`). Las cámaras respondían
  "cero eventos" a un nombre mal oído (`0354cea`).
- **Instalación.** `install.sh` copiaba units atadas a `~/Documentos/crotolamo2`
  y nunca instalaba la del HUD; `aec.sh` daba la fuente AEC por ausente bajo
  `pipefail`. `3f7a26d`
- **MCP.** Un server muerto seguía anunciando sus tools, y el timeout no cubría
  la escritura: un server colgado bloqueaba el turno para siempre. `6508165`
- **Ventanas de Hyprland.** Las tools fallaban tras cada login porque el
  servicio arranca antes que Hyprland. `0354cea`

Del backlog de septiembre, las tres ALTA quedaron cerradas: V1 (`269196e`), V2
(`269196e` + `4d2b642`) y V4 (`b4ac3fe` + `2f82ce6`). Las que allá estaban
marcadas [DECISIÓN] siguen siéndolo y están en la §7.

## 3. Rendimiento

Nueve causas raíz. En cada tabla, la última columna dice si la mejora va
encendida de fábrica. Cómo leer el método: **bench** = harness con dobles en
los bordes y, donde se indica, reloj virtual (los tiempos son los del modelo
supuesto, no los de tu red); **modelos reales** = Whisper, Silero, openWakeWord
y Piper de verdad en esta máquina sin GPU, con micrófono y altavoz falsos;
**loopback** = el cliente real de GLM contra un servidor local con TLS.

Ojo: "tu modo" en estas tablas es `use_oww = false` con los modelos de Whisper
de FÁBRICA (`base` para la orden, `tiny` para el wake). Según tu
`VALIDACION_MIC_CROTOLAMO.md`, tú corres `small` + `base` (~+425 MB de RAM):
tus cifras absolutas de RAM, de arranque y de CPU por pasada de Whisper serán
más altas; lo que debería mantenerse es la diferencia entre antes y después.

### 3.1 El turno del LLM era una sola llamada bloqueante

El turno se diseñó como petición/respuesta y el streaming se añadió después,
como un callback que cruza cinco capas. Ninguna era dueña de la respuesta
mientras llegaba: se tiraba la conexión en cada llamada, no había forma de
cancelar un turno y, con GLM (que ve tools en todos los turnos), la respuesta se
retenía entera hasta el final.

Qué cambió: el transporte reutiliza la conexión tras un stream; un consumidor
que aborta cuelga el socket; hay cancelación entre tokens y antes de cada tool;
y los dos loops preparan la conexión con la nube en segundo plano con la primera
voz de la orden (también la del seguimiento). `e536618`, `fcd68c8`, `0c6d1d9`,
`6ac95b2`, `7edde4f`.

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Conexiones TCP+TLS nuevas en 6 llamadas en streaming seguidas | 6 | 1 | bench, sockets reales | ON |
| Lo mismo en turnos de voz separados 20 s | 6 | 4 | bench (commit `e536618`) | ON |
| Socket abierto tras abortar un stream (la nube sigue generando) | 1 | 0 | bench | ON |
| Apretones TCP+TLS dentro del turno, 3 órdenes, loop simple | 1, 1, 1 | 0, 0, 0 | loopback | ON |
| Lo mismo, loop concurrente | 1, 0, 0 | 0, 0, 0 | loopback | ON |
| Seguimiento tras una respuesta hablada: apretones en el turno | 2 | 0 | loopback | ON |
| CPU de la primera petición con conexión de repuesto (TLS 1.3) | 21 ms | 1 ms | loopback | ON |
| Contexto TLS por conexión HTTPS nueva | ~20 ms de CPU cada vez | una vez | loopback | ON |
| Primera palabra, charla de 3 frases con GLM | 1.7 s | 0.6 s | bench, reloj virtual (400 ms al primer token, 30 tok/s) | OFF (`stream_release = "sentence"`) |
| Primera palabra tras una tool / con el respaldo local | 2.1 s / 11.5 s | 1.7 s / 9.3 s | igual | OFF (igual) |
| Tras un barge-in, lo que gasta el turno viejo | 1 llamada, 40 tokens, 1 tool | 0 / 0 / 0 | bench con los hilos reales | OFF (`barge_in_cancel`) |

Lo honesto: los apretones no desaparecen, pasan a segundo plano (uno por
orden). Y con la config de fábrica la primera palabra con GLM suena igual que
antes: en el bench la primera frase está en el cable a los 0.6 s y llega a la
voz a los 1.7 s, antes y después. Adelantarla es justo lo que hace `"sentence"`,
con su precio (§6).

### 3.2 Dos pipelines de voz, y un micrófono por llamada

No había un pipeline de voz sino dos loops completos que crecieron por
separado. El simple (el tuyo) abre y cierra el micrófono en cada grabación, y
entre dos grabaciones no oye nadie. Cada mejora se le añadió al loop en que se
estaba trabajando y nunca se portó al otro.

Qué cambió: existe `crotolamo/voice/capture.py` (un micrófono abierto por
sesión), un solo camino de habla y un solo detector de fin de frase. Casi todo
entra apagado. `27af075`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Aperturas del micrófono por minuto de silencio, loop simple | 10 | 0 | bench, reloj virtual | OFF (`audio_session`) |
| Órdenes que llegan enteras (4 dichas tras el wake) | 3 de 4 (se pierden 332 y 51 ms) | 4 de 4 | igual | OFF (igual) |
| Audio perdido al seguir tras una pausa de pensar | 377 ms | 0 | igual | OFF (igual) |
| Micrófono caído un minuto: líneas en pantalla / reaperturas | 119 / 119 | 2 / 11 | igual | ON |
| Primera palabra en el loop concurrente | 3009 ms | 2049 ms | bench, reloj virtual, LLM palabra a palabra | OFF (`stream_speak_concurrent`) |
| Silencio entre frases de una respuesta de 3 | 85 + 91 ms | 0 | bench, Piper simulado a 15 caracteres/s | OFF (`tts_presynth`) |
| Órdenes completas tomadas por "a medias" (+2.5 s de espera), de 40 de prueba | 26 | 19 | la heurística real sobre la lista del bench | ON |

Lo que sí va encendido en tu modo: el error del micrófono ya no inunda la
terminal (reintenta a 1, 2, 4… hasta 30 s; a cambio, tras un corte largo puede
tardar hasta 30 s en notar que el micrófono volvió), dos hilos ya no hablan uno
encima del otro, y 7 de las 26 frases completas que esperaban 2.5 s de más ya
no esperan ("sí", "¿por qué?", "dime cómo"…; arreglos de `124ecf4` y
`b77dae1`). Las 19 que quedan son D6.

### 3.3 No hay un detector de wake a la medida

Para "crotolamo" no existe un detector barato y validado. El modelo dedicado
es un puente ("hey jarvis") que se dispara con la propia voz de Piper, así que
lo apagaste y el trabajo lo hace Whisper `tiny` por ventanas: su coste es por
llamada y delante solo tiene un chunk de VAD.

Qué cambió: openWakeWord recibe frames completos, el detector se rearma tras
una pausa del panel, el diario deja de guardar lo que se oye en casa, y hay un
hueco para un pre-filtro delante de Whisper (apagado). `27af075`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Predicciones de openWakeWord por minuto (concurrente) | 1875 | 638 (openwakeword 0.4; en 0.6: 762) | bench, modelo real, mismos scores | ON |
| CPU de esas predicciones, por segundo de audio | 0.036 s | 0.023 s | modelo real (openwakeword 0.4; en 0.6: 0.035 → 0.027) | ON |
| Turno fantasma al reanudar tras una pausa del panel | 15 de 15 | 0 de 15 | modelo real, audio sintético | ON (`oww_rearm`) |
| Líneas del diario con lo que se oye, por hora de tele (wake difuso) | 720 | 0 | bench | ON |
| **Pasadas de Whisper por minuto de tele (tu modo)** | **12** | **12** | bench | igual |
| Pasadas de Whisper por minuto de toses e interjecciones | 10 | 10 | bench | igual |
| Pasadas por minuto de tele con pre-filtro | 12 | 0 | bench, con "hey jarvis" como sustituto | OFF (`pregate`) |

Esto es lo más importante que NO se arregló: el coste en reposo de tu modo con
voces en el cuarto es el de antes (la medición inicial lo puso en 13-16
segundos de CPU por minuto de tele). El pre-filtro solo sirve con un modelo que
oiga tu "crotolamo", y eso no se puede entrenar ni validar sin tu voz (D14).

### 3.4 Modelos consumidos con el envoltorio del fabricante

Cada modelo se integró con el paquete de conveniencia de su fabricante y sus
valores por defecto, pensados para scripts cortos, dentro de un servicio 24/7.
Un VAD de 2.3 MB arrastraba torch; importar openWakeWord traía sklearn y scipy
para una función de entrenamiento; Piper se quedaba para siempre con la memoria
de la frase más larga; Whisper le preguntaba al Hub en cada arranque.

Qué cambió: `crotolamo/voice/runtime.py` es el único dueño de cómo se cargan.
Silero corre por onnxruntime con el mismo archivo `.onnx` (salida idéntica al
bit); si una versión futura no encaja, cae sola al cargador de siempre con un
aviso. `d082104`, `c892d1f`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| RAM del servicio tras una orden, tu modo (simple difuso) | 872 MB | 683 MB | modelos reales, proceso nuevo por escenario | ON |
| Lo mismo, loop concurrente | 980 MB | 620 MB (578 MB desde `27af075`, que ya no carga el Whisper `tiny`) | igual | ON |
| Módulos importados en el hilo de audio en la primera orden | 953 (torch entre ellos) | 17 | igual | ON |
| Sesiones ONNX creadas para cargar un modelo de wake | 9 | 3 | igual | ON |
| Consultas al Hub de Hugging Face por arranque | 2 | 0 | igual, red bloqueada y contada | ON |
| Despertares al día del hilo monitor de tqdm | 8640 | 0 | igual | ON |
| Los cinco checks de voz del doctor | ~4 s | ~0.3 s | igual (en `d082104`) | ON |
| Memoria que Piper retenía tras UNA respuesta larga | ~409 MB | ~7 MB | modelos reales | ON (`piper_arena_keep_phonemes = 256`) |

El comando `crotolamo doctor` entero, cronometrado aquí en `b81e1c2` (sin key,
o sea sin la sonda a la nube): de ~3 s a menos de 1 s. Con la máquina
tranquila, 0.4-0.5 s con tu `use_oww = false` y 0.6-0.9 s con la config de
fábrica, porque desde `27af075` el check voz-oww carga el modelo de wake para
avisar si no carga.

Precio del último renglón de la tabla: las frases de más de ~115 caracteres se
sintetizan ~14 % más lento. Y lo que se quitó es torch en ejecución; en disco
sigue instalándose (§10.1).

### 3.5 Nadie era dueño del tamaño entre la tool y el cable

Los límites existían solo como conteos (turnos, tools, caracteres), cada uno
en su capa y aplicado al final. Una tool podía devolver cientos de miles de
caracteres, el historial guardado ERA lo que viajaba, y con el modelo local el
request no cabía en la ventana y Ollama cortaba el principio del prompt.

Qué cambió: `crotolamo/core/budget.py` es el dueño de las unidades. Tope a todo
resultado de tool, ajuste de cada request a la ventana del motor que responde
(sin tocar el historial), web con gzip y plazo total, hechos anclados. `bdb4e6b`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Requests al modelo local que no cabían en `num_ctx = 2048` | 29 de 33 | 0 | bench, sesión simulada | ON |
| Primer request al local tras caer la nube | 42 tools, 51.8 KB | 2 tools | igual | ON |
| Turnos en que el modelo ve tus hechos guardados | del 1 al 20 | todos | igual | ON (`pin_facts`) |
| Descarga de una página (artículo sintético) | 1.3 MB | 14.9 KB | bench, servidor local | ON |
| Resultado de tool más grande del bench | 170010 caracteres | tope de 24000 | bench | ON |
| Bytes a la nube en 30 turnos | 2.16 MB | 1.46 MB | bench | OFF (`stale_tool_turns = 2`) |
| Prefijo reutilizable entre turnos | 39.8 % | 88.8 % | bench | OFF (`trim_chunk = 7`) |
| Un "hola" tras 8 horas | 45.8 KB | 17.8 KB | bench | OFF (`idle_reset_min = 30`) |

Con la config de fábrica y la nube sana, lo que viaja por turno es
prácticamente lo de antes (2.15 → 2.16 MB en los turnos 31-60 del bench; la
diferencia es el bloque de hechos). El ahorro real en la nube está en las tres
claves apagadas, y la de `trim_chunk` depende de que Z.ai cachee prefijos, cosa
que no se pudo comprobar.

### 3.6 El estado se preguntaba en vez de notificarse

Lo que vive fuera de la llamada en curso se obtenía volviendo a preguntar: un
`playerctl` cada 1.5 s para siempre, hilos que despertaban cada 0.2-0.3 s solo
para ver si había que parar, un `stat` de `control.json` tres veces por
segundo. Había fuentes de eventos para todo.

Qué cambió: un `playerctl --follow` vivo avisa de cada cambio; en el loop
concurrente el apagado y el canal de control despiertan a quien espera
(inotify); el panel usa un solo hilo y repinta solo lo que cambia. `265eb22`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Procesos `playerctl` en una hora de reposo, sin nada sonando | 2401 | 14 | bench, reloj virtual, playerctl falso | ON (`media_watch = "follow"`) |
| Procesos `playerctl` en una hora CON música sonando | 2401 | 2402 | mismo bench | igual |
| Tiempo en notar que algo empezó a sonar | hasta 1.5 s | ~15 ms | `playerctl` 2.4.1 real con reproductores MPRIS falsos en un bus D-Bus privado (13-17 ms) | ON |
| Despertares por reloj del loop concurrente en reposo | 1100/min | 2/min | bench | ON (solo concurrente) |
| `stat` de `control.json`, loop concurrente | 200/min | 2/min | bench | ON (solo concurrente) |
| Panel abierto: hilos nuevos / repintados por minuto | 40 / 160 | 0 / 80 | panel real sobre un `gi` falso | ON |
| Panel abierto: `systemctl show` por minuto | 40 | 40 | igual | igual |

El ahorro de `playerctl` solo existe mientras no suena nada: con música, el
monitor sigue lanzando uno cada 1.5 s, igual que antes. Si tienes música casi
todo el día, esta mejora casi no te toca.

Sin tocar: el panel sigue lanzando `systemctl` cada 1.5 s mientras está
abierto; cada URL sigue costando un `flatpak info` (D37); y cada operación de
la libreta abre su conexión SQLite. El escenario del HUD de este bench ni
siquiera corrió: no hay GTK aquí.

Añadido por la auditoría, no heredado: tu loop simple ahora mira `control.json`
tres veces por segundo. Es un hilo vigía nuevo (`91f1394`); antes lo leía una
vez por ciclo de wake, y por eso el interruptor del panel se revertía solo.
Cuesta ~216 despertares y 0.023 s de CPU por minuto (0.073 con la escucha en
pausa). El loop concurrente ya lo hace por inotify; pasarlo también al simple
quedó sin hacer.

### 3.7 El arranque era un guion serie fijado antes de saber el modo

`run_listen` cargaba una lista fija de cosas, en serie y en el hilo principal,
antes de decidir qué loop iba a correr. Cargaba un Whisper que el modo
concurrente nunca usa, y con GLM de cerebro gastaba una petición real a la nube
para "calentar" algo que no hay que calentar.

Qué cambió: primero se decide el modo y luego se calienta solo lo que ese loop
usa, en el orden en que lo usa (Piper primero). Los servers MCP arrancan a la
vez. `c892d1f`, `27af075`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Primer audio del saludo, tu modo (difuso) | 0.79 s | 0.63 s | modelos reales, mediana de 3 | ON |
| Listo para oír el wake, tu modo | 7.21 s | 6.89 s | igual (ruido de ±0.4 s) | ON |
| Lo mismo, loop concurrente | 6.89 s | 5.66 s | igual | ON |
| RAM al quedar listo, loop concurrente | 483 MB | 442 MB | igual | ON |
| Peticiones reales a la nube por arranque, con GLM | 1 | 0 | bench | ON |
| Arranque con 3 servers MCP sanos y 1 roto | 1.69 s | 0.56 s | servers falsos | ON (MCP va apagado) |
| Arranque con 2 servers MCP colgados y 1 sano | 5.56 s | 2.51 s | igual | ON (MCP va apagado) |

En tu modo la ganancia es casi nada (0.3 s, dentro del ruido). Lo que queda
son los segundos del saludo, que bloquea: decirlo corto ahorra 3.61 s en tu
modo (D7).

### 3.8 La política del cerebro se fijaba una vez; el motor se elige cada vez

"Quién es el cerebro" se decidía una vez, al construir el agente; pero qué
motor responde de verdad se decide en cada llamada. Así, el turno que descubría
una caída mandaba al modelo local la petición armada para la nube, y el tiempo
de espera para conectar era el mismo número que el de la respuesta entera.

Qué cambió: política por motor evaluada en cada llamada; al caer la nube el
local recibe las tools enrutadas desde ese mismo turno; tiempo de conexión
separado (5 s por dirección); el enfriamiento cuenta desde que el local ya
respondió. `bdb4e6b`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Primera palabra del turno que descubre la caída (internet en agujero negro) | 497 s | 102 s | bench, reloj simulado | ON |
| Lo mismo con el wifi apagado | 377 s | 97 s | igual | ON |
| Suma de 6 turnos durante la caída | 2330 s | 155 s | igual | ON |
| Intentos fallidos a la nube en esos 6 turnos | 11 | 4 | igual | ON |
| Espera por conectar con la red caída, por dirección | 60 s | 5 s | igual | ON |
| Llamadas al modelo local en 6 órdenes, con atajos | 11 | 6 | igual | OFF (`fallback_shortcuts`) |
| "pausa la música" sin internet, con atajos | 8.4 s | 0.0 s | igual | OFF (igual) |

De los 102 s, 90 son la carga en frío de Ollama. Los tiempos del modelo local
y de esa carga son cifras del propio repo, no medidas aquí (no hay Ollama), y el
bench supone que el DNS contesta al instante.

### 3.9 El bucle del turno corría cada paso opcional siempre

El agente no tenía descripción del turno que estaba corriendo, así que cada
paso caro corría en todos: buscar recuerdos antes y extraer hechos después,
también en "pausa la música"; y varias tools de solo lectura pedidas juntas
corrían una tras otra.

Qué cambió: el turno se describe a sí mismo; las tools de solo lectura pedidas
en un mismo mensaje corren a la vez (hasta 4), con los resultados en el mismo
orden. `bdb4e6b`

| Qué | Antes | Después | Cómo se midió | De fábrica |
|---|---|---|---|---|
| Espera por 3 tools de solo lectura pedidas juntas | la suma (3 unidades) | la más lenta (1) | bench | ON |
| Peticiones al proveedor en la sesión de prueba, con memoria | 30 | 24 | bench | solo con `[memoria]` (va apagada) |
| Búsquedas de recuerdos en órdenes tipo "pausa la música" | 6 | 0 | bench | igual |
| Espera sentida con la memoria lenta, 6 turnos | 9.0 s | 1.5 s | bench | igual |
| Llamadas al LLM cuando el modelo insiste en una tool que falla | 6 (~110 KB) | 3-4 (~39-58 KB) | bench | OFF (`stop_on_repeated_failure`) |

Con tu config (memoria semántica apagada) lo único que te toca de esta causa
es el primer renglón.

## 4. Regresiones que el propio trabajo introdujo y la revisión cazó

Cada una se contrastó con el historial de git antes de llamarla regresión.

**Introducidas por la auditoría y ya arregladas:**

- **Wake difuso, saludo + nombre + orden.** "Buenos días, crotolamo, pon las
  noticias" dejó de ejecutar la orden: activaba y te la volvía a pedir.
  Reproducido en los tres árboles: en `f316103` devolvía la orden, en `124ecf4`
  devolvía vacío, en `b09e63d` vuelve a devolverla. Es tu modo diario.
- **Wake difuso, ventana de seguimiento.** Una frase que solo se parece al
  nombre ("¿cómo me llamo?", "gracias, crotolamo") se borraba entera y la
  conversación terminaba en silencio. Mismo origen (`124ecf4`) y mismo arreglo
  (`b09e63d`).
- **Memoria.** Al caer la nube a mitad de turno, el modelo local perdía los
  recuerdos de ese turno (los había dejado fuera la deduplicación de
  `bdb4e6b`). Arreglado en `7edde4f`. Solo con `[memoria]` activa.
- **Hechos.** Al quitar duplicados (`0354cea`), volver a guardar un hecho sin
  categoría lo reetiquetaba como "general" y `recall_facts("preferencias")`
  dejaba de encontrarlo. Arreglado en `7fef3a0`.
- **MCP.** El "cierre educado" de `6508165` abrió una ventana de 1-6 s en la
  que un server en retirada era invisible para el apagado por señal y
  sobrevivía a Crotolamo (antes eran microsegundos). Arreglado en `0354cea`.
  Quedaba un borde de esa misma espera, que la auditoría ensanchó pero no
  inventó: si un Ctrl-C (o, desde `f460fc0`, el SIGTERM o SIGHUP del shell)
  cortaba el cierre de un server colgado, nadie lo remataba y sobrevivía. Lo
  cazó la revisión posterior del apagado; arreglado en `3993c47`. El Ctrl-C
  durante la salida del shell es otro camino y sigue dejándolo vivo (§9).
- **Apagado.** La pausa de 120 s de la extracción (nueva en `bdb4e6b`) hacía
  que parar el servicio en ese rato esperara 2 s en balde y perdiera lo
  encolado. Arreglado en `f460fc0`.
- **Código nuevo apagado que ya nació con bugs.** Con `stream_release =
  "sentence"`: la respuesta de una tool directa no se hablaba y el texto salía
  pegado ("mismo.Traes", y Piper decía "punto"), `0354cea`; la pregunta de
  confirmación se colaba por delante de lo ya soltado, `2259ba2`. Con
  `pregate`: un pre-filtro roto se rearmaba y avisaba cada 6 s (~600 avisos por
  hora), `b09e63d`.
- **Lista numerada cuyo primer elemento empieza por cifra.** "1. 2 llamadas al
  banco / 2. Comprar leche / 3. Ir al gimnasio" pasó a decir sus números,
  pegados al elemento anterior ("2 llamadas al banco 2.", "Comprar leche 3.");
  antes sonaba sin números. Regresión de `2259ba2`, un arreglo de esta misma
  auditoría; la cazó la revisión posterior de ese commit y vivió hasta
  `22e52d2`. Medido contra el padre de `2259ba2`: de 8000 listas generadas
  que empiezan en 1 diferían 2771; ahora 0 (§11). La misma revisión midió
  que el precio declarado de `2259ba2` (la lista que no empieza en 1 dice sus
  números) sonaba peor de lo declarado: pegados a la línea anterior ("Ir al
  banco a las 10 5."). Desde `b81e1c2` cada número es su propia frase (§7.1).
- **Barge-in tras una pausa del panel** (loop concurrente con `--barge-in`).
  El arreglo de `2259ba2` dejó al detector sin oír durante la pausa y nada lo
  limpiaba al reanudar: el silencio puntuaba alto unos chunks. Con los
  umbrales de fábrica no llegó a abrir un turno falso (0 de 24). Limpiado en
  `b81e1c2` (§11).
- **Docs.** `README_M4` siguió diciendo que el esquema de una tool MCP pasa
  intacto cuando ya se recorta por encima de 4000 bytes. `2f82ce6`
- **Tests y CI.** El CI estuvo en rojo en `d082104`: un test nuevo construía su
  referencia con la firma de openWakeWord 0.4 y la CI de Python 3.11 instala la
  0.6 (`2fa637e`, solo el test). El test nuevo del SIGKILL de MCP fallaba 1 de
  cada ~15 bajo carga por una carrera del test, no del código (`09aeedb`). Lo
  mismo el test nuevo de señales del listener: 1 cuelgue en 3000 corridas
  bajo carga, por una carrera del test. Los del shell tenían el mismo hueco,
  demostrado a propósito con gdb y nunca visto en una corrida (40 de 40 bajo
  carga). Los dos, en `3993c47`. Y un test nuevo de `find_in_project`
  (`075187c`) daba por hecho el orden en que `grep` recorre una carpeta:
  pasaba aquí y falló en el runner nuevo de GitHub, la primera vez que el CI
  volvió a correr la suite (`4d0d7c9`). El rojo del CI del 20 de noviembre
  por `sounddevice` no está en esta lista: no lo causó la rama (§1).
- **Dos mensajes de commit dijeron más de lo que hicieron.** `b4ac3fe` dice que
  quita `_play_interruptible` y no lo quitó (lo quitó `b09e63d`); `a9a762d` dice
  que `stt_harness` ya no suelta traceback sin micrófono y solo le cambió las
  hotwords (lo arregló `dafbb54`).

**Introducidas por la auditoría y todavía abiertas** (son decisiones tuyas):

- `light_control` responde "prendí la luz" con una entidad que no es `light.*`
  (un enchufe, por ejemplo) aunque no pasó nada. Antes decía "no reconoció la
  luz". Viene de `18fb058`. → D40
- Con el loop concurrente (el de fábrica) y sin micrófono, el servicio sale
  con error en menos de 2 s (1.66 y 1.79 s medidos) y systemd lo reinicia cada
  ~5 s para siempre, cargando modelos y diciendo el saludo cada vez. Antes se
  quedaba vivo y sordo. Viene de `269196e` + `4d2b642`. Tu loop simple no
  sale: reintenta dentro del proceso. → D12
- "Muéstrame el estado del sistema" ya no ofrece las tools de sistema con el
  modelo local. Antes funcionaba de chiripa ("ram" casaba dentro de
  "muestRAMe"); las otras siete frases con "muéstrame" ya fallaban. Viene de
  `f87ca9b`. → D29
- Con openWakeWord 0.6 (Python 3.11), tras cada reinicio del detector el score
  sube hasta ~0.5 con puro ruido de cuarto durante ~1.2 s. El reinicio es
  nuevo (`124ecf4`) y necesario. Con el umbral de fábrica no dispara; por debajo
  de 0.5, sí. → D16
- La confirmación por voz ahora lee la ruta literal ("barra Documentos barra…")
  y la `~` no se oye. Viene de `2f82ce6`. → D1
- El apagado por señal espera más que antes: la auditoría le sumó el candado
  del seguidor de música (0.5 s), el vaciado de la memoria (hasta 2.5 s) y el
  candado del HUD (1 s), hasta ~4 s en el peor caso (y hasta 2 s más si un
  server MCP ignora el SIGTERM, recortados como servicio para no pasar de 8 s).
  Con tres reproductores atenuados, el D-Bus colgado y la memoria activa, pasa
  de 7 s a 11 s y supera el `TimeoutStopSec=10` del unit. → D20
- Una lista numerada que no empieza en 1 dice sus números, cada uno como
  frase corta aparte ("Reinicia el equipo", "4.", "Ir al banco a las 10").
  Antes los callaba. Viene de `2259ba2`, como precio declarado de no perder un
  número que es la respuesta. → §7.1
- Costes asumidos, no errores: retirar un server MCP colgado, o uno que falla
  al arrancar, tarda 1 s más por el cierre educado; y tu loop simple tiene un
  hilo nuevo que mira `control.json` tres veces por segundo (§3.6).

**Encontradas tarde pero que ya existían antes de la auditoría** (no son
regresiones): el bloque de código leído en voz alta línea a línea (`c46cbdc`);
la pausa del panel que no cortaba la ventana de seguimiento del loop simple
(`2f82ce6`); el número que se perdía al inicio de una frase, el barge-in que
abría una orden con la escucha en pausa y el HUD en "pensando" durante la
confirmación (`2259ba2`); las líneas de `faster_whisper` en el diario
(`f460fc0`).

**Un caso que un arreglo de la auditoría dejó sin cubrir** (tampoco es
regresión: antes de la auditoría la última extracción ya se perdía al salir,
§2): cerrar la ventana de `crotolamo shell` justo después de "salir".
`f460fc0` cubrió el cierre con el shell en el prompt; este lo midió y lo
cerró la revisión posterior del apagado (`3993c47`, §11).

## 5. Lo que vas a notar al usarlo

Con la config de fábrica más tu `use_oww = false` (loop simple, wake difuso).

**Lo que oyes**

- Una orden dicha tras un wake mal oído llega entera ("pon música", no
  "música").
- "Sí", "¿por qué?", "¿a dónde?" o "dime cómo" ya no esperan 2.5 s de más.
- Antes de una acción delicada oyes qué va a tocar, con la ruta leída tal cual
  (D1). Si contestas con una pregunta, con "se va a…" o con un "si…"
  condicional, cuenta como no. El sí tiene que ir al principio: "sí" o
  "confirmo" valen; "claro que sí", que antes valía, ahora obliga a repetir.
- Un número al principio de una frase se dice. Una lista numerada que empieza
  en 1 sigue sin leer sus números. Una que no empieza en 1 ahora sí los lee,
  cada uno como frase corta aparte: "Te faltan estos:", "3.", "Reinicia el
  equipo", "4."… (unos 0.3 s más de audio por número). Con un número en
  negrita ("**3.** Reinicia") suena pegado a la línea anterior, igual que
  antes de la auditoría (§7.1).
- Un bloque de código ya no se lee línea a línea. Si la respuesta es solo
  código, no se oye nada (D4).
- Si el LLM falla, lo oyes. Si la nube se corta a media respuesta sin haber
  hablado, contesta el modelo local; si ya había hablado, oyes el aviso como
  frase aparte.
- Una respuesta vacía se dice "Me quedé en blanco, patrón", no "Listo, patrón".
- "25°C" se dice con grados.
- `remember_fact` rechaza lo que parece contraseña ("eso no lo guardo ni a la
  mala"). Si usabas la libreta para eso, es un cambio que quizá quieras
  deshacer. No lo caza todo: una contraseña dictada en palabras o un número de
  tarjeta todavía se guardan (§2).
- Los hechos guardados siguen presentes tras muchos turnos.
- Pedirle leer un PDF, una imagen o un `.docx` da una negativa de una línea, no
  basura. `search_files` encuentra una nota por su título hablado y sin
  depender de acentos. "Abre terminal" conoce Ptyxis. `system_status` dice la
  carga. Con dos reproductores de la misma app, la orden va al que suena.
- `read_page` ya no lee direcciones 100.64.x.x (Tailscale): las trata como red
  interna, igual que 192.168.x.x.
- Pausar "Escuchar por voz" en el panel surte efecto al momento, también en la
  ventana de seguimiento.
- El saludo empieza un pelo antes (0.79 → 0.63 s aquí).
- Sin internet se puede usar: lento por Ollama, pero ya no minutos por turno.

**HUD y panel**

- Al empezar un turno el HUD ya no enseña la respuesta anterior bajo
  ESCUCHANDO, y durante una confirmación muestra HABLANDO y luego ESCUCHANDO.
- Si el listener muere, el HUD no se queda en HABLANDO.
- Los interruptores del panel no rebotan tras el clic; los fallos se ven; si el
  modo no se puede guardar, no reinicia el servicio y dice por qué.

**Diario (journal)**

- Ya no guarda en INFO lo que se oye en casa. Del wake difuso quedan las
  activaciones y los casi-aciertos, con las tres primeras palabras: `wake
  difuso: score=0.63 (umbral=0.72) cabeza='con todo el' -> no`. Todo, como
  antes, con `CROTOLAMO_WAKE_DEBUG=1`.
- Sin las ~12 líneas por minuto de `faster_whisper` ni el "Esperando palabra de
  activación..." cada 6 s.
- Líneas nuevas al arrancar: `voz caliente (...)` y, con GLM, `cerebro en la
  nube: no hay LLM que calentar`.
- Avisos nuevos que conviene conocer: `GLM cerró el stream sin [DONE] ni
  finish_reason` (D31), `el request para Ollama no cabía en num_ctx=…` (solo
  cuando responde el local, D25) y `stream_release: 'sentence' habría hablado un
  preámbulo de N caracteres` (el dato para decidir esa clave).

**Doctor e instalación**

- `crotolamo doctor` corre desde cualquier carpeta, tarda menos de 1 s (antes
  ~3 s) más lo que tarde la nube en contestar la sonda, pone `tts` en rojo si
  Piper no importa y rechaza las mismas keys raras que el runtime. Sigue dando
  un rojo falso de `llm-key` si lo corres en un shell sin la key (D43): usa
  `./launch.sh doctor`.
- `install.sh` escribe la ruta real del repo en los units e instala el del HUD.
  El servicio pasa a `Type=exec`: si el `.venv` no existe, `systemctl start`
  falla de verdad.
- `install.sh`, `launch.sh`, los README, `pyproject.toml` y las pistas de
  silero-vad (doctor y `adapters.py`) llevan `--extra-index-url
  https://download.pytorch.org/whl/cpu` (sin él, pip bajaba ~3 GB de CUDA).
  Nueve sitios siguen diciendo `pip install -e '.[voice]'` a secas: seis pistas
  del doctor, el mensaje y el docstring de `VoiceUnavailable` en `stt.py` y el
  mensaje de `wakeword.py`. En un venv nuevo, seguir esas pistas todavía baja
  la build de CUDA; añade el índice a mano.
- El menú de `launch.sh` ya no muere cuando una opción sale con error.

**Apagado**

- `systemctl stop`, Ctrl-C y cerrar la ventana de kitty dejan el HUD en reposo,
  la música a su volumen y ningún server MCP vivo. Lo que abriste por voz
  sobrevive a un reinicio del servicio (sin comprobar bajo systemd real, §8).
- Un server MCP que ignora el SIGTERM ya no sobrevive al listener: recibe
  SIGKILL 2 s después. Como servicio esa espera se recorta para no pasar de
  8 s desde que empezó el apagado (`TimeoutStopSec=10`). `09aeedb`, `f460fc0`
- En `crotolamo shell`, cerrar la ventana justo después de "salir" ya no
  corta la salida: el shell ignora ese cierre hasta terminar de guardar y de
  cerrar los servers MCP (la salida tiene tope). Solo se nota con `[memoria]`
  o MCP activos. `3993c47`
- Parar el servicio puede tardar algo más que antes, y con varios
  reproductores atenuados y el D-Bus colgado puede pasar del tope (D20).

## 6. Claves nuevas de config

De `git diff f316103..b81e1c2 -- config/crotolamo.toml`. "Sin clave" = va
comentada y rige el valor por defecto del código. Los arreglos del cierre
(§11) no añadieron ni cambiaron ninguna clave.

| Clave | De fábrica | Qué hace | Ganancia medida | Riesgo / por qué va así |
|---|---|---|---|---|
| `[paths].relative_base` | sin clave (el directorio de arranque) | Dónde caen las rutas relativas que manda el modelo | ninguna de velocidad; predecibilidad | Con `"~"` cada ruta relativa pide confirmación, y en el loop concurrente no corre (D36) |
| `[llm].connect_timeout` | sin clave = 5 s | Tope para conectar con Ollama | un host que no contesta: 180 s → 5 s | ON; con Ollama local no se nota |
| `[llm].stop_on_repeated_failure` | sin clave = false | No repetir en el mismo turno una llamada idéntica que ya falló | 6 llamadas (~110 KB) → 3-4 (~39-58 KB); "dímela por el shell" 1 vez, no 6 | Cambia lo que oyes: el modelo explica el fallo con sus palabras |
| `[llm.glm].connect_timeout` | sin clave = 5 s | Tope para conectar con la nube | red caída: 60 → 5 s por dirección | ON; con un enlace muy lento puedes caer al local sin motivo (súbelo a 10) |
| `[llm.glm].max_tokens` | sin clave | Tope de tokens de salida. El código ya la leía; solo se documentó | — | Si corta, la frase queda a medias |
| `[llm.glm].max_context_tokens` | sin clave = 128000 | Red de seguridad: un request mayor descarta sus turnos más viejos | evita un 400 permanente tras varias lecturas grandes | ON; un turno normal no se acerca |
| `[llm.glm].fallback_shortcuts` | sin clave = false | Atajos del motor local mientras responde el respaldo | 11 → 6 llamadas; "pausa la música" 8.4 → 0 s | Sin internet oyes la frase cruda de la tool |
| `[memory].pin_facts` | true | Hechos anclados tras la persona en cada request | visibles en todos los turnos (antes del 1 al 20) | ON; ~18 tokens por hecho en cada request; con el local a 2048 aprieta (D25) |
| `[memory].trim_chunk` | sin clave = 1 | Al llenarse la ventana, tira N turnos de golpe | con 7: prefijo reutilizable 40 % → 89 % | La ventana oscila 14-20 turnos; el ahorro depende del caché de Z.ai, sin comprobar |
| `[memory].stale_tool_turns` | sin clave = apagado | Resultados de tool viejos viajan como sus primeros 800 caracteres | con 2: −32 % de bytes | Para citar un archivo de hace 3 turnos lo vuelve a leer |
| `[memory].idle_reset_min` | sin clave = apagado | Tras N minutos sin hablar, ventana vacía | con 30: "hola" tras 8 h, 46 → 18 KB | Olvida la charla anterior |
| `[voice].piper_arena_keep_phonemes` | 256 | Las frases largas devuelven la memoria de Piper | ~409 → ~7 MB retenidos | ON; frases largas ~14 % más lentas de sintetizar |
| `[voice].audio_session` | false | El micrófono se abre una vez y un hilo lo lee sin parar | aperturas 10 → 0 por minuto; 4 de 4 órdenes enteras (antes 3 de 4) | Sin validar: suspender, reiniciar PipeWire, desenchufar, AEC |
| `[voice].mic_give_up_s` | 90 | Con `audio_session` y como servicio: salir tras N s de micrófono caído | — | Solo actúa con `audio_session = true` |
| `[voice].barge_in_cancel` | false | El turno interrumpido se corta en vez de terminar en silencio | 1 llamada, 40 tokens y 1 tool → 0 | Solo `--barge-in`; una tool no empezada ya no corre; sin validar con micrófono |
| `[voice].whisper_cpu_threads` | 0 | Hilos de CPU de cada Whisper | sin medir (depende de tu CPU) | 0 = lo de siempre |
| `[voice].stream_speak_concurrent` | false | El loop concurrente habla frase a frase | primera palabra 3009 → 2049 ms | Con GLM no gana nada sin `stream_release`; eco sin validar |
| `[voice].tts_presynth` | false | Piper sintetiza la frase siguiente mientras suena la actual | silencio entre frases 85 + 91 ms → 0 | Sin validar: cortes de audio en tu CPU |
| `[voice].stream_release` | sin clave = "final" | Cuándo se suelta a la voz una respuesta con tools a la vista | con "sentence": primera palabra 1.7 → 0.6 s | Un preámbulo de dos o más frases antes de una tool SÍ se oye |
| `[voice].media_watch` | "follow" | Un `playerctl --follow` vivo en vez de sondear cada 1.5 s | 2400 → ~12 procesos por hora sin música; con música, igual que antes | ON; sin probar con reproductores reales. "poll" = lo de antes |
| `[wake].oww_exact_frames` | false | Frames exactos de 1280 en el loop concurrente | lo que calibras es lo que decide | Con openwakeword 0.4 cambia ~15 % de resultados (de 120: 11 se pierden, 7 se ganan). En openWakeWord 0.5+ no cambia nada |
| `[wake].oww_rearm` | true | Al reanudar tras una pausa olvida el audio de antes | fantasma 15 de 15 → 0 de 15 | ON; solo loop concurrente |
| `[wake].pregate` | "" | Pre-filtro openWakeWord delante de Whisper en el wake difuso | minuto de tele: 12 → 0 pasadas | Necesita un modelo que oiga "crotolamo"; cuesta ~1.4-1.6 s de CPU por minuto |

## 7. [DECISIÓN] Lo que te toca decidir

Cada una: la pregunta, las opciones, qué pasa hoy y qué haría yo. Están
contrastadas con el código en `b81e1c2`; las que se resolvieron por el camino
van al final (§7.11).

### 7.1 Las dos que ya se tomaron por ti (y lo que queda)

**El número al empezar una frase.** Estaba planteada como decisión con tres
opciones: (a) quitar siempre el marcador "N.", (b) quitarlo solo si hay
indicios de lista, (c) no quitarlo nunca. La revisión de voz de punta a punta
la confirmó como defecto (la palabra que se perdía era la respuesta) y se
aplicó la **(b)** en `2259ba2`. Hoy: "42. ¿Algo más, patrón?" suena "42." y
luego "¿Algo más, patrón?"; "1. Abre… 2. Cierra…" sigue sin leer números,
empiece como empiece cada elemento. La (b) no lo salva todo: un número se
pierde todavía si es un "1." que abre la respuesta ("1. ¿Algo más, patrón?")
o si abre una línea después de una lista que empezó en 1 ("…En total son /
42. ¿Algo más?" suena "En total son ¿Algo más?"). En la medición de la
revisión de `2259ba2`, de 27000 respuestas generadas perdían un número 4259
antes del arreglo y 800 después, todas de esas dos formas. Si prefieres (a)
o (c), es un cambio pequeño en `crotolamo/voice/tts.py` (`_Lista`).

**El número suelto que abre una línea.** El precio de lo anterior: una lista
que no empieza en 1 (retomada en "3. …", o con el "1." escrito dentro de la
línea de entrada) dice sus números; antes de la auditoría los callaba. Entre
`2259ba2` y `b81e1c2` además los decía PEGADOS al final de la línea anterior:
"Te faltan estos: 3.", "Reinicia el equipo 4.", "Ir al banco a las 10 5.". La
revisión posterior de `2259ba2` lo planteó como decisión tuya (dejarlo, decir
cada número como frase propia, o volver a callarlos cuando la forma es de
lista) y se aplicó la segunda en `b81e1c2`. Hoy se oye "Te faltan estos:",
"3.", "Reinicia el equipo", "4.", "Ir al banco a las 10", "5.", "Comprar
pan": ningún número se pierde ni se pega, al precio de más pausas cortas.
Con Piper real son unos 0.3 s más de audio por número (una lista de tres
pasó de 8.61 a 9.53 s). Nadie lo ha oído en un altavoz: se midió lo que se
le pide a Piper. Deshacerlo son unas líneas de `_split`, en el mismo archivo.

**Lo que queda es de gusto.** La persona ya le pide al modelo no usar listas
ni negritas; todo esto es para cuando no obedece.

1. **¿Se dicen los números de una lista que no empieza en 1?** Hoy sí. La
   alternativa es volver a callarlos cuando la forma es de lista (la línea
   anterior acaba en ":" o siguen números consecutivos); entonces una
   respuesta numérica escrita con esa misma forma se vuelve a perder. Yo lo
   dejaría como está hasta oírlo.
2. **Listas medio numeradas.** En una lista que no empieza en 1, el elemento
   que empieza por cifra calla su número y los demás lo dicen: "Tu agenda: /
   2. 9:00 reunión con el equipo / 3. 11:30 dentista / 4. Comida con Ana"
   suena "Tu agenda: 9:00 reunión con el equipo 11:30 dentista", "4.",
   "Comida con Ana". Y un número solo en su línea, seguido de otra línea
   numerada, sigue pegado a la anterior ("Te faltan estos: 3."). No es
   regresión de `b81e1c2` (entre `2259ba2` y `b81e1c2` sonaba igual o peor:
   "…11:30 dentista 4."). Frente a antes de la auditoría sí cambia: esa
   agenda sonaba entera y sin números, y ahora suelta un "4." solo. Es parte
   del precio de `2259ba2`. Arreglarlo toca la regla del decimal partido
   ("40. 5 GB"), que se dejó fuera a propósito. Yo no lo tocaría.
3. **Números en negrita.** "**3.** Reinicia" no cuenta como marcador de
   lista: suena pegado a la línea anterior ("Ir al banco a las 10 5."),
   también en listas que empiezan en 1 ("Pasos: 1.", "Abre Spotify 2.").
   Igual que antes de la auditoría. Es un formato que los modelos de chat
   usan; si lo oyes, este sí lo arreglaría.

### 7.2 Los interruptores apagados

No son preguntas nuevas: son las claves OFF de la §6. Mi orden y mi voto; cómo
probar cada una, en la §8.

| Clave | Yo | Por qué |
|---|---|---|
| `[voice].audio_session = true` | sí, la primera | Es la mejora más grande de tu modo diario (no perder el principio de la orden). Antes, las cuatro pruebas de hardware |
| `[memory].stale_tool_turns = 2` | sí | −32 % de bytes a la nube con un coste raro (releer un archivo viejo) |
| `[llm].stop_on_repeated_failure = true` | sí | Evita los seis intentos y la frase enlatada |
| `[llm.glm].fallback_shortcuts = true` | sí | Sin internet, "pausa la música" pasa de 8.4 s a nada |
| `[voice].stream_release = "sentence"` | probar | Primero mira una semana la línea `habría hablado un preámbulo` del diario |
| `[voice].tts_presynth = true` | probar | Apágala si oyes cortes |
| `[memory].trim_chunk = 7` | solo con datos | Enciéndela si con DEBUG ves subir `cached_tokens`; en streaming esa línea puede no salir nunca: míralo con el shell sin streaming (§8, paso 14) |
| `[memory].idle_reset_min = 30` | a tu gusto | Es producto: olvida la charla tras media hora |
| `[paths].relative_base` | ver D36 | |
| `barge_in_cancel`, `stream_speak_concurrent`, `oww_exact_frames` | cuando vuelvas al loop concurrente | No aplican a tu modo |
| `[wake].pregate = "oww"` | todavía no | Sin modelo propio no despierta nunca (D14) |

### 7.3 Seguridad y confirmaciones

**D1. ¿Cómo dice la confirmación por voz una ruta?**
Opciones: (A) como hoy, literal; (B) literal, pero diciendo "tu carpeta
personal" por `~/`; (C) una forma hablable solo para la voz (carpetas con
pausas, "punto txt"), y el texto exacto para el shell; (D) solo el nombre del
archivo y la zona ("lista_super.txt, fuera de la zona libre").
Hoy: se lee símbolo a símbolo, la `~` no suena y `_` suena a espacio:
`~/.bashrc` y `/.bashrc` se oyen igual.
Recomiendo (B) ya y (C) cuando lo oigas y te canse: (B) cierra la única
ambigüedad que importa para la seguridad con un cambio mínimo.

**D2. ¿Qué tan estricta es la confirmación por voz?**
Opciones: (A) más estricta: solo "confirmo" o una respuesta muy corta que
empiece por un sí; (B) la regla actual más una lista corta de veto ("qué va",
"a ver", "espera"); (C) dejarla.
Hoy ya es bastante estricta: una pregunta, un "va a", un "si" condicional o
un sí que no esté entre las dos primeras palabras cuentan como no (comprobado
en `b81e1c2`: "claro que sí" ya no confirma; "sí", "sí claro" y "confirmo",
sí). Queda un sí falso: "¡Qué va!", que aquí es un no, confirma por su "va".
Recomiendo (B) con ese veto: lo demás ya falla hacia el lado seguro. La lista
de palabras de duda es vocabulario tuyo.

**D3. `launch_project` ejecuta sin preguntar cualquier `launch*.sh` del árbol
(T7 de septiembre).**
Opciones: (a) marcarla insegura: pregunta siempre; (b) buscar lanzadores solo
en el primer nivel del proyecto; (c) las dos; (d) preferir el menos profundo;
(e) dejarlo.
Hoy: lanza el de mayor rango aunque esté enterrado en `reference/` o en un
respaldo.
Recomiendo (b): es lo que esperas de "lanza X" y quita el caso peligroso sin
añadir un "sí" a cada lanzamiento.

### 7.4 Voz: lo que se oye

**D4. ¿Qué digo cuando la respuesta es solo código?**
Opciones: (A) silencio, como hoy; (B) una frase fija ("Te lo dejo en pantalla,
patrón"); (C) la frase solo si hay HUD.
Hoy: no suena nada y el HUD pasa por HABLANDO; el texto queda en HUD y consola.
Recomiendo (B): un silencio tras una pregunta parece una falla. La frase la
eliges tú.

**D5. ¿Y si el bloque de código nunca cierra su cerca?**
Opciones: (A) como hoy: se corta como siempre, el código se lee pero nunca se
pierde prosa; (B) callar todo lo que sigue a la cerca; (C) como B más una frase
de aviso.
Hoy: (A). Pasa cuando el modelo olvida cerrar o la respuesta se corta dentro
del bloque.
Recomiendo (A): con (B) una cerca suelta enmudece el resto de la respuesta, y
la persona ya le pide al modelo no usar bloques.

**D6. ¿Qué palabras dejan una orden "a medias"?**
Opciones: (a) dejar la lista; (b) la propuesta del diseño: una palabra con
tilde escrita o un "?" o "!" final cierran la frase; "todo/a/os/as", "menos",
"luego" y "entonces" salen de la lista; y una "para" o un "si" sueltos cuentan
como completos; (c) solo una parte.
Hoy: de 40 órdenes completas de prueba, 19 todavía esperan 2.5 s de más:
"súbele más", "cierra todo", "borra todo", "¿qué canción es esta?"…
Recomiendo (b): el prototipo del diseño dejó 1 de 40 y siguió detectando las 14
frases de verdad incompletas. El riesgo es que una pausa justo tras "para" corte
la orden.

**D7. ¿Saludo corto al arrancar?**
Opciones: (a) decir solo "Crotolamo escuchando, patrón." e imprimir el modo;
(b) dejarlo.
Hoy: el saludo dice también el modo y bloquea: mientras suena no oye.
Recomiendo (a): ahorra 3.61 s de sordera en tu modo (2.29 s en el concurrente)
en cada arranque, login y cambio de modo desde el panel.

**D8. ¿Un salto de línea suelto es fin de frase?**
Opciones: (a) sí: cada línea, su frase con pausa; (b) no, pero convertirlo en
una coma; (c) dejarlo y corregir los comentarios del código que dicen lo
contrario.
Hoy: un salto de línea solo corta si viene tras puntuación, si deja una línea
en blanco, si la línea siguiente lleva sangría o, desde `b81e1c2`, si la
línea siguiente abre con un número suelto (§7.1).
Recomiendo (c): con (a), una prosa que el modelo parte a mitad de frase sonaría
a trozos, y eso no se puede juzgar sin oírlo.

**D9. ¿Cómo se normaliza el audio antes de Whisper? (V6 de septiembre)**
Opciones: (A) normalizar solo si el pico pasa de 0.02; (B) poner un tope a la
ganancia (×8-×10); (C) no tocar la ganancia y no mandar a Whisper lo que no
tiene voz; (D) filtrar la transcripción a la salida.
Hoy: se lleva siempre el pico a 0.9 (un ruido de pico 0.014 se multiplica por
64). Medido con Whisper `base` real: un soplido normalizado dio órdenes
fantasma 6 de 6; crudo, 0 de 6. Parte de (C) ya está hecha en el loop
concurrente.
Recomiendo (B), medido con tu micrófono: la voz muy baja a veces solo sale
normalizada.

**D10. Las correcciones de `normalize.py` (V13 de septiembre).**
Dos preguntas. Las sustituciones casan dentro de otras palabras ("abre mis
sitios web" → "abre mi escritorios web"): ¿palabra completa? Y la regla que
convierte "qué es una carpeta…" en la ORDEN "crea una carpeta…", con una tool
que no pide confirmación: ¿se queda?
Hoy: las dos siguen igual (solo se arregló "yutube" → "youtubee").
Recomiendo palabra completa, y quitar las variantes "qué es" / "es" de la
regla: una pregunta no debe crear una carpeta.

**D11. Las variantes "podadas" siguen despertando.**
Opciones: (a) aceptarlo y corregir el comentario del toml; (b) una lista
`[wake].negatives`; (c) subir el umbral; (d) quitar la comparación por ventana
deslizante.
Hoy: "control amo" o "cuatro lamo" activan por parecido aunque no estén en la
lista.
Recomiendo (a) y mirar una semana los casi-aciertos del diario antes de añadir
negativos.

### 7.5 Wake, micrófono y servicio

**D12. Loop concurrente sin micrófono: el servicio se reinicia cada ~5 s y
saluda cada vez.**
Opciones: (a) espera creciente en el unit (`RestartSteps=` +
`RestartMaxDelaySec=`); (b) rendirse tras N fallos (`StartLimit…`) y quedar en
"failed"; (c) que el camino normal reintente dentro del proceso 90 s, como hace
`audio_session`; (d) aparte: no hablar el saludo en un reinicio automático.
Hoy: bucle sin fin; el límite por defecto de systemd no llega a saltar. Un
disparador realista: `input_device = "crotolamo_aec_source"` sin el módulo AEC
cargado. Lo mismo pasa si el listener falla al arrancar por otra causa (D50
es una), y el panel lo pinta como "Arrancando… cargando modelos de voz" con
el botón principal deshabilitado: desde ahí no se puede parar el bucle.
Recomiendo (a) + (d): un micrófono que vuelve se recupera solo y la laptop no
habla cada cinco segundos.

**D13. ¿La config de fábrica debe traer tu modo real?**
Opciones: (a) dejar `use_oww = true` + "hey jarvis"; (b) poner `use_oww =
false`, que es lo que corres. Se validó con micrófono en julio, antes de esta
auditoría; el código de ese camino cambió aquí (wake, fin de frase, loop
simple, TTS, carga de modelos) y todavía no se ha oído (§8, paso 7).
Hoy: un clon limpio despierta con "hey jarvis".
Recomiendo (b) una vez pasadas las pruebas de voz del paso 7: lo de fábrica
debería ser lo que se ha oído funcionar.

**D14. ¿Qué hacemos con el coste en reposo de tu modo con voces en el cuarto?**
Opciones: (a) nada; (b) exigir una duración mínima de voz antes de pasar la
ventana a Whisper (quita 10 de 10 pasadas por toses, 0 de 12 de tele; riesgo:
perder un "crotolamo" dicho muy rápido); (c) ventana del encoder ajustada al
segmento (87-90 % menos CPU del encoder por pasada en la medición inicial, y
el encoder es el 53-87 % de cada pasada; pero 1 de 3 muestras alucinó); (d)
entrenar el modelo propio con tus grabaciones y usarlo de pre-filtro
(`pregate`).
Hoy: 12 pasadas por minuto de tele.
Recomiendo (d), que es la que lo arregla de raíz, y antes grabar el corpus de
la §10.2 para medir (b) y (c) sin arriesgar nada. Ojo: el `crotolamo.onnx`
versionado está sin su archivo de pesos.

**D15. En el loop concurrente, la pausa del panel deja el micrófono abierto.**
Opciones: (a) dejarlo (reanuda al instante); (b) cerrar el stream al pausar;
(c) cerrarlo solo con `audio_session = true`.
Hoy: lee y tira audio 31 veces por segundo y el indicador de captura del
escritorio sigue encendido.
Recomiendo (c): no toca el camino de fábrica.

**D16. openWakeWord 0.6 (Python 3.11): no bajar el umbral de 0.5.**
Opciones: (a) solo documentarlo; (b) pasar ~2 s de ceros tras cada reinicio
(reinicios con score sobre 0.3: 5.7 % → 0.3 %); (c) ignorar el score 1.3 s
tras cada reinicio; (d) nada.
Hoy: con 0.5 no dispara nunca (lo frena la segunda etapa del modelo); con 0.3
abre un turno fantasma tras ~4 % de los turnos.
Recomiendo (a) por ahora: solo afecta a `use_oww = true`. Revisar (b) cuando
tengas modelo propio.

**D17. ¿Fijar versiones de openwakeword y del resto del extra de voz?**
Opciones: (a) `openwakeword>=0.6.0` solo en Python < 3.12; (b) fijar una
versión por intérprete (0.6 en 3.11, 0.4 en 3.12); (c) dejarlo flotando.
Hoy: sin fijar; en 3.12 pip instala la 0.4 (de 2023) y en 3.11 la 0.6. El
código además se apoya en detalles internos de faster-whisper, piper-tts,
silero-vad y onnxruntime (con caída automática al camino de siempre).
Recomiendo (b), con las versiones de tu `.venv` (`pip freeze`): instalaciones
repetibles, y el CI sigue ejercitando las dos.

**D18. El HUD reintenta cada 3 s para siempre si no hay pantalla.**
Opciones: (A) dejarlo; (B) espera creciente (`RestartSteps=6`,
`RestartMaxDelaySec=60`); (C) atarlo a `graphical-session.target`; (D) límite
duro de arranques.
Hoy: (A).
Recomiendo (B): con (C) en Hyprland sin uwsm el HUD no arrancaría nunca.

**D19. ¿Qué pasa si arranca un segundo listener?**
Opciones: (a) candado en `~/.crotolamo/listener.lock`: el segundo avisa y sale
con 0; (b) que `launch.sh` ofrezca parar el servicio; (c) que el nuevo releve
al viejo; (d) documentarlo.
Hoy: dos procesos peleando por el mismo micrófono y el mismo HUD.
Recomiendo (a) + (b): con salida 0, systemd no entra en bucle de reinicios.

**D20. Reponer la música al apagar no tiene tope.**
Opciones: (a) dejarlo; (b) reponer los reproductores en paralelo; (c) un plazo
global; (d) subir `TimeoutStopSec`.
Hoy: 2 s por reproductor atenuado si el D-Bus está colgado, uno tras otro. El
peor caso medido es de 11.05 s y pasa los 10 s del unit: tres reproductores,
todos los candados tomados y el vaciado de la memoria atascado en su tope de
2.5 s, o sea con `[memoria]` activa. Con la memoria apagada, como va de
fábrica, ese mismo caso midió 8.55 s (9.05 s si además hay un server MCP que
ignora el SIGTERM). Esto es así con o sin el remate de los servers MCP
(`f460fc0`), pero NO es anterior a la auditoría: con el código de `f316103` la
reposición de esos tres reproductores tarda 7.0 s (medido sobre su
`emergency_restore`; el resto de aquel apagado no esperaba a nada) y cabía.
Los ~4 s de más los añadió la auditoría al apagado por señal: el candado del
seguidor de música (0.5 s), el vaciado de la memoria (hasta 2.5 s) y el
candado del HUD (1 s).
Recomiendo (b) (§10.4).

**D21. `[voice].sample_rate` distinto de 16000.**
Opciones: (A) rechazar otro valor al arrancar; (B) soportarlo remuestreando.
Hoy: el toml ya dice que solo vale 16000, pero nadie lo valida: con otro valor
las órdenes salen vacías.
Recomiendo (A): PipeWire ya entrega 16 kHz de cualquier micrófono.

### 7.6 Memoria

**D22. `olvidar_recuerdo` borra el recuerdo equivocado.**
Opciones: (A) dos pasos: "el más parecido es «X», ¿lo olvido?"; (B) un paso,
pero solo borra si comparte una palabra de contenido con lo que dijiste; (C) B
para borrar directo y A cuando no hay palabra en común; (D) dejarlo.
Hoy, solo con `[memoria]` activa y mem0 arriba: borra el más parecido y te dice
cuál. Medido: equivocado en 21 de 42 peticiones. Con (B): 0 equivocados, pero
las 17 paráfrasis ("lo de mi mascota") responden "no encontré".
Recomiendo (C): cero borrados equivocados sin perder las paráfrasis, al precio
de un turno más solo en el caso dudoso.

**D23. El filtro de secretos rechaza frases inocentes y deja pasar secretos
dictados en palabras.**
Opciones: (a) dejarlo (máxima cautela: "el secreto de mi salsa es chipotle" se
rechaza); (b) exigir dígito, mayúscula o símbolo (deja pasar "mi contraseña es
gatonegro"); (c) regla laxa para palabras inequívocas (contraseña, password,
api key) y estricta para las ambiguas (secreto, token); (d) la otra mitad de
la pregunta, que se puede sumar a cualquiera: tratar casi cualquier "mi
contraseña/clave es …" como secreto y rechazar números de tarjeta. Cierra los
huecos de la §2 (contraseñas dictadas en palabras, tarjeta y CVV) al precio de
más falsos positivos.
Hoy: (a), con esos huecos abiertos.
Recomiendo (c): quita los falsos positivos sin dejar pasar "mi contraseña es
gatonegro". Y (d) al menos para las palabras inequívocas (contraseña,
password): va en la misma línea que (c) y cierra el hueco de las contraseñas
dictadas en palabras. Lo de las tarjetas es decisión aparte.

**D24. ¿Qué turnos se mandan a extraer hechos?**
Opciones: (a) como hoy: todos menos las órdenes ancladas ("pausa la música");
(b) saltar también los turnos con error del LLM y los vacíos ("gracias", "sí");
(c) además, agrupar la extracción tras un rato de silencio.
Hoy: (a). Cada extracción son ~8.7 mil tokens contra la misma key y el mismo
límite que el cerebro.
Recomiendo (b): son turnos que casi nunca traen un hecho nuevo. Ni (c) ni la
otra idea que quedó apuntada (pausar la extracción mientras responde el
respaldo local) se pueden validar sin GLM real.

### 7.7 Núcleo: respaldo local y routing

**D25. ¿Subir `num_ctx` del modelo local a 4096?**
Opciones: (a) dejar 2048; (b) 4096 (unos +112 MB de caché, estimados, no
medidos); (c) `pin_facts = false` si algún día usas solo el local.
Hoy: 2048, y cada request se ajusta antes de mandarlo. Con 2048 caben la
persona, 8 tools y poco más: el local pierde pronto el hilo, y con 30 hechos
anclados (~540 tokens) aprieta más.
Recomiendo (a) mientras GLM sea el cerebro. Si ves seguido el aviso `no cabía
en num_ctx`, sube a 4096.

**D26. Si caen la nube y el local, ¿qué digo?**
Opciones: (A) como hoy, el error de Ollama; (B) el de la nube (su texto ya
pregunta "¿Hay internet?"); (C) uno combinado.
Hoy: oyes "No pude hablar con Ollama…" y la causa real queda en el log.
Recomiendo (B): dice la causa sin inventar una frase nueva.

**D27. El modelo repite una llamada idéntica que SÍ funcionó.**
Opciones: (A) dejarlo; (B) cortar a la tercera repetición seguida; (C) no
re-ejecutar nunca una idéntica en el mismo turno.
Hoy: "abre youtube" con el modelo atascado puede abrir 6 pestañas. La clave
`stop_on_repeated_failure` solo cubre las que fallan.
Recomiendo (B) con tres: tolera "salta dos canciones" en dos pasos.

**D28. La persona le pide al modelo confirmar él mismo lo delicado.**
Opciones: (A) dejar la frase "confirmas con el patrón antes de moverle"; (B)
quitarla: el guard ya pregunta, y ahora dice qué va a tocar; (C) reescribirla.
Hoy: posible doble confirmación; y con el modelo local, tu "sí" de respuesta
llega a un turno sin tools y hay que repetir la orden entera.
Recomiendo (B): una sola confirmación, la del guard, que además ya nombra el
objetivo. Quitar la frase no arregla el "hazlo otra vez" ni "el segundo": eso
es la segunda pregunta de D29.

**D29. Routing con el modelo local: cómo se reparte el tope de 8 tools.**
Varias preguntas con la misma raíz: el grupo de archivos (8 tools) llena el
tope y deja fuera a los demás; las palabras clave cortas casan con charla
normal ("eso suena lógico" → música, "playa" → "play"); y un server MCP grande
no cabe.
Opciones: (a) dejarlo y añadir frases clave según aparezcan huecos; (b) repartir
el tope entre los grupos que casan (medido: peor, rompe `launch_project` y
`make_dir`); (c) palabra completa para las claves cortas; (d) dejar que el tope
se desborde unas tools cuando un grupo se quedaría fuera.
Hoy: (a). Con GLM arriba nada de esto importa: ve las 42 tools siempre.
Recomiendo (a) + (c), y añadir "estado del sistema" como frase clave para
recuperar la que rompió la auditoría.
Segunda pregunta, la de los turnos de continuación: con el modelo local, un
turno que no casa ninguna palabra clave ("sí", "hazlo otra vez", "el segundo")
va sin tools aunque continúe la orden anterior (comprobado en `b81e1c2`: las
tres frases enrutan a cero tools).
Opciones: (A) dejarlo: la charla sigue rápida y hay que repetir la orden
entera; (B) heredar una vez las tools del turno anterior: el "gracias" tras
una orden pasaría de ~1-2 s a ~30 s en CPU, según las cifras del propio
router; (C) heredarlas solo si el turno parece continuación (lista corta:
"sí", "hazlo", "dale", "confirmo", "otra vez", "y ahora…") o si la respuesta
anterior acabó en pregunta.
Hoy: (A). Recomiendo (C); con GLM arriba no aplica.

**D30. Con GLM, ¿`list_windows` y `app_status` devuelven su texto directo?**
Opciones: (a) como hoy: GLM redacta la respuesta; (b) retorno directo: una ida
y vuelta menos (~1.2 s), a cambio de oír la lista cruda.
Recomiendo (a): una lista de ventanas leída tal cual molesta más que 1.2 s.

**D31. Un stream que acaba sin su marca de cierre, ¿es un error?**
Opciones: (a) como hoy: si trae texto o tool-calls completos se acepta con un
aviso; vacío o con argumentos a medias es error; (b) error siempre.
Hoy: (a), porque no se pudo comprobar que Z.ai mande siempre `[DONE]`.
Recomiendo mirar el diario dos semanas: si el aviso `cerró el stream sin
[DONE]` no sale nunca en uso normal, pasar a (b).

**D32. La pre-conexión a la nube no tiene interruptor.**
Opciones: (a) dejarla sin clave; (b) añadir una clave para apagarla.
Hoy: va encendida en los dos loops. Con la primera voz de cada orden (también
la del seguimiento) abre en segundo plano una conexión nueva con la nube; un
wake falso seguido de silencio no abre nada. No cambia lo que oyes.
Recomiendo (a): cuesta un apretón en segundo plano por orden y es lo que salva
el turno si la red cambió sin avisar (§9). Una clave más solo se justifica si
tu proveedor te cobrara o limitara las conexiones.

**D33. Afinados del respaldo que no se implementaron.**
Tratar un 429 como "vivo pero ocupado" en vez de pasar al local 60 s; sondear
la nube fuera del turno (quitaría los 4 × 5 s que quedan); precargar Ollama al
caer la nube; liberar el modelo local al volver; y dejar de reutilizar una
conexión ociosa de más de 15 s también en las respuestas sin streaming.
Hoy: nada de eso. Un 429 con espera de 5 s todavía cuesta un turno local de
~97 s en el bench.
Recomiendo no tocar nada hasta ver una caída real en tu diario: todo esto
cambia qué motor responde y no se pudo probar contra GLM.

### 7.8 Tools y escritorio

**D34. Un `.desktop` de proyecto se lanza con `gtk-launch` y dice "Lancé"
aunque no arranque nada (T8).**
Opciones: (a) `gio launch <ruta>`; (b) `gtk-launch` solo si ese ID está
instalado y, si no, decirlo; (c) leer `Exec=` y ejecutarlo; (d) cambiar la
frase.
Recomiendo (b): conserva lo que hoy funciona y deja de mentir. Decídela junto
con D3.

**D35. ¿Corregir en silencio los tipos que manda el modelo? (T16)**
Opciones: (a) estricto, como hoy: "Argumentos inválidos" y una vuelta más; (b)
convertir solo lo que no pierde nada ("6" → 6, 8.0 → 8, null en un opcional);
(c) además cualquier cadena numérica.
Recomiendo (b): ahorra una vuelta al LLM sin adivinar nada.

**D36. ¿Dónde caen las rutas relativas? (T10)**
Opciones: (A) en el home (pide confirmación cada vez); (B) en una carpeta de la
zona libre; (C) rechazarlas; (D) como hoy.
Hoy: contra el directorio de arranque; con el servicio, dentro del checkout del
repo. La clave `[paths].relative_base` ya existe, sin definir.
Recomiendo (B): `relative_base = "~/Documentos"`, predecible y sin preguntas
nuevas. Y un anexo: hoy
`write_file('$HOME/x.txt')` crea una carpeta que se llama `$HOME`; expandirlo.

**D37. ¿Qué navegador abre las URLs? (T6b)**
Opciones: (A) como hoy: Opera GX en flatpak si está, si no `xdg-open`; (B) una
clave `[apps].navegador`; (C) siempre `xdg-open`.
Recomiendo (B): además quita el `flatpak info` que hoy se lanza antes de cada
URL.

**D38. La lista negra de búsquedas rechaza "bomba de agua" y "XXXTentacion".**
Opciones: (a) dejarla (compara por subcadena); (b) por palabra completa; (c)
palabra completa y frases inequívocas ("bomba casera") en vez de "bomba" a
secas; (d) quitar esos términos.
Hoy: (a).
Recomiendo (c): deja pasar la ferretería y la música sin soltar lo que querías
bloquear.

**D39. Atajos que apuntan a un proyecto.**
Opciones: (A) dejarlo: hoy se guardan como búsqueda en Google del nombre; (B)
abrir la carpeta del proyecto, como en Crotolamo 1; (C) lanzar el proyecto.
Recomiendo (B): es inofensivo y es lo que hacía antes.

**D40. `light_control` con entidades que no son luces.**
Opciones: (A) soportarlas llamando al servicio de su dominio (un enchufe o un
ventilador se volverían controlables por voz sin confirmación); (B) rechazar lo
que no empiece por `light.` con un mensaje claro.
Hoy: dice "prendí" y no pasó nada.
Recomiendo (B) ya; (A) si de verdad tienes una lámpara en un enchufe.

**D41. `camera_snapshot("todas")`.**
Opciones: (a) aceptar vacío y "todas" como "el último evento de cualquier
cámara"; (b) exigir siempre una cámara; (c) "todas" sí, vacío sigue
preguntando.
Hoy: (b); "todas" vale en `camera_events` pero no aquí.
Recomiendo (c): "todas" es lo que dirías; un vacío suele ser que el modelo
olvidó el argumento.

**D42. Una nota con un título que ya existe.**
Opciones: (a) como hoy: crea `titulo_2` y te lo dice; (b) añadir al final de la
existente (útil para una lista de compras); (c) sufijo con fecha; (d) pedir
confirmación.
Recomiendo (a) hasta que eches de menos (b).

### 7.9 Doctor, empaquetado, repo y CI

**D43. El doctor no mira `~/.config/crotolamo/env`.**
Opciones: (a) leerlo como lo lee systemd y dar verde ("presente en el archivo,
no en este shell"); (b) lo mismo con aviso amarillo; (c) dejar el rojo y
corregir el texto.
Hoy: `python -m crotolamo doctor` en un shell sin la key da un rojo REQUERIDO
que te manda a poner la key donde ya está. `./launch.sh doctor` sí la carga.
Recomiendo (a): ese archivo es de donde la lee el servicio de verdad.

**D44. El check `spotify` del doctor vigila una variable que nadie lee.**
Opciones: (a) apagarlo (`spotify_client_id_env = ""`); (b) apuntarlo al
consumidor real; (c) dejarlo y corregir el texto.
Hoy: rojo opcional en cada corrida.
Recomiendo (a) hasta que exista el módulo.

**D45. ¿Se soporta instalar sin `-e`? (el T18 de septiembre)**
Opciones: (a) no, y decirlo en el README; (b) sí: empaquetar la config y el
doctor; (c) solo una variable `CROTOLAMO_CONFIG`.
Hoy: fuera de `pip install -e` no arranca (no encuentra `config/`).
Recomiendo (a): es una laptop y un dueño.

**D46. El `crotolamo.desktop` de la raíz y el `.gitignore`.**
El `.desktop` raíz lleva quemada la ruta de tu home, y `.gitignore` ignora
rutas que tienen archivos versionados (un `settings.json` de herramienta y el
`.onnx` de `wakeword_training/`).
Recomiendo borrar el `.desktop` raíz (`install.sh` ya instala el lanzador) y
decir en `.gitignore`, con excepciones `!`, qué se versiona a propósito.

**D47. El CI corre dos veces por cada push a una rama con PR.**
Opciones: (a) `push` solo en main; (b) saltar el duplicado con un `if`; (c)
`concurrency` con cancelación; (d) dejarlo.
Hoy: esta rama no tiene PR abierto y cada push lanza una sola corrida (runs
51 a 59: una por push). Con un PR abierto serían dos, y cada corrida son dos
jobs de 5 a 6 minutos en el runner nuevo (run 59: 5 min 06 s y 6 min 11 s).
Recomiendo (c): todas las ramas conservan su CI y lo superado se cancela.

**D48. `scripts/verify_phase.py` está huérfano** (dice "Fase N: verde"
corriendo una parte mínima de la suite).
Recomiendo borrarlo: CI y `pytest -q` ya lo cubren.

**D49. El nivel de log se aplica al logger raíz.**
Opciones: (a) dejarlo; (b) raíz en WARNING y el nivel configurado solo para
`crotolamo`; (c) como b más una clave para las librerías.
Hoy: (a), con `faster_whisper` ya callado aparte.
Recomiendo (a): el ruido conocido ya se quitó.

**D50. Sin servidor de audio al arrancar, el modo de fábrica muere con
traceback.**
Lo destapó el rojo del CI (§1) y ya existía antes de la auditoría (medido en
`f316103` y en `b81e1c2` con el mismo doble de `sounddevice`). Si el
PortAudio de la máquina trae la API de PulseAudio y al arrancar no hay
servidor de audio, `import sounddevice` lanza `PortAudioError`;
`WakeWordDetector.available()` solo atrapa `ImportError` y `OSError`. Con
`use_oww = true`, `crotolamo listen` muere ahí y, como servicio, systemd lo
reinicia hasta que hay audio (la familia de D12).
Opciones: (A) dejarlo: reinicia hasta que hay audio y entonces elige bien el
modo; (B) que `available()` lo tolere: el loop simple reintenta dentro del
proceso (medido en una copia: a 1, 2, 4, 8 y 16 s), pero el modo de fábrica
se quedaría en loop simple aunque el audio vuelva.
Hoy: (A). Con tu `use_oww = false` no te toca: `available()` ni se llama y
el loop simple ya reintenta así (comprobado en `b81e1c2` con el mismo doble:
no muere y reintenta a 1, 2, 4 y 8 s). No se sabe si el PortAudio de tu
Fedora trae esa API.
Recomiendo (A), y mirar en la laptop qué APIs trae tu PortAudio (§8, paso 3).

### 7.10 Menores

Preguntas chicas que dejaron los arreglos. Sin hacer nada, queda lo de "Hoy".

| Tema | Hoy | Yo haría |
|---|---|---|
| Palabras de duda en la confirmación ("para"…) | no hay lista | se resuelve con D2 |
| `open_url` con `mailto:` | lo abre como `https://mailto:…` | rechazar lo que no sea http(s) |
| Un atajo aprendido a "spotify" | abre el reproductor web; "abre spotify" lanza la app | que `[apps]` gane también en los atajos |
| `localhost:8123` como destino | lo toma por búsqueda | tratarlo como `http://` |
| `move_file` dentro de una carpeta con un archivo del mismo nombre | lo pisa, como `mv` (ya pidió confirmación) | dejar |
| `close_window` con dos ventanas de la misma app | cierra la primera | negarse y listarlas |
| "play" con dos reproductores | va al primero de playerctl | dejar |
| `aec.sh on` | solo comprueba el sink por defecto y te da el comando | que lo ponga él |
| Panel: "Arrancando… cargando modelos" | sale "Escuchando" mientras aún carga | `Type=notify` cuando toque el unit |
| Música atenuada tras un crash duro (SIGKILL) | se queda al 20 % | guardar el volumen y reponerlo al arrancar |
| `StreamSpeaker.finish()` espera 120 s por frase si el audio se traba | 120 s | 30 s |
| MCP: una línea que no es JSON mata al server | estricto | tolerar solo antes del `initialize` |
| MCP: dos timeouts seguidos desconectan al server | sí | dejar |
| `portaudio` en el doctor | opcional (sale 0 sin audio) | dejar |
| Fijar ruff, mypy y pytest en CI | flotan | fichero de restricciones solo para CI |
| mem0 que no arranca | apagada hasta reiniciar | reintento cada N minutos |
| Hechos guardados en SQLite con la memoria caída | no se leen al volver | `memoria migrar` (ya no duplica) |
| `llm_provider = "auto"` con cerebro local y key de nube | extrae en la nube | dejar |
| Valores por defecto cuando falta una clave de voz | 800 / 640 / 600 ms según el módulo | unificarlos a los del toml |
| Guarda de 0.5 s tras responder; gracia de 1.5 s del oído | constantes | claves, tras probar con tu eco |
| `MALLOC_ARENA_MAX=2` en el unit | sin poner | medir en la laptop (~44 MB en la medición inicial) |
| Panel por señales de D-Bus en vez de `systemctl` cada 1.5 s | sondea mientras está abierto | solo si te molesta |
| `interfaces.py` (los Protocol de voz) | nadie los usa | borrar hasta que haya un segundo backend |
| `wake_debug --como-produccion` no reinicia el detector al disparar | así, para que el pico no dependa del umbral | dejar |
| Banda de casi-aciertos del wake difuso en el diario | constante: desde 0.17 bajo el umbral (0.55 con 0.72) | dejar; `CROTOLAMO_WAKE_DEBUG=1` lo enseña todo |
| Orden dicha en la misma frase que el wake | se usa la transcripción del modelo `tiny` | re-transcribirla con el modelo de órdenes (una pasada más) |
| Aviso `no cabía en num_ctx` | uno por request ajustado | bajarlo a INFO tras el primero si usas mucho el local |
| Pedir el `usage` a GLM en streaming (`stream_options.include_usage`) | no se pide: `cached_tokens` no se ve en la voz | pedirlo si Z.ai lo acepta; cambia el payload |
| Umbral más alto con música (difuso 0.72 → 0.85; openWakeWord 0.5 → 0.7) | sube los fallos de wake justo cuando suena música | con pre-filtro, aplicarlo solo al filtro |
| Resumen de historial mientras responde el respaldo local (solo con `[memory].compaction = true`, que va apagada) | se sigue resumiendo | saltarlo durante la caída |
| Cadencia del keep-alive TCP de la nube (10 s de espera, sondas cada 5 s, 3 sondas) | constante | dejar |
| Ids de tool-call estables también hacia Ollama (~25 B por llamada guardada) | se mandan | dejar |
| `audio_session` + openWakeWord en `--simple`: la conversión a int16 puede diferir en 1 LSB | sin medir con micrófono | comprobar el score al activarla |
| Servers MCP al arrancar | el loop espera a sus handshakes | registro tardío, cuando uses MCP |
| Cambiar de modo desde el panel | reinicia el servicio y recarga modelos | cambio en caliente, cuando toque el panel |
| Caducidad de la credencial de música en el toml versionado | `musica_caduca = 2026-11-29` | moverla a `crotolamo.local.toml` |

### 7.11 Ya resueltas después (no hace falta decidir)

La confirmación que no nombraba el objetivo (`2f82ce6`); los binarios leídos
como texto (`0354cea`); el micrófono que se cerraba con el oído dentro de
`read()` (`269196e`); el diario con todo lo oído en INFO (`27af075`); Ptyxis y
los atajos a un dominio pelado (`0354cea`); el Whisper `tiny` que se cargaba
en loop simple + openWakeWord solo para la confirmación (ya no se carga; la
confirmación usa el modelo de órdenes, `91f1394` y `27af075`); borrar
`_play_interruptible` (`b09e63d`); y el número de lista que sonaba pegado a
la línea anterior (`b81e1c2`, §7.1).

Dos que cambiaron de forma pero siguen siendo tuyas: los atajos del local sin
internet y el streaming retenido con GLM ya no son preguntas abiertas de
diseño; son las claves apagadas `fallback_shortcuts` y `stream_release` de la
§7.2.

## 8. Lo que NO se pudo validar aquí

### Qué significa por área

| Área | Con qué se ejercitó | Lo que no se sabe |
|---|---|---|
| Micrófono y altavoz | `sounddevice` falso con audio guionizado; Whisper, Silero, openWakeWord y Piper reales sobre voz sintética y ruido generado | Umbrales con tu voz y tu cuarto; eco; AEC; qué hace PortAudio al suspender, al reiniciar PipeWire o al desenchufar, y si el de tu Fedora trae la API de PulseAudio (D50); cortes con `tts_presynth`; un barge-in de verdad; cómo suena un número de lista dicho como frase aparte (§7.1) |
| Pantalla (HUD y panel) | Un `gi` falso; alguna comprobación suelta con la libgtk real bajo xvfb | Nunca se abrió el HUD ni el panel. El escenario del HUD del bench no pudo ni importar Pango |
| MPRIS | El `playerctl` 2.4.1 real contra reproductores MPRIS falsos en un bus D-Bus privado; en el bench y en los tests, un `playerctl` falso que imita su formato | `playerctl --follow` con Spotify o un navegador reales (y va ON) |
| systemd | Nada corrió bajo `systemd --user` | `Type=exec`, los `EnvironmentFile`, el scope de las apps abiertas por voz, los tiempos reales de apagado, el bucle de reinicio |
| Ollama | Un servidor local falso con los formatos de su documentación y de su código | El formato real del stream con tool_calls en tu versión; el ajuste a `num_ctx`; todos los tiempos del modelo local |
| GLM (Z.ai) | El cliente real contra servidores falsos en loopback con TLS | Latencias reales; si Z.ai manda siempre `[DONE]`; si cachea prefijos; cómo responde a un 429 |
| mem0 | Un módulo `mem0` falso; solo el cálculo de scores se rehízo con las ruedas reales de fastembed y mem0 | Todo el camino real: Chroma, embeddings, extracción, RAM e hilos |
| Hyprland, ydotool, flatpak, Home Assistant, Frigate | Ejecutables y servidores falsos | Todo en vivo |
| Tu Python | La suite corre en 3.11 (openwakeword 0.6) y 3.12 (0.4) | Cuál usa tu `.venv`. Tu `ci.yml` dice que la laptop corre 3.11; si es así pip te instala openWakeWord 0.6: te aplica D16 y de las cifras de CPU valen las de 0.6. Confírmalo en el paso 1 |
| CI de GitHub | Los dos jobs (Python 3.11 y 3.12) en ubuntu-26.04 | Verde solo en el run 59 (`4d0d7c9`). Los siete anteriores salieron en rojo sin correr un test y nadie los miró; el octavo, en rojo por un test de esta auditoría. Ningún commit intermedio del 20 de noviembre tiene una suite verde en el CI (ruff y mypy sí pasaron): solo la corrida local (§1) |

Una nota más: los tiempos de apagado de la §2 y de D20 se midieron con
procesos sueltos y señales reales, no con `systemctl stop`.

### Lista de comprobación en la laptop

En orden, con la rama ya en la laptop
(`git fetch origin && git checkout claude/optimistic-ride-iloitn`). Cada paso
dice qué resultado es "ok". Los comandos van desde la raíz del repo.

1. **Instalar lo nuevo.** Reescribe los units con la ruta real, recarga systemd
   y, con `--start`, reinicia el servicio (sin `--start` el instalador no
   arranca ni reinicia nada y seguirías con el código viejo en memoria). De
   paso, apunta qué Python y qué openWakeWord tienes.

   ```
   ./desktop/install.sh --start
   .venv/bin/python -V
   .venv/bin/pip show openwakeword | grep Version
   systemctl --user is-active crotolamo.service
   ```

   Ok: el instalador termina con "Listo, patrón." y el servicio dice `active`.

2. **Suite.**

   ```
   .venv/bin/python -m pytest -q
   ```

   Ok: ningún `failed`, y `3866 passed` si la voz de Piper está en
   `~/Documentos/crotolamo2/voices/` (el `[paths].voces` de fábrica; los tests
   no miran tu `crotolamo.local.toml`). Aquí salió `3864 passed, 2 skipped`
   solo porque el contenedor no tiene esa voz: los dos saltados la piden, y
   con ella a mano corren y pasan (comprobado dándosela con
   `CROTOLAMO_TEST_PIPER_VOICE`: 3866 de 3866). Algún test más se salta si a
   tu máquina le falta otra cosa (por ejemplo, el Whisper `tiny` sin
   descargar en la caché de Hugging Face: 1 saltado; en el runner de GitHub
   se saltan 3); `pytest -q -rs` dice cuál y por qué. Si `.venv` no tiene
   pytest: `.venv/bin/pip install -e '.[dev]'`.

   Si falla `test_find_in_project_recorta_una_linea_kilometrica`, a tu copia
   le falta el commit `4d0d7c9` (§1): el fallo es del test, no del producto.

   Correr la suite no abre el micrófono ni hace sonar nada: de los 3866
   tests, uno solo llega al `sounddevice` de verdad, y lo único que hace es
   importarlo y llamar a `stop()`.

3. **Doctor.** Con `launch.sh`, no con `python -m crotolamo doctor` a secas
   (D43).

   ```
   ./launch.sh doctor; echo "salida: $?"
   ```

   Ok: todos los REQUERIDOS en verde y `salida: 0`. Esperado y real: el aviso
   de `musica-caduca` (la fecha del toml es 2026-11-29) y el rojo opcional de
   `spotify` (D44). Mira también el check opcional `portaudio`: si sale en
   rojo con `PulseAudio_Initialize: Can't connect to server`, tu PortAudio es
   de los de D50. Un verde no lo descarta: con el servidor de audio arriba
   sale verde tenga o no esa API. Para saberlo:

   ```
   .venv/bin/python -c "import sounddevice as sd; print([a['name'] for a in sd.query_hostapis()])"
   ```

   Si la lista trae `PulseAudio`, tu PortAudio es de los de D50. Aquí, con
   PortAudio 19.6.0, da `['ALSA', 'OSS']`; no se pudo correr contra uno que
   la traiga (el nombre está leído en el código de PortAudio).

4. **Smoke.** Primero sin micrófono, luego con él.

   ```
   ./launch.sh smoke
   .venv/bin/python scripts/smoke_voz.py --mic
   ```

   Ok del primero: `OK: la cadena TTS->STT funciona.` El segundo te pide
   hablar ("Habla ahora, patrón (grabo hasta el silencio)..."): di una frase;
   ok es `STT (micrófono) devolvió: '<lo que dijiste>'` y código de salida 0
   (con texto vacío sale con 1). Cualquier `FALLO: …` trae la causa.

5. **Arranque y diario.**

   ```
   systemctl --user restart crotolamo.service
   journalctl --user -u crotolamo.service -f
   ```

   Ok: sale `voz caliente (...)` con piper, silero y whisper, y `cerebro en la
   nube: no hay LLM que calentar`. No deben salir: `VAD: … uso el cargador de
   silero_vad, que importa torch` (el camino sin torch no encajó: +200 MB),
   `media: el seguimiento por eventos falló` ni reinicios en bucle. Si oyes
   huecos entre las frases del saludo, anótalo: el calentamiento ahora corre a
   la vez.

6. **RAM.** Da una orden y mira la memoria residente (sale en KB).

   ```
   ps -o rss= -p "$(systemctl --user show -p MainPID --value crotolamo.service)"
   ```

   Ok: con los Whisper de fábrica (`base` + `tiny`) ronda 680000; antes
   rondaba 870000. Con tus `small` + `base` espera en torno a 1.1 GB, unos
   1100000 (estimado: esos 683 MB más los ~425 MB que tu
   `VALIDACION_MIC_CROTOLAMO.md` les pone; no medido aquí). Lo que importa es
   que quede unos 190 MB por debajo de lo que marcaba antes con tu misma
   config, y que tras una respuesta larga no se quede 400 MB arriba.

7. **Once pruebas de voz** (cada una ata un arreglo o una regresión):
   - "Crotolamo, pon música" de corrido → la ejecuta sin pedirla otra vez.
   - "Buenos días, crotolamo, pon las noticias" → la ejecuta.
   - "Crotolamo" … pausa de 2 s … la orden → la oye entera.
   - En la ventana de seguimiento: "¿cómo me llamo?" → contesta.
   - "¿Y por qué?" → contesta sin la espera de 2.5 s.
   - "¿Cuánto es seis por siete?" → se oye "42".
   - Los números de una lista, sin depender de lo que conteste el modelo
     (solo usa el altavoz):

     ```
     .venv/bin/python -c "from crotolamo.settings import get_settings as g; \
     from crotolamo.voice.tts import TTS; t = TTS.from_settings(g()); \
     t.speak_sentences('Faltan estos:\n3. Reinicia\n4. Ir al banco a las 10\n5. Pan')"
     ```

     → se oyen el 3, el 4 y el 5, cada uno como frase corta aparte y ninguno
     pegado al final de la línea anterior. Si las pausas te estorban, es el
     punto 1 de la §7.1.
   - "Dame un ejemplo en Python de leer un archivo" → no lee el código.
   - "Borra el archivo prueba.txt de Documentos" → la pregunta nombra la ruta;
     responde "¿se va a borrar todo?" → cancela; repite y di "confirmo" → borra.
   - Pausa "Escuchar por voz" en el panel justo tras una respuesta → lo que
     digas después no se ejecuta.
   - Pon música, despiértalo → la música baja al momento y vuelve a su volumen
     exacto al terminar.

8. **Apagado.** Con música sonando y atenuada (en mitad de un turno):

   ```
   time systemctl --user stop crotolamo.service
   journalctl --user -u crotolamo.service -n 20
   ```

   Ok: menos de 10 s, la música vuelve a su volumen, el HUD queda en reposo y
   el diario no dice `stop-sigterm timed out`.

   Solo si usas `[memoria]` o MCP: en `./launch.sh shell` dile algo que
   valga recordar, escribe `salir` y cierra la ventana en cuanto veas
   "Crotolamo apagado, patrón.". Ok: no queda vivo ningún proceso de tus
   servers MCP (`pgrep -af` con su nombre) y en la sesión siguiente se
   acuerda de lo que le dijiste.

9. **Apps que sobreviven.** Abre algo por voz y reinicia el servicio.

   ```
   systemctl --user restart crotolamo.service
   ```

   Ok: la app sigue abierta.

10. **Sin internet.** Apaga el wifi y da una orden, con el diario a la vista.

    ```
    journalctl --user -u crotolamo.service -f
    ```

    Ok: sale `el motor principal falló (…); uso el local …` y contesta el
    modelo local (el primer turno carga Ollama en frío; con el wifi apagado el
    bench daba ~97 s hasta la primera palabra, casi todo esa carga). Al volver
    el wifi: `el motor principal volvió`.

11. **La libreta.** Ahora los hechos viajan en cada petición a la nube.

    ```
    ./launch.sh shell
    ```

    Pregunta "¿qué hechos tienes guardados?" y borra con "olvida el hecho
    número N" cualquier contraseña vieja. Ok: no queda ninguna.

12. **Solo si vuelves a openWakeWord.** Con el servicio parado:

    ```
    systemctl --user stop crotolamo.service
    .venv/bin/python scripts/wake_debug.py --como-produccion --threshold 0.3
    systemctl --user start crotolamo.service
    ```

    Ok: en el resumen, el "score MÍNIMO de las activaciones" queda claramente
    por encima del "score MÁXIMO del ruido". Con openWakeWord 0.6 no pongas el
    umbral bajo 0.5 (D16).

13. **Modelo de Whisper.**

    ```
    .venv/bin/python scripts/stt_harness.py record --n 10 --dir stt_bench
    .venv/bin/python scripts/stt_harness.py bench --dir stt_bench --models tiny,base,small
    ```

    Ok: gana el modelo más chico con 90 % o más de intents OK y media bajo 3 s.

14. **Las claves apagadas, de una en una.** En `config/crotolamo.local.toml`,
    reinicia tras cada cambio y úsalo un día antes de pasar a la siguiente.
    - `[voice] audio_session = true`. Antes, las cuatro pruebas: suspender y
      reanudar, `systemctl --user restart pipewire`, desenchufar los audífonos,
      y con `crotolamo_aec_source` si lo usas; tras cada una, el wake debe
      seguir respondiendo. Ayuda si una orden empezada medio segundo tras el
      wake ya llega entera. Si el diario avisa que la sesión no pudo abrir el
      micrófono, sigue como antes por su cuenta.
    - `[memory] stale_tool_turns = 2`. Ayuda si con `CROTOLAMO_LOG=DEBUG` baja
      `prompt_tokens` en la línea `GLM: prompt_tokens=… cached_tokens=…`. Mal si
      el asistente "olvida" un archivo que leyó hace poco más veces de las que
      toleras.
    - `[memory] trim_chunk = 7`. Ayuda solo si `cached_tokens` sube en esa
      misma línea. Si no sube, quítala.
    - Para las dos anteriores: esa línea solo sale si la respuesta trae
      `usage`. En streaming (el servicio de voz y `./launch.sh shell`) no se le
      pide a GLM, así que puede no salir nunca. Para verla seguro usa el shell
      SIN streaming:

      ```
      set -a; . ~/.config/crotolamo/env; set +a
      CROTOLAMO_LOG=DEBUG .venv/bin/python -m crotolamo shell 2>glm.log
      grep 'GLM: prompt_tokens' glm.log
      ```

    - `[llm] stop_on_repeated_failure = true`. Pide algo que el guard bloquee:
      debes oír la explicación una vez, no "Me enredé en demasiados pasos".
    - `[llm.glm] fallback_shortcuts = true`. Sin wifi, "pausa la música" debe
      ser instantánea.
    - `[voice] stream_release = "sentence"`. Antes, tras unos días de uso
      normal, mira cuántas veces habría hablado un preámbulo:

      ```
      journalctl --user -u crotolamo.service | grep -c "habría hablado"
      ```

      Si casi no sale, enciéndela. Ayuda si la primera palabra con GLM llega
      antes; mal si oyes "Voy a revisar…" antes de cada tool.
    - `[voice] tts_presynth = true`. Ayuda si desaparece el hueco entre
      frases; mal si oyes chasquidos o cortes.
    - `[memory] idle_reset_min = 30`. Es gusto: tras media hora callado empieza
      de cero.
    - Solo loop concurrente: `stream_speak_concurrent`, `barge_in_cancel` (con
      `--barge-in` y audífonos) y `oww_exact_frames` (recalibra antes el umbral
      con el paso 12). Y, con `--barge-in`, una comprobación en torno al
      arreglo de `b81e1c2`: pausa "Escuchar por voz" a media respuesta larga
      y reanúdala antes de que acabe, con el cuarto en silencio; no debe
      cortarse la respuesta ni abrirse un turno solo. Con los umbrales de
      fábrica tampoco pasaba antes del arreglo (0 de 24 aquí): sirve para ver
      que nada se rompió, no para ver el arreglo.
    - `[voice] whisper_cpu_threads`: déjala en 0 salvo que midas que Whisper y
      Piper se estorban.

## 9. Lo que se revisó y no se tocó

### Refutados con evidencia (los más instructivos)

- **Una afirmación de quien coordinó la auditoría: "el loop concurrente
  desperdicia un apretón TLS por orden".** Es cierto que la pre-conexión abre
  un socket nuevo
  aunque el anterior siga vivo (4 conexiones para 4 órdenes). El escéptico lo
  tumbó con un bench: si la red cambió en silencio entre dos órdenes, el
  repuesto recién comprobado salva el turno (la orden 2 responde en 0.0 s); con
  el arreglo propuesto, esa orden se come el tiempo de lectura entero (60 s en
  producción) y acaba en el modelo local. Se queda como está: cuesta un apretón
  en segundo plano por orden y compra robustez.
- **"Los servers MCP heredan todo tu entorno, API key incluida."** Cierto y
  aceptado: el README lo dice y da el remedio (`env -i` o un contenedor). Un
  filtro no protegería: un server corre como tú y puede leer la key del disco o
  de `/proc`. Tenlo presente si conectas un server de terceros.
- **"El guard no clasifica rutas bajo nombres de argumento que no conoce"** en
  tools MCP (por ejemplo `repo_path`). Es el límite documentado de una
  heurística; la defensa real con un server del que no te fíes es `confirm =
  "always"`.
- **"`terminate_now()` toma un candado dentro del manejador de señal y puede
  colgar el apagado."** Mecánicamente posible; medido: 0 de 16049 señales
  encontraron el candado tomado, con el hilo principal haciendo llamadas MCP sin
  parar.
- **"`crotolamo listen --help` arranca un listener de verdad."** Cierto. Se
  dejó: rechazar banderas desconocidas convertiría un typo en `listener.env` en
  un servicio que no arranca y se reinicia sin fin (D12).
- **"Frigate responde 200 sin cuerpo y se oye 'cero eventos'."** Reproducido
  con un socket crudo; refutado porque el Frigate real manda cabeceras y cuerpo
  juntos, y un cuerpo cortado ya se trata como respuesta rara. Ya existía.
- **Config mal tipeada a mano** (un número entre comillas, una sección que no
  es tabla, `inf`): la mayoría se refutó porque el proyecto no valida el tipo
  de cada clave en ningún lado; se arreglaron las que tumbaban el arranque o
  dejaban sin tools.
- **Formas que los proveedores no mandan** (`choices: [null]`, un `message` que
  es cadena): refutadas leyendo el código de Ollama y la forma real de las
  respuestas.
- **De la primera pasada, el único refutado**: "el doctor tarda 2.3 s en
  importar openWakeWord y Silero". Se dijo que no era evitable sin un falso
  verde. Luego la causa raíz de los modelos lo hizo posible (el runtime dejó de
  importar torch) y los cinco checks de voz del doctor bajaron de ~4 s a
  ~0.3 s (el comando entero, de ~3 s a menos de 1 s).

### Dejado sin arreglar, con su razón

- **`KillMode=process` en el unit**: no se puso. Lo que abres por voz va ahora
  a su propio scope, y el kill por cgroup es lo que limpia los servers MCP si el
  listener se cae.
- **JSON crudo de un tool-call hablado** con el modelo local en charla (38
  caracteres en el bench, antes y después): arreglarlo retiene texto legítimo y
  su frecuencia con tu modelo no se pudo medir.
- **La marca `[ahora: …]` en cada mensaje** (C7 de septiembre): sigue.
- **Avisos cosméticos que vas a ver**: al parar el servicio con MCP activo, un
  `transporte roto … matado por SIGTERM` por server; con openWakeWord 0.4
  (Python 3.12), un aviso de `CUDAExecutionProvider` al arrancar. No cambian
  nada.
- **El comentario del toml que dice "~32 tools"**: hay 42. Es una cifra
  histórica de la explicación y se dejó.
- **Sin veredicto**: dos hallazgos bajos de la unidad del estado del HUD, de
  los que no consta la verificación del escéptico. Uno quedó arreglado de
  todos modos desde otra unidad (el HUD ya no se congela con un
  `hud_state.json` que no es UTF-8, `0354cea`). El otro sigue: el HUD se fía
  del pid que publicó el estado; tras un apagón, si otro proceso reutiliza ese
  pid, podría enseñar un HABLANDO viejo hasta que el listener nuevo publique.
- **El decimal partido al principio de una línea**: "40. 5 GB libres" se oye
  "5 GB libres". Idéntico antes de la auditoría. No hay evidencia de que el
  cerebro escriba un decimal con un espacio (la persona le pide los números
  en palabras) y los remedios rompen otro caso: unirlo convertiría "12. 3
  son de tu jefe" en "12.3 son…" (§11).
- **Límites del shell tras los arreglos del cierre de ventana** (`f460fc0` y
  `3993c47`): (a) desde que el shell empieza a salir ("salir", Ctrl-D o una
  señal), SIGTERM y SIGHUP se ignoran hasta que el proceso termina (la salida
  tiene tope; Ctrl-C y SIGKILL siguen valiendo); (b) al cerrar la terminal el
  proceso acaba con una traza que nadie ve y código 1, pero la limpieza
  corre; (c) un Ctrl-C durante esa salida corta el paso en curso; si cae
  mientras se cierran los servers MCP, el server queda vivo: medido en
  `f460fc0` y otra vez en `b81e1c2` (4 de 4, con un server sordo al EOF y
  con uno que ignora SIGTERM). El arreglo de `3993c47` no cubre este camino,
  y el cierre en serie de antes tenía la misma debilidad.

## 10. Ideas de eficiencia (no implementadas)

### 10.1 Quitar torch de la instalación

Qué: versionar en el repo el `silero_vad.onnx` (2.3 MB; el paquete se declara
MIT) y quitar `silero-vad` del extra `[voice]`. El runtime ya abre ese archivo
con onnxruntime; torch solo se importa en el camino de respaldo.

Ganancia: `silero-vad` es el único paquete que exige torch. En este entorno
`torch+cpu` ocupa 765 MB en disco y, con lo que solo él arrastra (sympy,
networkx, mpmath), unos 800 MB de un `site-packages` de 1.6 GB. Menos descarga
en cada instalación y en cada corrida del CI. De paso desaparece el
`--extra-index-url` que hoy hay que recordar en ocho sitios (los dos README,
`install.sh`, `launch.sh`, `ci.yml`, `pyproject.toml`, el doctor y
`adapters.py`); una constante compartida solo uniría los dos que son Python
(y serviría para las nueve pistas que hoy no lo llevan, §5).

Coste y riesgo: un binario de 2.3 MB en el repo, que hay que actualizar a mano
si Silero publica otro. Se pierde el respaldo al cargador del fabricante;
quedaría el VAD por energía, que ya existe. Riesgo bajo: la salida con ese
mismo archivo ya está probada idéntica al bit. Pero los tests que comparan esa
salida con el cargador del fabricante piden torch y dejarían de correr: la
identidad al bit quedaría sin vigilar, salvo que silero-vad se quede solo en
el extra de desarrollo/CI. Y el check `voz-silero` del doctor tendría que
mirar el archivo versionado y no el paquete.

### 10.2 Un corpus de validación grabado una vez

Qué: diez minutos de audio crudo de tu cuarto, grabados una sola vez: el nombre
a varias distancias, órdenes, ruido, la tele, la voz de Piper. Y poder
reproducirlo por el camino de producción.

Qué existe: `stt_harness record` / `bench` ya graba órdenes una vez y las
repite por cada modelo de Whisper. Qué falta: `wake_debug.py` solo lee el
micrófono en vivo; no hay grabación continua ni una fuente "desde archivo" para
el wake difuso y el pre-filtro. Los grabadores ya aceptan una fuente inyectada
(es lo que usa `audio_session`), así que el hueco es chico.

Ganancia: la fila de micrófono de la §8 (umbrales, ruido, tele) y las
decisiones que dependen de lo que se oye dejan de ser "pruébalo a oído" y pasan
a ser una comparación que puedes repetir: D6, D9, D14, D16 y, del paso 14,
`oww_exact_frames`, `pregate` y la parte de `audio_session` que es "la orden
llega entera".

Coste y riesgo: un script de grabación, una fuente de archivo y cuidado con
dónde guardas el audio (es tu casa). No cubre el eco real del altavoz, lo que
hace PortAudio al suspender o desenchufar, los cortes de `tts_presynth`, ni
las claves de memoria y de respaldo del paso 14, que dependen de GLM y Ollama
y no del audio: todo eso seguirá siendo en vivo.

### 10.3 El arnés de voz de punta a punta, dentro del repo

Qué: la revisión de voz construyó un arnés (unas 1700 líneas, unos 40
escenarios) que mueve los dos loops enteros a través de `run_listen` con dobles
solo en los bordes y registra lo que se oye y lo que muestra el HUD. Vive en
una carpeta temporal y se va a perder. Subirlo como test de "qué se oye".

Ganancia: los fallos de voz que cazó la revisión final (las dos regresiones
del wake difuso, el código leído en voz alta, el número perdido) eran todos de
ese tipo: cada pieza pasaba sus tests y lo que salía por el altavoz estaba mal,
con la suite en verde.

Qué existe: de esa revisión ya quedaron 34 tests en el repo
(`tests/test_revision_final_x1voz.py`), sobre un arnés más chico, y otros 42
de la revisión posterior (`tests/test_revision_final_x8voz2.py`), que fijan
lo que le llega a Piper por los dos loops. Falta la matriz completa de
escenarios y la comparación de "lo que se oye" entre dos árboles: es
justo lo que cazó la regresión de las listas (§11).

Coste y riesgo: mantener el arnés, y unas respuestas de referencia que hay que
actualizar a propósito cada vez que cambie lo que se dice.

### 10.4 Reponer la música en paralelo al apagar

Qué: `Ducker.emergency_restore` repone los reproductores uno tras otro, con
hasta 2 s de espera cada uno si el D-Bus está colgado. Hacerlo a la vez.

Ganancia: el peor caso medido (tres reproductores atenuados con el D-Bus
colgado, candados tomados y el vaciado de la memoria atascado) baja de 11.05 s,
que pasa del `TimeoutStopSec=10`, a lo que mide ese mismo escenario con un solo
reproductor: 7.04 s.

Coste y riesgo: pocas líneas en `crotolamo/voice/media_aware.py`; corre dentro
del apagado por señal, así que pide el mismo cuidado que el resto de ese
camino.

## 11. Adenda: la revisión de los últimos arreglos y el cierre

Los dos últimos bloques de arreglos tuvieron una revisión adversarial
posterior: alguien que no los escribió los comparó con el árbol anterior, y
un escéptico intentó tumbar cada hallazgo. Entre las dos, 10 hallazgos: 5
confirmados y arreglados, 4 refutados y 1 decisión tuya, que después se
aplicó. Los commits de este cierre:

| Commit | Qué lleva | Suite local (pasan, 3.12 y 3.11) |
|---|---|---|
| `22e52d2` | La regresión de las listas que empiezan en 1 (§11.1) | 3818 |
| `601f02f` | La primera versión de este informe | — |
| `3993c47` | Shell tras "salir", cierre cortado de un server MCP, carreras de tests (§11.2) | 3833 |
| `4f64649` | Solo tests: la suite ya no toca el audio de la máquina (el CI, §1) | 3834 |
| `b81e1c2` | El número suelto como frase propia; el VAD del barge-in (§11.1) | 3864 |
| `4d0d7c9` | Solo un test: el que suponía el orden de una carpeta (el CI, §1) | 3864 |

### 11.1 El commit de voz (`2259ba2`)

Comparado con su padre, los dos árboles lado a lado: la segmentación en
frases (27000 respuestas generadas por 7 caminos, 4000 listas con elementos
que empiezan por cifra y 20000 textos adversariales), la confirmación en el
loop simple, los avisos del loop concurrente, el barge-in en pausa y los
tests (con el arreglo revertido fallan 28 de 34; los otros 6 son guardas
declaradas). Cuatro hallazgos:

- **Confirmado: una regresión de este mismo trabajo.** Una lista que empieza
  en 1 y cuyo primer elemento empieza por cifra ("1. 2 llamadas al banco / 2.
  Comprar leche…", o una agenda con horas) pasó a decir sus números, pegados
  al elemento anterior: la comprobación del decimal partido iba por delante
  de abrir la lista. **[ARREGLADO]** `22e52d2`: de 8000 listas generadas que
  empiezan en 1 diferían del padre 2771; ahora 0, también en streaming.
- **Decisión tuya, que se aplicó: el número suelto sonaba pegado** a la línea
  anterior ("Ir al banco a las 10 5."). **[ARREGLADO]** `b81e1c2`: cada
  número es su propia frase (§7.1). Lo comprobaron dos verificadores que no
  lo escribieron: uno de punta a punta (el listener entero en cuatro modos)
  y otro con un diferencial de 103263 respuestas generadas y 41 salidas por
  respuesta (entera, en streaming y por los dos loops, hasta lo que se le
  pide a Piper): ninguna palabra perdida, duplicada ni cambiada de orden; las
  respuestas sin número suelto, idénticas byte a byte; la primera frase sale
  en el mismo carácter que antes.
- **Refutado: la confirmación abre el micrófono después de escribir el estado
  del HUD**, y un disco atascado podría comerse el "confirmo". El orden es
  real; el coste, 0.23 ms de mediana y 6.5 ms de máximo en 3000 escrituras.
  Perder la respuesta pide un atasco de 0.3 a 0.7 s justo ahí. Al revisor le
  pasó una vez: con el disco ocupado por un `find /` suyo midió 49, 50 y
  190 ms en 3 de 18 corridas, y en una se perdió el "confirmo". El escéptico
  solo vio atascos con la máquina a carga 13 sobre 4 CPU y los atribuyó a
  esta máquina, no al código. El código ya hacía lo mismo antes de la
  auditoría con el micrófono de la orden, y falla hacia el lado seguro:
  acción denegada.
- **Refutado: "40. 5 GB libres" se oye "5 GB libres".** Se reproduce y es
  idéntico antes de la auditoría; se deja, con su razón, en la §9.

Fuera de los hallazgos, el revisor anotó un efecto secundario del propio
`2259ba2`: desde ese commit el detector del barge-in deja de oír durante la
pausa del panel (antes seguía oyendo, que era el bug) y nada lo limpiaba al
reanudar (loop concurrente con `--barge-in`; no es tu modo). **[ARREGLADO]**
`b81e1c2`. Con el Silero real sobre audio sintético (voz de Piper y silencio
generado; sin micrófono), al reanudar el silencio puntuaba 0.999, 0.986,
0.927…; ahora 0.171 como mucho. Con los umbrales de fábrica no llegaba a
abrir un turno falso (0 de 24 antes y después): es defensivo. Con un Silero
de pruebas que guarda memoria y `barge_in_min_chunks = 40` sí lo abría, 3 de
3; ahora 0 de 3. Precio: con voz (también de Piper) sonando justo al
reanudar, el barge-in dispara de media un chunk (32 ms) más tarde.

### 11.2 Los commits de apagado (`09aeedb` y `f460fc0`)

Se revisó el cierre en paralelo de los servers MCP (1, 3 y 6 servers sordos
al EOF salen en 1.0 s; antes de `09aeedb`, 1.0 / 3.0 / 6.0 s), el remate de
los que ignoran SIGTERM, las señales del shell (en el prompt, a medio
stream, en una tool MCP y en la confirmación), el vaciado de la memoria, el
nivel de log, el tope del servicio y los tests (de 28 mutaciones del código
de los dos commits, la suite caza 26). Seis hallazgos, ninguna decisión:

- **Confirmado (medio): cerrar la ventana justo después de "salir" todavía
  mataba la salida.** `f460fc0` cubrió el cierre con el shell en el prompt.
  Tras "salir", el vaciado de la memoria (hasta 5 s) y el cierre de los
  servers MCP corren con "Crotolamo apagado, patrón." ya en pantalla, y los
  manejadores de señal ya se habían repuesto. Medido: SIGHUP 0.2 s después
  del "apagado", rc=-1 y la extracción del último turno perdida; SIGTERM,
  rc=-15; en una pty real, perdida 3 de 3. **[ARREGLADO]** `3993c47`, para
  `python -m crotolamo shell` (lo que corre `launch.sh`) y para
  `interfaces/shell.py` como script: rc=0 y extracción guardada.
- **Confirmado (bajo): un cierre cortado dejaba vivo al server MCP.** Si un
  Ctrl-C, o el SIGTERM o SIGHUP del shell, llegaba mientras se retiraba un
  server colgado, el cierre se abandonaba y nadie lo seguía. Ahora quien
  corta manda SIGKILL al grupo antes de irse. **[ARREGLADO]** `3993c47`. No
  cubre el Ctrl-C durante la salida del shell (§9).
- **Confirmados (bajos, solo tests): el fallo suelto de
  `test_sighup_apaga_limpio_como_sigterm`** y el mismo hueco en los tests de
  señales del shell. Carrera del test: su proceso hijo esperaba la señal en
  una sola llamada bloqueante, y la que llega justo antes ya no la
  despierta. El del listener: 1 cuelgue en 3000 corridas bajo carga;
  esperando en tramos, 0 en 3000. Los del shell solo se colgaron a
  propósito, con gdb (40 de 40 corridas bajo carga pasaron). El listener no
  tenía la culpa. **[ARREGLADO]** `3993c47`, sin cambiar lo que comprueban.
- **Refutado: `close_all()` puede volver con un retiro ajeno todavía
  cerrando.** La carrera existe y el cierre en paralelo de `09aeedb` la
  ensanchó (provocándola a propósito: 1 de 40, 4 de 40 y 3 de 60; antes, 0
  de 40). Pero la ventana mide entre 90 y 285 µs, y pide que un retiro
  coincida por azar con la salida y que ese server siga vivo e ignore el
  EOF. Sin efecto medible con los llamadores reales.
- **Refutado: dos mutaciones que la suite no caza** (que una segunda señal
  reinicie la gracia del SIGKILL; que `kill_survivors` no vacíe su lista).
  Ciertas, pero sin efecto que se note: el único llamador sale del proceso
  justo después, y el retraso posible es de 0.2 ms en el caso real (dos
  SIGHUP seguidos), con tope de 2 s.

Visto y sin hallazgo. El cierre en paralelo arranca hilos durante la salida
del intérprete, cosa que la documentación de Python da por prohibida desde
3.12; en 3.12.3 y 3.13.18 arranca, y forzando el fallo cae al cierre en serie
y cierra todo. No se pudo correr en Python 3.14. Y el tope de 8 s del apagado
como servicio cuenta desde que arranca el manejador de Python, no desde el
SIGTERM de systemd: con el hilo principal dentro de una llamada en C arrancó
0.42 y 1.67 s tarde. Solo pesa en el loop simple y en el terreno de D20.

### 11.3 El CI de GitHub

Entra en este cierre por lo que tardó en mirarse: siete runs en rojo sin
correr un solo test, un octavo (`b81e1c2`) en rojo por un test de esta
auditoría que suponía el orden de los archivos de una carpeta, y verde solo
en el noveno (run 59, `4d0d7c9`: 3863 pasan y 3 se saltan en Python 3.12 y
3.11). El detalle está en la §1. Lo que deja: el juez es el CI, no la corrida
local, y aquí se miró tarde.

### 11.4 Dónde se paró

Los arreglos de `3993c47` y los cambios de tests de `4f64649` y `4d0d7c9` no
tienen otra revisión encima. El de `22e52d2` no tuvo revisión propia; el
segundo verificador de `b81e1c2` comparó de paso 20280 respuestas con listas
que empiezan en 1 contra el padre de `2259ba2`: dichas enteras suenan
idénticas las 20280; en streaming solo difiere lo que `2259ba2` arregló a
propósito y 10 casos de texto al azar. El de `b81e1c2` tuvo una ronda de dos
verificadores: ninguna regresión, y cada uno dejó anotado un residuo de
severidad baja, los dos anteriores al cambio (puntos 2 y 3 de la §7.1). El
flujo iba a abrir una ronda de reparación para esos dos residuos; se paró
antes de que tocara nada, con los archivos byte a byte iguales a los
verificados. No se abrió ninguna ronda de revisión más. Lo que queda abierto
es lo de la §7; lo que falta por validar, lo de la §8.
