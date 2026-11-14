"""Selección del motor de inferencia: Ollama (local) o GLM (nube).

Un único punto donde se decide con qué LLM habla Crotolamo. Ambos clientes
exponen el mismo contrato (`LLMEngine`: chat, chat_stream -> ChatResponse), así
que el resto del código —`ToolAgent`, `Conversation`, las tools— no distingue
cuál corre.

Se elige con [llm].backend en la config, o con la env CROTOLAMO_LLM_BACKEND
(que manda sobre el toml, útil para probar sin editar archivos).

Aquí vive también `HTTPTransport`, la base HTTP con keep-alive que comparten
los dos clientes.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import select
import socket
import ssl
import threading
import time
import urllib.parse
from typing import TYPE_CHECKING, Any, Callable, Protocol

from crotolamo.logging_setup import get_logger

if TYPE_CHECKING:
    from crotolamo.core.llm import ChatResponse

log = get_logger("core.engine")

OLLAMA = "ollama"
GLM = "glm"
VALID_BACKENDS = (OLLAMA, GLM)

# Reloj del transporte (tope del drenaje y edad de un keep-alive). En Linux,
# CLOCK_BOOTTIME: sigue contando mientras el portátil está suspendido, y
# monotonic() no; tras despertar, un socket de "hace 5 s" puede tener horas.
_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", None)


def _reloj() -> float:
    if _BOOTTIME is not None:
        return time.clock_gettime(_BOOTTIME)
    return time.monotonic()


# Tope para CONECTAR: el connect TCP (que socket.create_connection aplica a CADA
# dirección que da el DNS) y el handshake TLS. Va aparte del de LEER, que cubre
# la generación entera (60 s la nube, 180 s Ollama). Con un único número, una
# nube en un agujero negro (wifi arriba, internet caído) costaba 60 s por
# dirección en cada sonda antes de pasar al modelo local; un handshake sano
# tarda milisegundos, así que 5 s sobran incluso tethering por el móvil.
DEFAULT_CONNECT_TIMEOUT_S = 5.0

# Keepalive TCP del keep-alive ocioso: sin él, Linux no prueba un socket
# quieto en 2 h y una ruta muerta solo se descubría escribiendo y esperando el
# timeout de lectura entero. Con estos valores el kernel lo da por muerto entre
# 15 y 25 s después del corte (una sonda tras 10 s sin tráfico; 3 sin respuesta,
# cada 5 s) y `_dropped` lo ve antes de escribir. De paso mantiene viva la
# entrada del NAT del router. Precio: mientras el servidor no cierre el ocioso,
# un segmento vacío (y su ACK) cada 10 s; por eso solo contra servidores de
# fuera de esta máquina (ver HTTPTransport.__init__).
_KEEPALIVE = (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 3))


def _tope_de_conexion(connect_timeout: Any, timeout: float) -> float:
    """El tope de conectar pedido, o el de fábrica (sin pasarse del de leer).

    Un valor raro en la config no puede tumbar la red: con 0 el connect fallaría
    al instante en cada petición y con un negativo reventaría con un ValueError
    crudo. Se avisa en el log y se usa el de fábrica.
    """
    por_defecto = min(timeout, DEFAULT_CONNECT_TIMEOUT_S)
    if connect_timeout is None:
        return por_defecto
    try:
        segundos = float(connect_timeout)
    except (TypeError, ValueError):
        segundos = math.nan
    if math.isfinite(segundos) and segundos > 0:
        return segundos
    log.warning("connect_timeout inválido (%r): uso %s s", connect_timeout, por_defecto)
    return por_defecto


def _es_loopback(host: str) -> bool:
    """¿El servidor está en esta misma máquina (Ollama local)?"""
    host = host.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _enable_keepalive(sock: socket.socket) -> None:
    """Keepalive TCP en el socket (ver _KEEPALIVE). Best-effort: sin estas
    opciones (otro sistema) el socket queda como siempre."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for nombre, valor in _KEEPALIVE:
            opcion = getattr(socket, nombre, None)
            if opcion is not None:
                sock.setsockopt(socket.IPPROTO_TCP, opcion, valor)
    except OSError:
        pass


def _disable_keepalive(sock: Any) -> None:
    """Quita el keepalive TCP (best-effort; sin socket de verdad, nada)."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 0)
    except (AttributeError, OSError):
        pass


def _dropped(conn: Any) -> bool:
    """¿El kernel ya dio por muerto el keep-alive ocioso de `conn`?

    Un socket QUIETO no tiene nada que leer: si select() lo da por legible es
    que trae un EOF (el servidor lo cerró) o un error (RST, o el keepalive TCP
    vio la ruta caída). Mismo criterio que is_connection_dropped de urllib3.
    Sin socket que mirar (aún sin conectar, o los dobles de los tests) cuenta
    como vivo; con un descriptor que select() no admite, también (como antes).
    Sobre TLS, lo legible puede ser solo TLS (ver `_tls_cerrado`).
    """
    sock: Any = getattr(conn, "sock", None)
    fileno = getattr(sock, "fileno", None)
    if fileno is None:
        return False
    if fileno() < 0:
        return True  # ya cerrado
    try:
        legible, _, _ = select.select([sock], [], [], 0)
    except ValueError:
        return False  # descriptor >= FD_SETSIZE (1024): no se puede mirar
    except OSError:
        return True
    if legible and isinstance(sock, ssl.SSLSocket):
        return _tls_cerrado(sock)
    return bool(legible)


def _tls_cerrado(sock: ssl.SSLSocket) -> bool:
    """¿Lo legible de un socket TLS quieto es un cierre, o solo mensajes de TLS?

    TLS 1.3 manda los tickets de sesión DESPUÉS del handshake: una conexión recién
    abierta que aún no leyó nada (el repuesto de `HTTPTransport.preconnect`) los
    tiene pendientes y select() la da por legible sin que le pase nada (medido
    contra un servidor TLS 1.3 de loopback; con TLS 1.2, no). Sin esto, cada
    repuesto se tiraba al usarlo. Una lectura SIN bloquear los procesa: si no queda
    nada para la aplicación, OpenSSL pide más y la conexión sigue viva; un EOF, un
    RST o bytes que nadie pidió (un 408 antes de colgar) la dan por muerta.
    """
    try:
        previo = sock.gettimeout()
        sock.settimeout(0)
        try:
            sock.recv(1)  # b"" = EOF; un byte = algo que nadie pidió
        finally:
            sock.settimeout(previo)
    except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
        return False  # solo eran mensajes de TLS: sigue viva
    except (OSError, ValueError):
        return True
    return True


class _HTTPConnection(http.client.HTTPConnection):
    """HTTPConnection con dos topes: el corto para CONECTAR, el de siempre para LEER.

    http.client usa su único `timeout` para el connect TCP, el handshake TLS y
    cada lectura. Aquí connect() corre con `connect_timeout` y, ya conectado,
    el socket pasa al timeout de lectura de siempre.
    """

    def __init__(self, host: str, port: int | None, *, timeout: float,
                 connect_timeout: float, keepalive: bool, **kwargs: Any) -> None:
        super().__init__(host, port, timeout=timeout, **kwargs)
        self.connect_timeout = connect_timeout
        self.keepalive = keepalive

    def connect(self) -> None:
        lectura = self.timeout
        self.timeout = self.connect_timeout
        try:
            super().connect()  # TCP por dirección y, en HTTPS, el handshake TLS
        finally:
            self.timeout = lectura
        self.sock.settimeout(lectura)
        if self.keepalive:
            _enable_keepalive(self.sock)


class _HTTPSConnection(_HTTPConnection, http.client.HTTPSConnection):
    """Lo mismo sobre TLS: el handshake también queda bajo el tope de conectar."""


class LLMEngine(Protocol):
    """Contrato que cumple todo motor: LLMClient, GLMClient y FallbackLLM.

    `ToolAgent` y las interfaces solo dependen de esto; nada más del cliente
    concreto (host, base_url, api_key...) es parte del contrato.
    """

    @property
    def model(self) -> str:
        """Nombre del modelo que respondería ahora mismo."""
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> "ChatResponse":
        ...

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        # El motor ignora lo que devuelva on_token; solo FallbackLLM lo mira
        # (False = "retenido, no sonó"), por eso no se exige `-> None`. Si
        # on_token trae `checkpoint` (LiveStreamer), el motor lo consulta con
        # cada línea del stream: es por donde se cancela un turno a medias.
        on_token: Callable[[str], object] | None = None,
    ) -> "ChatResponse":
        ...


class HTTPTransport:
    """Conexión HTTP(S) persistente (keep-alive) compartida por los clientes LLM.

    POR QUÉ: `urllib.request.urlopen` abre TCP+TLS nuevos en CADA petición; un
    turno GLM con tools hace 2+ llamadas HTTPS secuenciales pagando el handshake
    completo cada vez. `http.client` reutiliza el socket entre peticiones.

    Qué maneja:
    - Reconexión transparente si el servidor cerró el keep-alive entre turnos
      (un único reintento ante `BadStatusLine`/`ConnectionError`, y el
      `ssl.SSLEOFError` con que Python 3.11/3.12 avisa de lo mismo sobre TLS).
    - Streaming: la respuesta devuelta se puede iterar línea a línea (SSE y
      NDJSON). El parser se detiene en el fin LÓGICO ([DONE] / done:true), un
      paso antes del fin FÍSICO del cuerpo (el chunk terminal). Si el cliente
      lo avisa con `finish(resp)`, la siguiente petición de ese hilo lee esa
      cola, acotada en bytes y en tiempo total, y reutiliza el keep-alive.
      Sin el aviso, si la cola no llega a tiempo o si la conexión lleva más
      de `_REUSE_MAX_IDLE_S` quieta, la conexión sucia se tira y se abre
      otra, como antes.
    - Un fallo a media petición (timeout esperando la respuesta, por ejemplo)
      cierra la conexión en el acto: con el socket abierto el servidor seguía
      generando para nadie hasta la siguiente petición de ese hilo.
    - Dos topes: `connect_timeout` para conectar (TCP, por cada dirección del
      DNS, y TLS; 5 s de fábrica) y `timeout` para leer (la generación entera).
    - Un keep-alive ocioso que el kernel ya dio por muerto se tira ANTES de
      escribir (ver `_reuse_or_close`); contra un servidor que no está en esta
      máquina, el keepalive TCP lo descubre 15-25 s después de un corte.

    SEGURO ENTRE HILOS vía conexión POR HILO (threading.local): en el modo de
    voz concurrente, WarmLLM y BrainThread pueden llamar chat() sobre el mismo
    cliente —y por tanto este transport— a la vez; con un socket único eso lo
    corrompía (CannotSendRequest, respuestas cruzadas). Cada hilo tiene ahora
    su propio keep-alive, sin locks ni contención. close() cierra SOLO la
    conexión del hilo que lo llama (nadie necesita hoy un cierre global; los
    sockets de hilos muertos los recoge el GC).

    Pre-conexión (`preconnect`): UNA conexión de repuesto, abierta desde otro
    hilo (el aviso del wake) mientras el patrón dice la orden. No es de ningún
    hilo: se la lleva entera la PRIMERA petición que llega, de cualquier hilo, y
    releva al keep-alive de ese hilo si tenía uno (ver post_json), así que el
    modelo por hilo no cambia. Uno que nadie pide lo cierra su segador (un Timer
    daemon, cancelado si alguien se lo lleva) al cumplir _SPARE_MAX_AGE_S. Sin
    repuesto, todo es exactamente como arriba.

    Stdlib puro, igual que el resto del proyecto: cero dependencias.
    """

    def __init__(
        self, base_url: str, timeout: float = 60, connect_timeout: float | None = None,
    ) -> None:
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"URL inválida para el transporte HTTP: {base_url!r}")
        self._https = parts.scheme == "https"
        self._host = parts.hostname
        self._port = parts.port  # None = puerto por defecto del esquema
        self.base_path = parts.path.rstrip("/")
        self.timeout = timeout  # leer: la generación entera
        # None = DEFAULT_CONNECT_TIMEOUT_S (sin pasarse de `timeout`).
        self.connect_timeout = _tope_de_conexion(connect_timeout, timeout)
        # Keepalive TCP solo contra un servidor de fuera: el loopback no se cae
        # en silencio y Ollama no cierra nunca un keep-alive ocioso, así que ahí
        # solo sería un sondeo cada 10 s, 24/7, sin nada que descubrir.
        self._keepalive = not _es_loopback(self._host)
        # Estado por hilo: .conn (la conexión keep-alive), .last (la última
        # respuesta, para detectar streams a medias), .finishable (el
        # consumidor avisó que .last llegó a su fin lógico) y .finished_at
        # (cuándo, con `_reloj`). Ver docstring.
        self._local = threading.local()
        # El repuesto de `preconnect`: (conexión ya conectada, cuándo se abrió con
        # `_reloj`) o None. Compartido entre hilos, por eso bajo su lock; el lock
        # solo cubre tomarlo o dejarlo, nunca la red.
        self._spare: tuple[http.client.HTTPConnection, float] | None = None
        # Su segador (ver preconnect): cierra el repuesto que nadie pidió al caducar.
        # Va y viene con `_spare`, bajo el mismo lock.
        self._spare_reaper: threading.Timer | None = None
        self._spare_lock = threading.Lock()
        # Una sola pre-conexión en vuelo: wakes seguidos con el DNS colgado no
        # apilan hilos ni sockets (el segundo vuelve al momento).
        self._preconnecting = threading.Lock()
        # El contexto TLS de todas sus conexiones https (ver `_contexto_tls`).
        self._tls: ssl.SSLContext | None = None

    @property
    def _conn(self) -> http.client.HTTPConnection | None:
        return getattr(self._local, "conn", None)

    @_conn.setter
    def _conn(self, value: http.client.HTTPConnection | None) -> None:
        self._local.conn = value

    @property
    def _last(self) -> http.client.HTTPResponse | None:
        return getattr(self._local, "last", None)

    @_last.setter
    def _last(self, value: http.client.HTTPResponse | None) -> None:
        self._local.last = value

    def _connect(self) -> http.client.HTTPConnection:
        if self._https:
            return _HTTPSConnection(
                self._host, self._port, timeout=self.timeout,
                connect_timeout=self.connect_timeout, keepalive=self._keepalive,
                context=self._contexto_tls())
        return _HTTPConnection(self._host, self._port, timeout=self.timeout,
                               connect_timeout=self.connect_timeout, keepalive=self._keepalive)

    def _contexto_tls(self) -> ssl.SSLContext:
        """El SSLContext de este transporte: uno para todas sus conexiones https.

        Sin `context=`, http.client crea uno nuevo por conexión y vuelve a cargar
        el bundle de CAs del sistema (~20 ms de CPU) en el hilo que conecta: el
        del turno cada vez que la nube cerró el keep-alive entre turnos, en cada
        reintento y en cada preconnect. Es la misma receta que la suya (el gancho
        de PEP 476, ALPN http/1.1, PHA), así que verifica igual. Un SSLContext es
        seguro entre hilos. Por transporte y no global: cada cliente lee las CAs
        (SSL_CERT_FILE) en su primera conexión; un bundle que cambie en disco
        después se ve al reiniciar, como en tools/_web.py. Perezoso: construir el
        cliente no lo paga, y lo suele pagar el preconnect, fuera del turno. Dos
        hilos a la vez la primera vez lo crean dos veces como mucho; cualquiera
        de los dos sirve, así que no hace falta lock.
        """
        contexto = self._tls
        if contexto is None:
            contexto = ssl._create_default_https_context()
            contexto.set_alpn_protocols(["http/1.1"])
            if contexto.post_handshake_auth is not None:
                contexto.post_handshake_auth = True
            self._tls = contexto
        return contexto

    # Cola de un stream ya terminado lógicamente: línea en blanco + chunk
    # terminal, unos pocos bytes que para la siguiente petición ya llegaron.
    # Los topes acotan el caso raro (un servidor que sigue mandando o que
    # retrasa el chunk terminal): pasado cualquiera, se descarta como antes.
    # El de tiempo es del drenaje ENTERO, no de cada lectura: un goteo más
    # rápido que eso retenía la petición hasta juntar los 64 KiB.
    _DRAIN_BYTES = 65536
    _DRAIN_TIMEOUT_S = 0.05
    # Un stream terminado solo deja su keep-alive a la petición que llega en
    # estos segundos (las llamadas de un turno con tools, una réplica rápida).
    # Un socket muerto en silencio (NAT, cambio de wifi) no avisa: escribir
    # "funciona" y la respuesta no llega nunca, así que costaría el timeout de
    # lectura entero y abriría el breaker. Antes del finish() los streams
    # nunca reutilizaban; pasado este tope, conexión nueva como entonces.
    # El keepalive TCP + `_dropped` NO lo vuelven innecesario: solo ven un corte
    # de hace 15-25 s o más. Medido con el bench del diseño (reloj simulado), en
    # el loop de voz con stream: un corte de hace 10-20 s cuesta 5 s con este
    # tope y 60 s sin él; el precio es un handshake por turno con más de 15 s
    # de hueco, lo que los streams pagaban siempre. Sin stream (respuesta leída
    # entera) no aplica, como siempre: ahí costaría ese handshake en cada turno.
    _REUSE_MAX_IDLE_S = 15.0

    def finish(self, resp: Any) -> None:
        """El consumidor llegó al fin LÓGICO de `resp` ([DONE] / done:true).

        Solo lo ANOTA, así que el turno en curso no paga nada: la lectura de
        la cola la hace la siguiente petición de este hilo (ver `_drain`).
        Un aviso sobre otra respuesta, o desde otro hilo, no cuenta.
        """
        if resp is self._last and resp is not None:
            self._local.finishable = True
            self._local.finished_at = _reloj()
            if self._keepalive and not resp.isclosed():
                # Con la cola sin leer, este keep-alive solo se reutiliza dentro de
                # _REUSE_MAX_IDLE_S (15 s) y el keepalive TCP no da veredicto antes
                # de 10 + 5*3 = 25 s: ahí solo eran sondas por la radio (medido: 4
                # en 41 s, y la conexión se tiraba igual) hasta que el servidor
                # cerrara el ocioso. Vuelve al reutilizarla (`_reuse_or_close`).
                _disable_keepalive(getattr(self._conn, "sock", None))
                self._local.keepalive_off = True

    def _drain(self, resp: Any) -> bool:
        """Lee la cola de `resp`, acotada; True si el socket quedó limpio.

        read1 lee lo de UN chunk cada vez, con el timeout del socket recortado
        a lo que queda del tope: entre lectura y lectura se mira el reloj, así
        que un goteo no estira el total más allá de _DRAIN_TIMEOUT_S (más la
        última espera, acotada por lo que quedaba).
        """
        if getattr(resp, "will_close", False):
            return False  # `Connection: close`: el servidor no lo va a reutilizar
        sock = getattr(self._conn, "sock", None)
        limite = _reloj() + self._DRAIN_TIMEOUT_S
        leidos = 0
        try:
            previous = sock.gettimeout() if sock is not None else None
            try:
                while not resp.isclosed():
                    queda = limite - _reloj()
                    if queda <= 0 or leidos >= self._DRAIN_BYTES:
                        return False
                    if sock is not None:
                        sock.settimeout(queda)
                    data = resp.read1(self._DRAIN_BYTES - leidos)
                    if not data:
                        break
                    leidos += len(data)
            finally:
                if sock is not None:
                    sock.settimeout(previous)
        except Exception:  # noqa: BLE001 - si no se puede drenar, se descarta
            return False
        return bool(resp.isclosed())

    def close(self) -> None:
        """Suelta la conexión actual (la siguiente petición abrirá otra)."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 - cerrar es best-effort
                pass
            self._conn = None
        self._last = None
        self._local.finishable = False
        self._local.keepalive_off = False

    # Un repuesto de `preconnect` solo sirve a la petición que llega en estos
    # segundos: la orden que el patrón dice tras el wake (unos segundos de voz
    # más el STT). Más tarde se cierra sin usarlo y se conecta como siempre: un
    # socket viejo puede haber muerto sin avisar (cambio de wifi, NAT). Más largo
    # que _REUSE_MAX_IDLE_S porque el repuesto conserva el keepalive TCP, así que
    # `_dropped` ve un corte de hace 15-25 s, como en el keep-alive ocioso tras
    # una respuesta leída entera (que ni siquiera tiene tope de edad).
    _SPARE_MAX_AGE_S = 30.0
    # La vida que un wake necesita que le quede al repuesto que encuentra: se usa
    # DESPUÉS, con la orden (hasta 4 s esperando la primera voz, la frase, el
    # silencio de corte, las rondas del endpoint) y el STT. Con menos, `preconnect`
    # abre otro: antes se conformaba con uno a punto de caducar, decía True y el
    # turno pagaba el handshake igual (medido: repuesto de 27 s y 6 s de orden).
    # Precio: wakes seguidos abren como mucho uno cada 15 s, no cada 30.
    _SPARE_MIN_LIFE_S = 15.0

    def _spare_usable(self, conn: Any, opened_at: float, life_left: float = 0.0) -> bool:
        """El repuesto sigue en edad (con `life_left` s por delante) y el kernel no
        lo dio por muerto."""
        return (_reloj() - opened_at <= self._SPARE_MAX_AGE_S - life_left
                and not _dropped(conn))

    @staticmethod
    def _discard(conn: Any) -> None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001 - cerrar es best-effort
            pass

    def preconnect(self) -> bool:
        """Abre por adelantado UNA conexión de repuesto (TCP y, en HTTPS, TLS).

        Para pagar el handshake mientras el patrón dice la orden, no dentro del
        turno. Mismo camino que cualquier conexión (`_connect`: tope de conectar,
        keepalive TCP, contexto TLS); la primera petición que llega, de cualquier
        hilo, se la lleva (`_take_spare`). Con un repuesto vivo y al que aún le
        quedan _SPARE_MIN_LIFE_S, no hace nada. El que nadie pide lo cierra su
        segador al cumplir _SPARE_MAX_AGE_S (`_expire_spare`).

        Nunca lanza: es una optimización. Sin red, o con otra pre-conexión aún en
        curso, devuelve False y el turno conecta como siempre. True = hay un
        repuesto listo.
        """
        if not self._preconnecting.acquire(blocking=False):
            return False
        try:
            with self._spare_lock:
                if self._spare is not None and self._spare_usable(
                        *self._spare, life_left=self._SPARE_MIN_LIFE_S):
                    return True
            conn = None
            try:
                conn = self._connect()
                conn.connect()
            except Exception as error:  # noqa: BLE001 - nunca tumba a quien avisa
                log.debug("preconnect: no pude abrir la conexión de repuesto (%s)", error)
                if conn is not None:
                    self._discard(conn)
                return False
            # El segador: sin él, tras un wake sin turno LLM (falso wake, una orden
            # del fastpath) el repuesto seguía abierto pasados sus 30 s, con sondas
            # de keepalive cada 10 s y luego en CLOSE_WAIT, hasta la siguiente
            # petición o el siguiente wake (horas, en el servicio 24/7). Un Timer
            # daemon espera sin sondear; se cancela si una petición se lleva el
            # repuesto o lo releva otro, así que hay uno vivo como mucho.
            segador = threading.Timer(self._SPARE_MAX_AGE_S, self._expire_spare, (conn,))
            segador.name = "PreconnectReaper"
            segador.daemon = True
            with self._spare_lock:
                viejo, self._spare = self._spare, (conn, _reloj())
                segador_viejo, self._spare_reaper = self._spare_reaper, segador
            if segador_viejo is not None:
                segador_viejo.cancel()
            if viejo is not None:
                self._discard(viejo[0])  # caducado, caído o con poca vida: nunca dos
            try:
                segador.start()
            except RuntimeError as error:
                # Sin hilos libres: el repuesto vale igual y caduca como antes, al
                # pedirlo o en el wake siguiente (cancel() de uno sin arrancar, nada).
                log.debug("preconnect: no pude arrancar el segador del repuesto (%s)", error)
            log.debug("preconnect: conexión de repuesto lista")
            return True
        finally:
            self._preconnecting.release()

    def _expire_spare(self, conn: Any) -> None:
        """El segador: el repuesto cumplió _SPARE_MAX_AGE_S sin que nadie lo pidiera.

        Corre en el hilo del Timer de `preconnect`. Solo cierra si el repuesto
        guardado sigue siendo `conn`: si ya se lo llevó una petición, o lo relevó
        otro wake, no toca nada. Nada sale de aquí: queda en el log. El Timer cuenta
        con monotonic, que no avanza con el portátil suspendido: tras despertar llega
        tarde, pero `_take_spare` mide con `_reloj` y no usa uno caducado igual.
        """
        try:
            with self._spare_lock:
                if self._spare is None or self._spare[0] is not conn:
                    return
                self._spare, self._spare_reaper = None, None
            log.debug("la conexión abierta por adelantado caducó sin usarse: la cierro")
            self._discard(conn)
        except Exception:  # noqa: BLE001 - el segador nunca tumba nada
            log.debug("no pude cerrar la conexión de repuesto caducada", exc_info=True)

    def _take_spare(self) -> http.client.HTTPConnection | None:
        """El repuesto de `preconnect`, entero para el hilo que lo pide, o None.

        Uno caducado o caído se cierra en silencio y la petición conecta como
        siempre, sin escribir en un socket muerto.
        """
        with self._spare_lock:
            spare, self._spare = self._spare, None
            segador, self._spare_reaper = self._spare_reaper, None
        if segador is not None:
            segador.cancel()  # sale en el acto: el repuesto ya no espera a nadie
        if spare is None:
            return None
        conn, opened_at = spare
        if self._spare_usable(conn, opened_at):
            log.debug("uso la conexión abierta por adelantado")
            return conn
        log.debug("la conexión abierta por adelantado caducó o se cayó: conecto otra")
        self._discard(conn)
        return None

    def _reuse_or_close(self) -> None:
        """La política de reutilización del keep-alive: UNA, para stream y no-stream.

        La conexión de este hilo se reutiliza solo si:
        1. Su última respuesta acabó. Leída entera (lo normal sin stream), o un
           stream que avisó su fin lógico (finish) hace como mucho
           _REUSE_MAX_IDLE_S y cuya cola se drenó dentro de su tope. Si quedó a
           medias, el socket trae bytes pendientes y no sirve.
        2. Y el kernel no la dio por muerta (`_dropped`): el servidor la cerró, o
           el keepalive TCP vio la ruta caída. Se mira DESPUÉS de drenar, porque
           la cola pendiente también hace legible el socket.
        Si no, se cierra sin escribir nada y la petición abre otra conexión, bajo
        el tope de conectar. Antes, un keep-alive muerto se descubría escribiendo:
        una escritura fallida más el reintento, o el timeout de lectura entero.
        """
        last = self._last
        finishable = getattr(self._local, "finishable", False) and (
            _reloj() - self._local.finished_at <= self._REUSE_MAX_IDLE_S)
        self._local.finishable = False
        if last is not None and not last.isclosed():
            if not (finishable and self._drain(last)):
                self.close()
                return
        if self._conn is not None and _dropped(self._conn):
            log.debug("keep-alive caído mientras estaba ocioso: conexión nueva")
            self.close()
        if getattr(self._local, "keepalive_off", False):
            # Un stream terminado le quitó el keepalive (ver finish); si la
            # conexión sigue, lo recupera: tras una respuesta leída entera el
            # ocioso ya no tiene tope y es lo que descubre la ruta caída.
            self._local.keepalive_off = False
            if self._conn is not None and getattr(self._conn, "sock", None) is not None:
                _enable_keepalive(self._conn.sock)

    def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> http.client.HTTPResponse:
        """POST JSON a `base_path + path` reutilizando la conexión si se puede.

        Devuelve la respuesta SIN leer el cuerpo: el caller decide si hace
        `.read()` completo o la itera línea a línea (streaming). No lanza por
        status >= 400; eso lo mapea cada cliente a sus mensajes en personaje.
        """
        # errors="replace": el historial puede traer sustitutos sueltos (el
        # nombre de un archivo con bytes no-UTF-8 que listó una tool, un
        # "\ud83d" suelto de un JSON ajeno). Con el codec estricto eso lanzaba
        # UnicodeEncodeError aquí, en ESTE turno y en todos los siguientes
        # mientras el mensaje siguiera en el historial. Lo irrepresentable sale
        # como "?"; el texto normal se codifica byte a byte igual que antes.
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8", errors="replace")
        hdrs = {"Content-Type": "application/json", **(headers or {})}

        # El repuesto de preconnect, si lo hay y sigue sano, releva incluso a un
        # keep-alive propio vivo: es la conexión más nueva (comprobada, <= 30 s) y su
        # handshake ya está pagado. En el loop concurrente BrainThread contesta sin
        # stream y su keep-alive no caduca: si el repuesto fuera solo para un hilo
        # sin conexión, no se usaría nunca tras el primer turno (un handshake tirado
        # por wake y dos sockets ociosos con sondas de keepalive). Se mira UNA vez,
        # antes del primer intento: el reintento de abajo conecta de nuevo, como
        # siempre (tras un cambio de red el repuesto puede estar tan muerto como el
        # propio). Sin repuesto, todo es como antes.
        spare = self._take_spare()
        if spare is not None:
            self.close()  # el keep-alive propio, si lo había: no quedan dos
            self._conn = spare
        else:
            self._reuse_or_close()

        url = f"{self.base_path}{path}"
        for intento in (1, 2):
            if self._conn is None:
                self._conn = self._connect()
            try:
                self._conn.request("POST", url, body=body, headers=hdrs)
                resp = self._conn.getresponse()
            except (http.client.BadStatusLine, http.client.CannotSendRequest,
                    ConnectionError, ssl.SSLEOFError) as error:
                # El servidor cerró el keep-alive mientras esperábamos: se
                # reconecta y se reintenta UNA vez. A la segunda, se propaga.
                # SSLEOFError es ese mismo socket muerto visto por TLS en Python
                # 3.11/3.12 (en 3.13 ya llega como ConnectionError).
                self.close()
                if intento == 2:
                    raise error
                continue
            except BaseException:
                # Cualquier otro fallo (timeout esperando la respuesta, red
                # caída): la petición quedó a medias y el servidor seguiría
                # generando para nadie. Se cuelga ya; el error sigue su camino.
                self.close()
                raise
            self._last = resp
            return resp
        raise AssertionError("inalcanzable")  # pragma: no cover


def preconnect_in_background(llm: Any) -> threading.Thread | None:
    """Pre-conecta el motor `llm` en un hilo daemon, si sabe hacerlo.

    Para el aviso del wake: el handshake TCP+TLS con la nube se paga mientras el
    patrón dice la orden, no dentro del turno (ver HTTPTransport.preconnect). Sin
    un `preconnect` llamable (un motor de mentiras, un agente raro) no arranca
    nada. El hilo vive lo que tarda en conectar (el DNS y luego el tope de
    conectar por dirección, como cualquier petición; uno solo a la vez por
    transporte) y ningún error sale de él: queda en el log. Devuelve el hilo,
    para que su dueño lo espere con un tope al apagar, o None.
    """
    preconnect = getattr(llm, "preconnect", None)
    if not callable(preconnect):
        return None

    def _run() -> None:
        try:
            preconnect()
        except Exception:  # noqa: BLE001 - una optimización nunca tumba al wake
            log.debug("preconnect en segundo plano falló", exc_info=True)

    hilo = threading.Thread(target=_run, name="Preconnect", daemon=True)
    try:
        hilo.start()
    except RuntimeError as error:  # sin hilos libres: el turno conecta como siempre
        log.debug("preconnect: no pude arrancar el hilo (%s)", error)
        return None
    return hilo


def resolve_backend(settings) -> str:
    """Nombre del backend a usar. La env pisa el toml; un valor raro cae a ollama."""
    raw = os.environ.get("CROTOLAMO_LLM_BACKEND") or settings.llm.get("backend", OLLAMA)
    backend = str(raw).strip().lower()
    if backend not in VALID_BACKENDS:
        log.warning("backend '%s' desconocido; uso '%s'", backend, OLLAMA)
        return OLLAMA
    return backend


def build_primary_llm(settings):
    """Cliente primario SIN envolver en FallbackLLM, con transport propio.

    Para usos auxiliares (hoy: el warm-up del listener) que no deben compartir
    ni el socket ni el circuit breaker del cliente del loop: un fallo del
    warm-up sobre el FallbackLLM compartido abriría el breaker 60s y degradaría
    el primer comando real al modelo local sin motivo.
    """
    backend = resolve_backend(settings)
    if backend == GLM:
        from crotolamo.core.glm import GLMClient, _find_api_key

        if _find_api_key() is not None:
            return GLMClient.from_settings(settings)

    from crotolamo.core.llm import LLMClient

    return LLMClient.from_settings(settings)


def build_llm(settings):
    """Construye el cliente del backend configurado.

    Con GLM, el cliente va envuelto en `FallbackLLM`: si la nube falla EN CALIENTE
    (se cayó el wifi, la key caducó, un 429), los turnos siguientes los atiende
    Ollama local. Sin eso, un corte de internet dejaba a Crotolamo respondiendo
    "¿Hay internet, patrón?" en vez de usar el modelo que ya está instalado.

    Si se pide GLM y no hay API key, ni siquiera se intenta: se usa Ollama directo.
    Más vale un Crotolamo lento que un Crotolamo mudo.
    """
    backend = resolve_backend(settings)

    from crotolamo.core.llm import LLMClient

    local = LLMClient.from_settings(settings)

    if backend == GLM:
        from crotolamo.core.glm import GLMClient, _find_api_key

        if _find_api_key() is None:
            log.warning(
                "backend=glm pero no hay API key (CROTOLAMO_GLM_API_KEY). "
                "Caigo a Ollama local."
            )
        else:
            from crotolamo.core.fallback import FallbackLLM

            remote = GLMClient.from_settings(settings)
            cooldown = settings.llm.get("glm", {}).get("fallback_cooldown_s", 60.0)
            log.info(
                "motor: GLM (%s) vía %s, con respaldo local (%s)",
                remote.model, remote.base_url, local.model,
            )
            return FallbackLLM(remote, local, cooldown_s=cooldown)

    log.info("motor: Ollama (%s) en %s", local.model, local.host)
    return local
