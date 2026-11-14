"""Tests del respaldo local: si GLM se cae, Crotolamo sigue vivo con Ollama."""

from __future__ import annotations

import logging

import pytest

from crotolamo.core.fallback import FallbackLLM
from crotolamo.core.glm import GLMAuthError
from crotolamo.core.llm import ChatResponse, LLMError, TransientLLMError


class _Fake:
    """Cliente de mentiras con el contrato de LLMClient.

    `falla=True` modela un fallo de DISPONIBILIDAD (timeout, wifi caído, 5xx):
    lanza TransientLLMError, que es lo único que debe abrir el breaker. Para
    errores permanentes (auth, 400) se pasa `error=` explícito.
    """

    def __init__(self, name: str, falla: bool = False, tokens: tuple[str, ...] = (),
                 error: Exception | None = None) -> None:
        self.model = name
        self.falla = falla
        self.tokens = tokens
        self.error = error
        self.llamadas = 0

    def _boom(self):
        if self.error is not None:
            raise self.error
        if self.falla:
            raise TransientLLMError(f"{self.model} caído")

    def chat(self, messages, tools=None):
        self.llamadas += 1
        self._boom()
        return ChatResponse(content=f"soy {self.model}")

    def chat_stream(self, messages, tools=None, on_token=None):
        self.llamadas += 1
        for t in self.tokens:
            if on_token:
                on_token(t)
        self._boom()
        return ChatResponse(content=f"soy {self.model}")


def test_usa_el_primario_cuando_funciona():
    nube, local = _Fake("glm"), _Fake("ollama")
    assert FallbackLLM(nube, local).chat([]).content == "soy glm"
    assert local.llamadas == 0


def test_cae_al_local_si_el_primario_revienta():
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    assert FallbackLLM(nube, local).chat([]).content == "soy ollama"


def test_circuit_breaker_no_reintenta_la_nube_durante_el_cooldown():
    """Sin esto, cada turno pagaría el timeout de la nube caída."""
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=300)

    for _ in range(4):
        assert llm.chat([]).content == "soy ollama"

    assert nube.llamadas == 1  # solo el primer intento
    assert local.llamadas == 4


def test_tras_el_cooldown_reintenta_la_nube(monkeypatch):
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=60)

    reloj = {"t": 1000.0}
    monkeypatch.setattr("crotolamo.core.fallback.time.monotonic", lambda: reloj["t"])

    assert llm.chat([]).content == "soy ollama"   # trip
    assert nube.llamadas == 1

    reloj["t"] += 61                              # pasó el cooldown
    nube.falla = False                            # volvió el wifi
    assert llm.chat([]).content == "soy glm"
    assert nube.llamadas == 2


def test_recuperarse_limpia_el_breaker(monkeypatch, caplog):
    """Tras caerse y volver, el breaker queda como recién estrenado: la vuelta se
    anota UNA vez y no en cada turno bueno que venga después.

    (Antes este test nunca tiraba el breaker: la nube funcionaba desde la
    primera llamada, y pasaba igual con `_recover` convertido en un no-op.)
    """
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=60)

    reloj = {"t": 1000.0}
    monkeypatch.setattr("crotolamo.core.fallback.time.monotonic", lambda: reloj["t"])

    assert llm.chat([]).content == "soy ollama"   # trip
    assert llm.degraded

    reloj["t"] += 61                              # pasó el cooldown
    nube.falla = False                            # volvió el wifi
    with caplog.at_level(logging.INFO, logger="crotolamo.core.fallback"):
        assert llm.chat([]).content == "soy glm"
        assert llm.chat_stream([]).content == "soy glm"   # ya recuperado: no re-avisa

    assert llm.active is nube and not llm.degraded
    assert caplog.text.count("el motor principal volvió") == 1


def test_streaming_cae_al_local_si_falla_ANTES_de_emitir():
    nube = _Fake("glm", falla=True, tokens=())      # revienta sin hablar
    local = _Fake("ollama", tokens=("hola ", "patrón"))
    dichos: list[str] = []

    r = FallbackLLM(nube, local).chat_stream([], on_token=dichos.append)
    assert r.content == "soy ollama"
    assert dichos == ["hola ", "patrón"]


def test_streaming_NO_reintenta_si_ya_habló():
    """Reintentar repetiría el principio de la frase por el altavoz."""
    nube = _Fake("glm", falla=True, tokens=("Órale ", "patrón"))
    local = _Fake("ollama", tokens=("hola",))
    dichos: list[str] = []

    with pytest.raises(LLMError):
        FallbackLLM(nube, local).chat_stream([], on_token=dichos.append)

    assert dichos == ["Órale ", "patrón"]  # no se repitió
    assert local.llamadas == 0


def test_model_refleja_el_cliente_activo():
    nube, local = _Fake("glm", falla=True), _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=300)
    assert llm.model == "glm"
    llm.chat([])                # trip
    assert llm.model == "ollama"


def test_si_ambos_caen_propaga_el_error_del_local():
    nube, local = _Fake("glm", falla=True), _Fake("ollama", falla=True)
    with pytest.raises(LLMError, match="ollama"):
        FallbackLLM(nube, local).chat([])


def test_toolagent_completa_el_turno_si_el_primario_cae_a_media_iteracion(monkeypatch):
    """Integración ToolAgent + FallbackLLM: la nube pide una tool y REVIENTA en la
    2ª iteración del loop; el respaldo local debe redactar la respuesta final."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry, desktop

    monkeypatch.setattr(desktop, "run_detached", lambda args: None)

    class _Primario:
        model = "glm"

        def __init__(self):
            self.llamadas = 0

        def chat(self, messages, tools=None):
            self.llamadas += 1
            if self.llamadas == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[{"name": "open_url", "arguments": {"url": "x.com"}}],
                    raw_message={"tool_calls": [{
                        "function": {"name": "open_url", "arguments": {"url": "x.com"}},
                    }]},
                )
            raise TransientLLMError("glm caído a media faena")

    class _Local:
        model = "ollama"

        def chat(self, messages, tools=None):
            return ChatResponse(content="Abierta desde el local, patrón.")

    nube = _Primario()
    llm = FallbackLLM(nube, _Local())
    agent = ToolAgent(
        llm, Conversation("SYS"), registry=default_registry(),
        guard=Guard(allowed_roots=[]), fastpath=False,
    )

    reply = agent.handle_turn("abre x.com")
    assert reply == "Abierta desde el local, patrón."
    assert nube.llamadas == 2  # sirvió la 1ª iteración y reventó en la 2ª


def test_build_llm_envuelve_en_fallback_cuando_hay_key(monkeypatch):
    from crotolamo.core.engine import build_llm
    from crotolamo.core.glm import GLMClient
    from crotolamo.core.llm import LLMClient
    from crotolamo.settings import get_settings

    monkeypatch.setenv("CROTOLAMO_LLM_BACKEND", "glm")
    monkeypatch.setenv("CROTOLAMO_GLM_API_KEY", "secreta")

    llm = build_llm(get_settings())
    assert isinstance(llm, FallbackLLM)
    assert isinstance(llm.primary, GLMClient)
    assert isinstance(llm.secondary, LLMClient)


# --- T6: el breaker SOLO se abre con errores transitorios ---

def test_error_permanente_se_propaga_sin_abrir_el_breaker():
    """Un 400 (payload mal formado) es un BUG, no una caída: degradar 60s a
    Ollama en silencio lo escondería. Se propaga y la nube se reintenta al
    siguiente turno."""
    nube = _Fake("glm", error=LLMError("GLM respondió 400, patrón: payload roto"))
    local = _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=300)

    with pytest.raises(LLMError, match="400"):
        llm.chat([])
    assert local.llamadas == 0        # nada de degradar en silencio
    assert llm.active is nube          # el breaker NO se abrió

    with pytest.raises(LLMError, match="400"):
        llm.chat([])
    assert nube.llamadas == 2          # se volvió a intentar (sin cooldown)


def test_auth_error_cae_al_local_sin_reintentar_la_nube():
    """Una key inválida no se arregla sola: reintentarla cada turno es ruido.
    Se marca la nube como caída hasta reinicio y se usa el local."""
    nube = _Fake("glm", error=GLMAuthError("GLM me rechazó la credencial (401), patrón."))
    local = _Fake("ollama")
    llm = FallbackLLM(nube, local, cooldown_s=0.0)  # sin cooldown: probaría la nube SIEMPRE

    for _ in range(3):
        assert llm.chat([]).content == "soy ollama"

    assert nube.llamadas == 1          # un solo intento, no un bucle de 401s
    assert llm.model == "ollama"


def test_streaming_error_permanente_tambien_se_propaga():
    nube = _Fake("glm", error=LLMError("GLM respondió 400, patrón."))
    local = _Fake("ollama")
    llm = FallbackLLM(nube, local)
    with pytest.raises(LLMError, match="400"):
        llm.chat_stream([])
    assert local.llamadas == 0
    assert llm.active is nube


# --- T6: clasificación en los clientes ---

def test_clientes_clasifican_transitorio_vs_permanente(monkeypatch):
    from tests.test_transport import _Resp, _glm, _ollama

    # GLM: 500 y timeout son transitorios; 400 es permanente; 401 es auth.
    client, _ = _glm(monkeypatch, [_Resp(status=500, body=b"boom")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client, _ = _glm(monkeypatch, [TimeoutError("timed out")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client, _ = _glm(monkeypatch, [_Resp(status=400, body=b"bad request")])
    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "hola"}])
    assert not isinstance(exc.value, TransientLLMError)

    client, _ = _glm(monkeypatch, [_Resp(status=401, body=b"bad key")])
    with pytest.raises(GLMAuthError) as exc:
        client.chat([{"role": "user", "content": "hola"}])
    assert not isinstance(exc.value, TransientLLMError)

    # Ollama: 500/conexión transitorios; 404 (modelo inexistente) permanente.
    client = _ollama([_Resp(status=500, body=b"panic")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client = _ollama([ConnectionRefusedError("refused")])
    with pytest.raises(TransientLLMError):
        client.chat([{"role": "user", "content": "hola"}])

    client = _ollama([_Resp(status=404, body=b"model not found")])
    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "hola"}])
    assert not isinstance(exc.value, TransientLLMError)


# --- Lo retenido de la nube nunca se mezcla con la respuesta local -----------
# (causa raíz "el turno del LLM es una llamada atómica y bloqueante", paso 6).
# "Ya habló" sigue siendo lo que on_token devuelve (False = retenido, ver
# LiveStreamer.feed); lo nuevo es que, antes de repetir la llamada en local,
# FallbackLLM reinicia el streamer para que lo retenido no salga pegado.

_LOCAL = ("Hola, ", "patrón. ", "Todo ", "bien.")


def _streamer(dichos, modo):
    from crotolamo.core.streaming import LiveStreamer

    if modo == "final":       # tools a la vista, lo de siempre
        return LiveStreamer(dichos.append, hold_until_done=True)
    if modo == "sentence":    # tools a la vista, opt-in: retiene hasta cerrar frase
        return LiveStreamer(dichos.append, hold_until_done=True, release="sentence")
    return LiveStreamer(dichos.append)   # charla: retiene si empieza como JSON


@pytest.mark.parametrize("modo, tokens_nube", [
    ("final", ("¡Claro que sí, ", "patrón! Voy a")),
    ("sentence", ("La nube iba diciendo", " algo sin cerrar")),
    ("charla", ('{"name": "ram', '_usage"')),
])
@pytest.mark.parametrize("corte", [TransientLLMError("glm caído"),
                                   GLMAuthError("GLM me rechazó la credencial (401)")])
def test_lo_retenido_de_la_nube_no_se_mezcla_con_el_local(modo, tokens_nube, corte):
    nube = _Fake("glm", tokens=tokens_nube, error=corte)
    local = _Fake("ollama", tokens=_LOCAL)
    dichos: list[str] = []
    streamer = _streamer(dichos, modo)

    r = FallbackLLM(nube, local).chat_stream([], on_token=streamer)
    streamer.flush_if_held("".join(_LOCAL))

    assert local.llamadas == 1
    assert r.content == "soy ollama"
    assert "".join(dichos) == "".join(_LOCAL)   # ni un trozo de la nube


def test_restart_por_defecto_no_cambia_los_trozos_que_se_oyen():
    """Sin tools a la vista, la nube empezó como JSON (retener) y se cayó: antes
    del paso 6 el local seguía retenido y se oía de UNA vez el texto final (con
    los post-hooks). `restart` no puede deshacer esa decisión: compara los
    trozos, no solo lo concatenado."""
    nube = _Fake("glm", tokens=('{"', "name"), falla=True)
    local = _Fake("ollama", tokens=_LOCAL)
    dichos: list[str] = []
    streamer = _streamer(dichos, "charla")

    FallbackLLM(nube, local).chat_stream([], on_token=streamer)
    assert dichos == []                 # el local también quedó retenido
    streamer.flush_if_held("Hola, patrón. Todo bien.")

    assert dichos == ["Hola, patrón. Todo bien."]


def _agente_sin_tools(llm):
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry

    # route_fn vacío: el turno no lleva tools (charla), hold_until_done=False.
    return ToolAgent(llm, Conversation("SYS"), registry=default_registry(),
                     guard=Guard(allowed_roots=[]), fastpath=False, route_fn=lambda t: [])


def test_toolagent_charla_tras_json_de_la_nube_oye_lo_de_siempre():
    """El mismo caso por el turno completo (lo que reporta la revisión con
    6508165 como referencia): un único trozo, el texto final."""
    nube = _Fake("glm", tokens=('{"', "name"), falla=True)
    local = _Fake("ollama", tokens=_LOCAL)
    dichos: list[str] = []

    reply = _agente_sin_tools(FallbackLLM(nube, local)).handle_turn("hola", on_token=dichos.append)

    assert (reply, dichos) == ("soy ollama", ["soy ollama"])


def test_toolagent_charla_si_tambien_cae_el_local_solo_se_oye_el_error():
    nube = _Fake("glm", tokens=('{"',), falla=True)
    local = _Fake("ollama", tokens=("Hola", " pa"), falla=True)
    dichos: list[str] = []

    reply = _agente_sin_tools(FallbackLLM(nube, local)).handle_turn("hola", on_token=dichos.append)

    assert (reply, dichos) == ("ollama caído", ["ollama caído"])


def test_sin_reiniciar_lo_retenido_se_habria_pegado():
    """Control del anterior: el streamer "sentence" sin `restart` mezclaba."""
    nube = _Fake("glm", tokens=("La nube iba diciendo",), falla=True)
    local = _Fake("ollama", tokens=_LOCAL)
    dichos: list[str] = []
    streamer = _streamer(dichos, "sentence")

    # `streamer.feed` (método suelto): FallbackLLM no alcanza `restart`.
    FallbackLLM(nube, local).chat_stream([], on_token=streamer.feed)

    assert "".join(dichos).startswith("La nube iba diciendoHola")


def test_no_reinicia_ni_reintenta_si_ya_salio_texto():
    class _Espia:
        def __init__(self):
            self.dichos: list[str] = []
            self.reinicios = 0

        def __call__(self, texto):
            self.dichos.append(texto)
            return True          # salió hacia el patrón

        def restart(self):
            self.reinicios += 1

    nube = _Fake("glm", tokens=("Órale ",), falla=True)
    local = _Fake("ollama", tokens=("hola",))
    espia = _Espia()

    with pytest.raises(LLMError):
        FallbackLLM(nube, local).chat_stream([], on_token=espia)

    assert (espia.dichos, espia.reinicios, local.llamadas) == (["Órale "], 0, 0)


def test_un_on_token_sin_restart_sigue_funcionando():
    """Cualquier callable vale: sin `restart` no hay nada que reiniciar."""
    nube = _Fake("glm", tokens=("retenido",), falla=True)
    local = _Fake("ollama", tokens=("hola",))
    dichos: list[str] = []

    def retiene(texto):
        dichos.append(texto)
        return False

    r = FallbackLLM(nube, local).chat_stream([], on_token=retiene)
    assert r.content == "soy ollama"
    assert dichos == ["retenido", "hola"]


def test_toolagent_sentence_la_nube_cae_sin_haber_hablado_y_responde_el_local_limpio():
    """El turno completo con el opt-in: la nube suelta media frase (retenida por
    "sentence"), se cae el wifi y el local contesta; se oye SOLO el local."""
    from crotolamo.core.agent import ToolAgent
    from crotolamo.core.memory import Conversation
    from crotolamo.safety.guard import Guard
    from crotolamo.tools import default_registry

    nube = _Fake("glm", tokens=("La nube iba", " diciendo algo"), falla=True)
    local = _Fake("ollama", tokens=_LOCAL)
    agent = ToolAgent(FallbackLLM(nube, local), Conversation("SYS"),
                      registry=default_registry(), guard=Guard(allowed_roots=[]),
                      fastpath=False, stream_release="sentence")
    dichos: list[str] = []

    reply = agent.handle_turn("hola", on_token=dichos.append)

    assert reply == "soy ollama"
    assert "".join(dichos) == "".join(_LOCAL)


class _ConPreconnect(_Fake):
    def __init__(self, name, resultado=True, error=None):
        super().__init__(name)
        self.resultado = resultado
        self.error_pre = error
        self.preconexiones = 0

    def preconnect(self):
        self.preconexiones += 1
        if self.error_pre is not None:
            raise self.error_pre
        return self.resultado


def test_preconnect_delega_en_el_motor_activo():
    nube, local = _ConPreconnect("glm"), _ConPreconnect("ollama", resultado=False)
    llm = FallbackLLM(nube, local, cooldown_s=300)

    assert llm.preconnect() is True
    assert (nube.preconexiones, local.preconexiones) == (1, 0)

    llm._trip(TransientLLMError("glm caído"))          # breaker abierto: manda el local
    assert llm.preconnect() is False
    assert (nube.preconexiones, local.preconexiones) == (1, 1)


def test_preconnect_nunca_lanza_y_sin_soporte_es_false():
    assert FallbackLLM(_Fake("glm"), _Fake("ollama")).preconnect() is False
    roto = _ConPreconnect("glm", error=OSError("sin red"))
    assert FallbackLLM(roto, _Fake("ollama")).preconnect() is False


# --- Un turno cancelado (barge-in) no sigue gastando en ningún motor -----------
# El punto de control del consumidor (`LiveStreamer.checkpoint`) lo consultan los
# parsers con cada línea del stream; FallbackLLM lo deja a la vista del primario
# a través de su envoltorio y lo consulta antes de arrancar el local.


class _ConsultaElControl(_Fake):
    """Motor que, como los parsers reales, consulta el punto de control del
    on_token que recibe con cada evento (aquí, uno antes de cada token)."""

    def chat_stream(self, messages, tools=None, on_token=None):
        self.llamadas += 1
        checkpoint = getattr(on_token, "checkpoint", None)
        for t in self.tokens or ("",):
            if checkpoint is not None:
                checkpoint()
            if on_token and t:
                on_token(t)
        self._boom()
        return ChatResponse(content=f"soy {self.model}")


def test_el_primario_ve_el_punto_de_control_a_traves_del_envoltorio():
    from crotolamo.core.streaming import LiveStreamer, TurnCancelled

    nube = _ConsultaElControl("glm", tokens=("",))      # un tool-call: sin texto
    local = _Fake("ollama", tokens=_LOCAL)
    llm = FallbackLLM(nube, local)
    streamer = LiveStreamer([].append, hold_until_done=True, should_abort=lambda: True)

    with pytest.raises(TurnCancelled):
        llm.chat_stream([], on_token=streamer)

    # TurnCancelled no es un fallo del motor: ni breaker ni local.
    assert (nube.llamadas, local.llamadas, llm.degraded) == (1, 0, False)


def test_cancelado_mientras_la_nube_cae_no_arranca_el_local():
    """El patrón interrumpe y la nube se cae en la misma llamada: el fallo de la
    nube cuenta (breaker), pero el local (segundos de prompt-eval en CPU) ya no
    se arranca para un turno que nadie va a oír."""
    from crotolamo.core.streaming import LiveStreamer, TurnCancelled

    vigente = {"ok": True}

    class _NubeQueCae(_Fake):
        def chat_stream(self, messages, tools=None, on_token=None):
            vigente["ok"] = False          # el barge-in llega mientras se cae
            return super().chat_stream(messages, tools, on_token)

    nube = _NubeQueCae("glm", falla=True)
    local = _Fake("ollama", tokens=_LOCAL)
    llm = FallbackLLM(nube, local)
    streamer = LiveStreamer([].append, hold_until_done=True,
                            should_abort=lambda: not vigente["ok"])

    with pytest.raises(TurnCancelled):
        llm.chat_stream([], on_token=streamer)

    assert local.llamadas == 0
    assert llm.degraded


def test_sin_should_abort_el_punto_de_control_no_cambia_nada():
    """Por defecto (sin should_abort) consultar el punto de control no hace
    nada: la nube cae con todo retenido, responde el local y se oye lo de
    siempre, de una vez."""
    from crotolamo.core.streaming import LiveStreamer

    dichos: list[str] = []
    nube = _ConsultaElControl("glm", tokens=("¡Claro que ", "sí, patrón!"), falla=True)
    local = _ConsultaElControl("ollama", tokens=_LOCAL)
    streamer = LiveStreamer(dichos.append, hold_until_done=True)

    r = FallbackLLM(nube, local).chat_stream([], on_token=streamer)
    streamer.flush_if_held("".join(_LOCAL))

    assert r.content == "soy ollama"
    assert (nube.llamadas, local.llamadas) == (1, 1)
    assert dichos == ["".join(_LOCAL)]
