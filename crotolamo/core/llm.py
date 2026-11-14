"""Cliente de Ollama con tool-calling. Reescritura de C1::ask_ollama.

Habla /api/chat por HTTP (stdlib, sin dependencias) sobre una conexión
persistente (`HTTPTransport`, keep-alive). Soporta el campo `tools` para
tool-calling nativo de qwen2.5-coder. Los errores se devuelven en personaje.
"""

from __future__ import annotations

import http.client
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn

from crotolamo.core.engine import HTTPTransport
from crotolamo.core.tool_parsing import arguments_complete, parse_arguments
from crotolamo.logging_setup import get_logger

log = get_logger("core.llm")

# Instrumentación opcional: con CROTOLAMO_LLM_PROF=1 imprime tiempo y tokens por
# llamada a /api/chat (prompt_eval_count, eval_count). Útil para diagnosticar el
# costo de los tool-schemas en CPU. Cero overhead si la env no está puesta.
_PROF = os.environ.get("CROTOLAMO_LLM_PROF") == "1"


class LLMError(RuntimeError):
    """Error hablando con Ollama, ya con mensaje en personaje."""


class TransientLLMError(LLMError):
    """Fallo de DISPONIBILIDAD del motor: timeout, conexión caída, 5xx, 429.

    Es lo único que debe abrir el circuit breaker de FallbackLLM. Los errores
    permanentes o de cliente (400 por payload roto, 404 de modelo inexistente,
    auth rechazada, basura no-JSON) siguen siendo LLMError/GLMAuthError:
    degradar 60s al modelo local por un bug de formato solo esconde el bug.
    """


@dataclass
class ChatResponse:
    content: str = ""
    # Lista de tool calls pedidos: [{"name": str, "arguments": dict}, ...]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # El mensaje crudo del asistente (para reinyectar al historial tal cual).
    raw_message: dict[str, Any] = field(default_factory=dict)


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        # Ollama suele mandar arguments como dict; algunos modelos lo mandan
        # como string JSON. Normalizamos a dict.
        args = parse_arguments(fn.get("arguments", {}))
        calls.append({"name": fn.get("name", ""), "arguments": args})
    return calls


def _read_body(resp) -> str:
    """Cuerpo de una respuesta de error, best-effort y acotado."""
    try:
        return resp.read().decode("utf-8", "replace")[:300]
    except Exception:  # noqa: BLE001 - el cuerpo del error es best-effort
        return ""


class LLMClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        temperature: float = 0.2,
        timeout: float = 120,
        keep_alive: str = "15m",
        num_ctx: int | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        # num_ctx: ventana de contexto que Ollama reserva. En CPU, una ventana
        # más chica reduce memoria y trabajo del KV-cache; con el routing de
        # tools (prompts cortos) 2048 sobra. None = default del modelo.
        self.num_ctx = num_ctx
        # keep_alive: cuánto mantiene Ollama el modelo (y su cache de prefijo KV)
        # residente tras una respuesta. En CPU es CLAVE: el primer turno paga el
        # prompt-eval completo de los tool-schemas (~lento), pero si el modelo
        # sigue caliente los turnos siguientes reusan el cache y son baratos.
        # "15m" balancea rapidez en sesión vs. liberar RAM cuando no se usa.
        self.keep_alive = keep_alive
        # Conexión HTTP persistente: evita abrir TCP nuevo en cada petición.
        self._transport = HTTPTransport(self.host, timeout=self.timeout)

    @classmethod
    def from_settings(cls, settings) -> "LLMClient":
        llm = settings.llm
        return cls(
            host=llm.get("host", "http://localhost:11434"),
            model=llm.get("model", "qwen2.5-coder:7b"),
            temperature=llm.get("temperature", 0.2),
            timeout=llm.get("timeout", 120),
            keep_alive=llm.get("keep_alive", "15m"),
            num_ctx=llm.get("num_ctx"),
        )

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        return options

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": stream,
            "messages": messages,
            "keep_alive": self.keep_alive,
            "options": self._options(),
        }
        if tools:
            payload["tools"] = tools
        return payload

    def _raise_http(self, status: int, body: str, location: str | None = None) -> NoReturn:
        """Mapea un status HTTP >= 300 de Ollama a un LLMError útil.

        Antes un 404/500 caía en la rama genérica de "¿Está vivo el servicio?",
        que despista: el servicio SÍ está vivo, es la petición la que falló.
        """
        if status < 400:
            # Permanente: [llm].host apunta a algo que no es Ollama (un proxy).
            raise LLMError(
                f"Ollama me mandó a otra dirección ({status}) en vez de responder, patrón. "
                f"Revisa [llm].host; me redirige a {location or 'no sé dónde'}."
            )
        if status == 404:
            # Permanente: el modelo no existe; reintentar no lo instala.
            raise LLMError(
                f"Ollama respondió 404, patrón. ¿Existe el modelo '{self.model}'? "
                f"Prueba `ollama pull {self.model}`. {body}".rstrip()
            )
        if status >= 500:
            raise TransientLLMError(
                f"Ollama tropezó por dentro ({status}), patrón: {body}"
            )
        raise LLMError(f"Ollama respondió {status}, patrón: {body}")

    def _send(self, payload: dict[str, Any]) -> http.client.HTTPResponse:
        """POST a /api/chat con los errores de transporte ya en personaje."""
        try:
            resp = self._transport.post_json("/api/chat", payload)
        except TimeoutError as error:
            raise TransientLLMError(
                f"Ollama no respondió a tiempo, patrón. ({error})"
            ) from error
        except (http.client.HTTPException, OSError) as error:
            raise TransientLLMError(
                f"No pude hablar con Ollama en {self.host}, patrón. "
                f"¿Está vivo el servicio? ({error})"
            ) from error
        # >= 300 y no >= 400: http.client NO sigue redirecciones; un 3xx pasaba
        # por respuesta y su cuerpo vacío, por una contestación en blanco.
        if resp.status >= 300:
            self._raise_http(resp.status, _read_body(resp), resp.getheader("Location"))
        return resp

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        t0 = time.time() if _PROF else 0.0
        resp = self._send(self._payload(messages, tools, stream=False))
        try:
            raw = json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, OSError) as error:
            # El cuerpo no llegó entero: el socket queda a medias. Se cuelga ya;
            # abierto, Ollama seguiría generando (y quemando CPU) para nadie.
            self._transport.close()
            raise TransientLLMError(
                f"Ollama no respondió a tiempo, patrón. ({error})"
            ) from error
        except http.client.HTTPException as error:
            # Cuerpo cortado a medias (IncompleteRead: el servidor o un proxy
            # cerraron antes de mandar lo prometido). Es HTTPException, NO
            # OSError: sin esta rama salía cruda de handle_turn (traceback) y
            # sin abrir el breaker de FallbackLLM, aunque es pura disponibilidad.
            self._transport.close()
            raise TransientLLMError(
                f"Ollama me cortó la respuesta a medias, patrón. ({error})"
            ) from error
        except ValueError as error:
            # JSONDecodeError y también UnicodeDecodeError (cuerpo que ni es
            # UTF-8: la página en latin-1 de un portal cautivo). Las dos son
            # ValueError; atrapando solo la primera, la segunda salía cruda.
            raise LLMError("Ollama me devolvió basura no-JSON, patrón.") from error

        if _PROF:
            n_tools = len(tools) if tools else 0
            print(
                f"[LLM-PROF] {time.time() - t0:5.1f}s | "
                f"prompt_eval={raw.get('prompt_eval_count', '?')} "
                f"gen={raw.get('eval_count', '?')} | tools_enviadas={n_tools}",
                flush=True,
            )

        message = raw.get("message", {}) or {}
        return ChatResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=_parse_tool_calls(message),
            raw_message=message,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token=None,
    ) -> ChatResponse:
        """Como chat() pero con stream=True: invoca on_token(chunk) por cada delta.

        Acumula el contenido y los tool_calls de todo el stream. Útil para
        respuesta token-a-token (Fase 6). Funciona mejor con GPU o modelos que usan
        el campo nativo tool_calls.

        Cierra el ciclo de la respuesta: al ver done=true avisa al transporte
        (`finish`) para reutilizar el keep-alive; ante un corte, un fallo o un
        consumidor que aborta, cuelga el socket, que es lo que hace que Ollama
        deje de generar.
        """
        resp = self._send(self._payload(messages, tools, stream=True))
        try:
            content, message, complete = self._parse_stream(resp, on_token)
        except (TimeoutError, OSError) as error:
            self._transport.close()
            raise TransientLLMError(
                f"Ollama no respondió a tiempo, patrón. ({error})") from error
        except http.client.HTTPException as error:  # stream cortado: ver chat()
            self._transport.close()
            raise TransientLLMError(
                f"Ollama me cortó la respuesta a medias, patrón. ({error})") from error
        except BaseException:
            # Error dentro del stream, corte sin nada usable (ver _parse_stream)
            # o el consumidor abortó (on_token lanzó): colgar YA. Con el socket
            # abierto Ollama sigue generando (y quemando CPU) para nadie.
            self._transport.close()
            raise
        if complete:
            self._transport.finish(resp)  # la cola se drena en la próxima petición
        else:
            # Final limpio sin done=true: se acepta (ver _parse_stream), pero
            # queda dicho en el log. Esa conexión no se reutiliza: lo normal es
            # que la cortara el servidor, y reusarla solo costaría una escritura
            # fallida más el reintento.
            log.warning(
                "Ollama cerró el stream sin done=true: respuesta posiblemente "
                "truncada (%d caracteres, %d tool_calls)",
                len(content), len(message.get("tool_calls") or []),
            )
            self._transport.close()

        return ChatResponse(
            content=content.strip(),
            tool_calls=_parse_tool_calls(message),
            raw_message=message,
        )

    @staticmethod
    def _consume_stream(resp, on_token) -> tuple[str, dict[str, Any]]:
        """Como `_parse_stream`, sin el aviso de fin lógico (tests y llamadores viejos)."""
        content, message, _complete = LLMClient._parse_stream(resp, on_token)
        return content, message

    @staticmethod
    def _parse_stream(resp, on_token) -> tuple[str, dict[str, Any], bool]:
        """Parsea el JSONL de /api/chat?stream=true. Reusable y testeable.

        Devuelve (contenido, mensaje del turno ENTERO, ¿llegó done=true?). Con
        el tercer valor el cliente le avisa al transporte que el socket se
        puede reutilizar. Los tool_calls llegan completos (no fragmentados
        como en el SSE de OpenAI) pero en un chunk con done=false, y el stream
        cierra con un chunk done=true cuyo message viene vacío. Quedarse con
        el último message borraba la llamada: la tool no corría y el turno se
        cerraba en falso. Si el chunk final SÍ trae tool_calls (formato
        antiguo, o la misma llamada repetida al cerrar), mandan esos: así
        ninguna se pierde ni se ejecuta dos veces.

        Lanza TransientLLMError si Ollama avisa de un fallo dentro del stream
        (una línea {"error": ...}, ya con el 200 dado). Si el stream se acaba
        sin done=true, lanza el mismo error cuando no deja nada usable (ni
        texto ni tool_calls, o unos argumentos a medias); con algo usable lo
        devuelve con False, igual que `parse_sse` (ver allí el porqué).

        Como `parse_sse`, consulta el `checkpoint` del on_token (si lo trae)
        con CADA línea: un chunk de razonamiento (`thinking`) o de tool_calls
        llega con content vacío y no pasa por on_token.
        """
        checkpoint = getattr(on_token, "checkpoint", None)
        parts: list[str] = []
        early_calls: list[Any] = []
        final_calls: list[Any] = []
        # Que el HTTP no se queje no basta: un FIN entre dos chunks acaba la
        # iteración por líneas sin error. El cierre de verdad es done=true.
        complete = False
        for raw_line in resp:
            if checkpoint is not None:
                checkpoint()
            # "replace": una línea que ni es UTF-8 es basura como cualquier otra
            # y la descarta el filtro de abajo; con el codec estricto reventaba
            # aquí con un UnicodeDecodeError crudo antes de llegar a él.
            line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue  # JSON válido que no es un chunk (null, una lista): basura
            if obj.get("error"):
                # Transitorio, como el 500 que Ollama da si falla ANTES de
                # empezar a responder: se le cayó el runner, se quedó sin RAM.
                raise TransientLLMError(
                    f"Ollama tropezó a media respuesta, patrón: {str(obj['error'])[:300]}"
                )
            msg = obj.get("message", {}) or {}
            # Varias llamadas pueden venir en un chunk o repartidas en varios.
            if obj.get("done"):
                final_calls.extend(msg.get("tool_calls") or [])
            else:
                early_calls.extend(msg.get("tool_calls") or [])
            delta = msg.get("content") or ""
            if delta:
                parts.append(delta)
                if on_token:
                    on_token(delta)
            if obj.get("done"):
                complete = True
                break
        calls = final_calls or early_calls
        content = "".join(parts)
        if not complete and (
            not (content.strip() or calls)
            or not all(
                arguments_complete((call.get("function") or {}).get("arguments", {}))
                for call in calls
            )
        ):
            raise TransientLLMError(
                "Ollama me cortó la respuesta a medias, patrón. (el stream acabó sin done)"
            )
        # Misma forma que devuelve `consume_sse`: `tool_calls` solo si hubo.
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        return content, message, complete
