"""Presupuesto de tamaño del request (crotolamo/core/budget.py).

Lo que se le exige: es PURO (nunca toca la lista de entrada), devuelve el MISMO
objeto si ya cabe, solo tira historia vieja por bloques enteros (un tool nunca
se queda sin el assistant que lo pidió), conserva siempre la cabeza de system y
el bloque en curso, y al final cabe: est_tokens <= max_tokens - REPLY_RESERVE.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from crotolamo.core import budget
from crotolamo.core.openai_adapter import to_openai_messages
from crotolamo.tools.base import truncate_for_context

PERSONA = "Eres Crotolamo, el asistente del patrón. " * 60  # ~2.4 KB, como la real
SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee un archivo de texto y devuelve su contenido.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "ruta"}},
                "required": ["path"],
            },
        },
    },
]


def _turno(i: int, resultado: str, cerrado: bool = True) -> list[dict]:
    """Un bloque como lo guarda Conversation: user, assistant que pide una tool,
    su resultado y (si el turno terminó) la respuesta final."""
    bloque = [
        {"role": "user", "content": f"lee el archivo {i}, patrón"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": f"/home/p/{i}.txt"}}},
        ]},
        {"role": "tool", "content": resultado, "name": "read_file"},
    ]
    if cerrado:
        bloque.append({"role": "assistant", "content": f"El archivo {i} dice cosas, patrón."})
    return bloque


def _historia(turnos: int, chars: int, cabeza: list[dict] | None = None) -> list[dict]:
    msgs = cabeza if cabeza is not None else [{"role": "system", "content": PERSONA}]
    msgs = list(msgs)
    for i in range(turnos - 1):
        msgs += _turno(i, f"{i:05d}" + "x" * (chars - 5))
    # El último turno, en curso: el modelo aún no ha visto el resultado.
    msgs += _turno(turnos - 1, "ULTIMO" + "y" * (chars - 6), cerrado=False)
    return msgs


def _bytes(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Unidades
# ---------------------------------------------------------------------------

def test_constantes_del_presupuesto():
    # TOOL_RESULT_CAP queda por encima de lo que ya devuelven las tools con tope
    # (read_file 20 000 + marca, read_project_file + cabecera, MCP + prefijo).
    assert budget.TOOL_RESULT_CAP == 24_000
    assert budget.BYTES_PER_TOKEN == 3.3
    assert budget.REPLY_RESERVE == 256


def test_est_tokens_son_los_bytes_json_del_payload_entre_3_3():
    msgs = [{"role": "system", "content": "SYS ñ"}, {"role": "user", "content": "hola"}]
    esperado = math.ceil((_bytes(msgs) + _bytes(SCHEMAS)) / 3.3)
    assert budget.est_tokens(msgs, SCHEMAS) == esperado
    # Sin tools no se cuentan (los clientes no mandan la clave "tools").
    assert budget.est_tokens(msgs) == math.ceil(_bytes(msgs) / 3.3)
    assert budget.est_tokens(msgs, []) == budget.est_tokens(msgs, None)


# ---------------------------------------------------------------------------
# (a) Lo que ya cabe sale intacto: el MISMO objeto
# ---------------------------------------------------------------------------

def test_un_payload_que_cabe_devuelve_el_mismo_objeto():
    msgs = _historia(3, 500)
    assert budget.fit_messages(msgs, SCHEMAS, 128_000) is msgs
    # Con tope de tool que ningún resultado supera, igual.
    assert budget.fit_messages(msgs, SCHEMAS, 128_000, tool_cap=500) is msgs


def test_cabe_pero_un_tool_pasa_el_tope_solo_se_recorta_ese():
    msgs = _historia(3, 5_000)
    original = copy.deepcopy(msgs)
    out = budget.fit_messages(msgs, SCHEMAS, 128_000, tool_cap=1_000)
    assert out is not msgs
    assert msgs == original  # la entrada no se toca
    assert len(out) == len(msgs)  # no se tira nada: ya cabía
    for antes, despues in zip(msgs, out):
        if antes["role"] == "tool":
            # La misma marca que el resto del proyecto, con el total original.
            assert despues["content"] == truncate_for_context(antes["content"], 1_000)
            assert despues["name"] == antes["name"]
        else:
            assert despues is antes


# ---------------------------------------------------------------------------
# (b), (c), (d): historia de 30 turnos con resultados de 20 KB
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_tokens,tool_cap", [
    (2_048, 2_365),      # el local de fábrica, con el tope que le dará su cliente
    (2_048, None),       # sin tope: el bloque en curso se recorta a mitades
    (128_000, None),     # la red de seguridad de la nube
])
def test_historia_larga_cabe_sin_huerfanos_ni_perder_cabeza(max_tokens, tool_cap):
    cabeza = [
        {"role": "system", "content": PERSONA},
        {"role": "system", "content": "[contexto que ya sabes sobre el patrón:\n- 1: café]"},
        {"role": "system", "content": "[resumen de la conversación previa]\nhablaron de café"},
    ]
    msgs = _historia(30, 20_000, cabeza=cabeza)
    original = copy.deepcopy(msgs)
    assert budget.est_tokens(msgs, SCHEMAS) > max_tokens  # de verdad no cabe

    out = budget.fit_messages(msgs, SCHEMAS, max_tokens, tool_cap=tool_cap)

    assert msgs == original  # puro
    # (d) cabe, con sitio para la respuesta.
    assert budget.est_tokens(out, SCHEMAS) <= max_tokens - budget.REPLY_RESERVE
    # (c) toda la cabeza de system, en orden, y el último user.
    assert out[:3] == cabeza
    ultimo_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    assert out[-3]["content"] == msgs[ultimo_user]["content"]
    assert out[-1]["role"] == "tool" and out[-1]["content"].startswith("ULTIMO")
    # (b) ningún tool huérfano: el adaptador OpenAI descarta los que no tienen
    # su assistant delante, así que no puede faltar ninguno.
    assert len(to_openai_messages(out)) == len(out)
    # La historia vieja se tira por bloques ENTEROS y de la más vieja a la más
    # nueva: lo que queda es una cola de la original, empezando por un user.
    resto = out[3:]
    assert resto[0]["role"] == "user"
    cola = msgs[len(msgs) - len(resto):]
    assert [m["role"] for m in resto] == [m["role"] for m in cola]
    assert [m["content"] for m in resto if m["role"] == "user"] == [
        m["content"] for m in cola if m["role"] == "user"
    ]


def test_el_bloque_en_curso_se_recorta_a_mitades_hasta_300():
    # Tres lecturas de 20 KB en el turno en curso y nada más que tirar.
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "lee tres"}]
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "read_file", "arguments": {"path": str(i)}}} for i in range(3)
    ]})
    msgs += [{"role": "tool", "content": "z" * 20_000, "name": "read_file"} for _ in range(3)]

    out = budget.fit_messages(msgs, None, 2_048)
    assert budget.est_tokens(out) <= 2_048 - budget.REPLY_RESERVE
    assert len(out) == len(msgs)
    caps = {len(m["content"].split("\n...[recortado")[0]) for m in out if m["role"] == "tool"}
    assert len(caps) == 1 and 300 <= caps.pop() < 20_000
    # Cada recorte sale del ORIGINAL (una sola marca, con el total de verdad).
    for m in out[3:]:
        assert m["content"].count("[recortado") == 1
        assert m["content"].endswith("[recortado, 20000 chars en total]")


def test_si_ni_con_300_cabe_devuelve_lo_mejor_posible_sin_tirar_el_turno():
    msgs = [{"role": "system", "content": "S" * 30_000}, {"role": "user", "content": "hola"}]
    msgs.append({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "read_file", "arguments": {}}},
    ]})
    msgs.append({"role": "tool", "content": "z" * 20_000, "name": "read_file"})

    out = budget.fit_messages(msgs, None, 2_048)
    # La persona sola ya no cabe: no se inventa nada, pero no se pierde el turno
    # y el resultado se queda en el suelo de 300.
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[-1]["content"] == truncate_for_context("z" * 20_000, budget.MIN_TOOL_CHARS)


def test_sin_user_no_hay_nada_que_tirar():
    msgs = [{"role": "system", "content": "S" * 30_000}]
    assert budget.fit_messages(msgs, None, 2_048) == msgs


def test_fit_informa_lo_que_tiro_y_lo_que_recorto():
    msgs = _historia(30, 20_000)
    resultado = budget.fit(msgs, SCHEMAS, 128_000)
    assert resultado.messages == budget.fit_messages(msgs, SCHEMAS, 128_000)
    assert resultado.est_before == budget.est_tokens(msgs, SCHEMAS)
    assert resultado.est_after == budget.est_tokens(resultado.messages, SCHEMAS)
    assert resultado.dropped_blocks == 30 - sum(
        1 for m in resultado.messages if m["role"] == "user"
    )
    assert resultado.clipped_tools == 0
    assert resultado.fits

    intacto = budget.fit(msgs[:5], SCHEMAS, 128_000)
    assert intacto.messages is not None and intacto.dropped_blocks == 0
    assert intacto.est_before == intacto.est_after


def _hueco_minimo(m: dict) -> dict:
    """Un hueco anclado en su suelo: su cabecera y la marca (si así pesa menos)."""
    lineas = m["content"].split("\n")
    n = len(lineas) - 1
    corto = {**m, "content": f"{lineas[0]}\n...[recortado, {n} de {n} líneas; "
                             "quedan las más nuevas]"}
    return corto if n and _bytes(corto) < _bytes(m) else m


def test_invariantes_con_historias_al_azar():
    """Las mismas garantías sobre cientos de historias y ventanas al azar."""
    import random

    rng = random.Random(20261114)
    for _ in range(300):
        msgs: list[dict] = [{"role": "system", "content": "S" * rng.randint(10, 3_000)}]
        if rng.random() < 0.5:
            msgs.append({"role": "system", "content": "[resumen]\n" + "r" * rng.randint(0, 500)})
        for t in range(rng.randint(0, 12)):
            msgs.append({"role": "user", "content": f"u{t} " + "ñ" * rng.randint(0, 200)})
            for _paso in range(rng.randint(0, 2)):
                n = rng.randint(1, 3)
                msgs.append({"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "t", "arguments": {"i": k}}} for k in range(n)
                ]})
                msgs += [{"role": "tool", "name": "t", "content": "x" * rng.randint(0, 9_000)}
                         for _k in range(n)]
            if rng.random() < 0.8:
                msgs.append({"role": "assistant", "content": "a" * rng.randint(0, 300)})
        tools = SCHEMAS if rng.random() < 0.5 else None
        max_tokens = rng.choice([512, 1_024, 2_048, 4_096, 32_000])
        tool_cap = rng.choice([None, 300, 1_200, 2_365])
        original = copy.deepcopy(msgs)

        res = budget.fit(msgs, tools, max_tokens, tool_cap=tool_cap)
        out = res.messages

        assert msgs == original
        assert res.est_after == budget.est_tokens(out, tools)
        assert res.fits == (res.est_after <= max_tokens - budget.REPLY_RESERVE)
        assert len(to_openai_messages(out)) == len(out)
        head = next((i for i, m in enumerate(msgs) if m["role"] != "system"), len(msgs))
        assert out[:1] == msgs[:1]  # la persona, siempre entera
        for i in range(1, head):
            # Un hueco anclado sale entero o, como último recurso, con su primera
            # línea, la marca y una cola de sus líneas (las más nuevas).
            if out[i] != msgs[i]:
                lineas, quedan = msgs[i]["content"].split("\n"), out[i]["content"].split("\n")
                assert quedan[0] == lineas[0] and quedan[1].startswith("...[recortado, ")
                assert quedan[2:] == lineas[len(lineas) - len(quedan) + 2:]
        assert res.clipped_pinned == sum(out[i] != msgs[i] for i in range(1, head))
        users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if users:
            # El bloque en curso entero, en su sitio (con sus tools recortados o no).
            cola = msgs[users[-1]:]
            assert [m["role"] for m in out[-len(cola):]] == [m["role"] for m in cola]
            assert out[-len(cola)] == cola[0]
        if res.dropped_blocks == 0 and res.clipped_tools == 0 and res.clipped_pinned == 0:
            assert out == msgs
        # Lo que se puede cumplir, se cumple: sin historia vieja y con los tools
        # del bloque en curso en el suelo, solo no cabe lo que no se puede tocar.
        if not res.fits:
            minimo = msgs[:1] + [_hueco_minimo(m) for m in msgs[1:head]] + [
                {**m, "content": truncate_for_context(m["content"], budget.MIN_TOOL_CHARS)}
                if m["role"] == "tool" else m
                for m in (msgs[users[-1]:] if users else msgs[head:])
            ]
            assert budget.est_tokens(minimo, tools) > max_tokens - budget.REPLY_RESERVE


# ---------------------------------------------------------------------------
# Los huecos anclados (hechos del patrón, resumen) también se pueden recortar,
# como ÚLTIMO recurso: con [memory].pin_facts de fábrica el bloque de hechos va
# en la cabeza, y con ~8 hechos o más el request local pasaba de num_ctx aun
# sin historia y con los tools en el suelo; Ollama cortaba el PRINCIPIO del
# prompt, la persona incluida. La persona (índice 0) no se toca nunca.
# ---------------------------------------------------------------------------

_GUSTOS = [
    "le gusta el café de olla sin azúcar",
    "su perro se llama Canelo y es un xoloitzcuintle",
    "trabaja de 9 a 6 en la oficina de Insurgentes",
    "prefiere que le hablen de tú y sin rodeos",
    "su hermana Mariana cumple años el 14 de marzo",
    "usa Fedora en la laptop y Hyprland como escritorio",
]


def _bloque_hechos(n: int) -> str:
    # El mismo formato que hooks._facts_block: cabecera, un hecho por línea (el
    # más NUEVO al final) y el corchete de cierre pegado al último.
    lineas = "\n".join(f"- (gustos) {_GUSTOS[k % len(_GUSTOS)]} ({k})" for k in range(n))
    return f"[contexto que ya sabes sobre el patrón:\n{lineas}]"


_OCHO = [
    {**SCHEMAS[0], "function": {**SCHEMAS[0]["function"], "name": f"tool_{i}",
                                "description": "Hace algo útil para el patrón. " * 3}}
    for i in range(8)
]


def test_con_muchos_hechos_anclados_se_quedan_los_mas_nuevos_y_la_persona_intacta():
    msgs = [{"role": "system", "content": PERSONA},
            {"role": "system", "content": _bloque_hechos(30)},
            {"role": "user", "content": "hola"}]
    original = copy.deepcopy(msgs)
    # Ni sin historia que tirar cabe: la persona, los esquemas y 30 hechos.
    assert budget.est_tokens(msgs, _OCHO) > 2_048 - budget.REPLY_RESERVE

    res = budget.fit(msgs, _OCHO, 2_048, tool_cap=budget.local_tool_cap(2_048))

    assert msgs == original  # puro
    assert res.fits and res.est_after == budget.est_tokens(res.messages, _OCHO)
    assert res.est_after <= 2_048 - budget.REPLY_RESERVE
    out = res.messages
    assert out[0] is msgs[0]  # la persona, el mismo objeto
    assert out[-1] == msgs[-1]  # el turno en curso, entero
    assert res.clipped_pinned == 1 and res.dropped_blocks == 0
    hechos = out[1]["content"].split("\n")
    # La cabecera dice qué es; luego la marca y las líneas MÁS NUEVAS, enteras.
    assert hechos[0] == "[contexto que ya sabes sobre el patrón:"
    quitadas = 30 - (len(hechos) - 2)
    assert hechos[1] == f"...[recortado, {quitadas} de 30 líneas; quedan las más nuevas]"
    assert 0 < quitadas < 30
    assert hechos[2:] == original[1]["content"].split("\n")[-(len(hechos) - 2):]
    assert hechos[-1].endswith("(29)]")
    # Las líneas que se quitan son las que hacía falta, no más: con una más no cabe.
    una_mas = {**out[1], "content": "\n".join(
        [hechos[0], f"...[recortado, {quitadas - 1} de 30 líneas; quedan las más nuevas]"]
        + original[1]["content"].split("\n")[-(len(hechos) - 1):])}
    assert budget.est_tokens([out[0], una_mas, out[2]], _OCHO) > 2_048 - budget.REPLY_RESERVE


def test_el_resumen_se_recorta_antes_que_los_hechos():
    resumen = "[resumen de la conversación previa]\n" + "\n".join(
        f"- hablaron del tema {k} con calma y detalle" for k in range(40))
    msgs = [{"role": "system", "content": PERSONA},
            {"role": "system", "content": _bloque_hechos(6)},
            {"role": "system", "content": resumen},
            {"role": "user", "content": "hola"}]
    res = budget.fit(msgs, _OCHO, 2_048)
    assert res.fits and res.clipped_pinned == 1
    assert res.messages[:2] == msgs[:2]  # persona y hechos, intactos
    assert res.messages[2]["content"].startswith(
        "[resumen de la conversación previa]\n...[recortado, ")
    assert res.messages[2]["content"].endswith("tema 39 con calma y detalle")


def test_lo_que_cabe_tirando_historia_no_toca_los_huecos_anclados():
    # El orden de siempre: primero la historia vieja y los tools; los huecos, al final.
    cabeza = [{"role": "system", "content": PERSONA},
              {"role": "system", "content": _bloque_hechos(4)}]
    msgs = _historia(10, 5_000, cabeza=cabeza)
    res = budget.fit(msgs, SCHEMAS, 2_048, tool_cap=budget.local_tool_cap(2_048))
    assert res.fits and res.dropped_blocks > 0 and res.clipped_pinned == 0
    assert res.messages[:2] == cabeza


def test_si_ni_la_cabecera_cabe_se_queda_la_cabecera_y_no_cabe():
    msgs = [{"role": "system", "content": "S" * 30_000},
            {"role": "system", "content": _bloque_hechos(30)},
            {"role": "user", "content": "hola"}]
    res = budget.fit(msgs, None, 2_048)
    assert not res.fits and res.clipped_pinned == 1
    assert res.messages[0] is msgs[0]
    assert res.messages[1]["content"] == (
        "[contexto que ya sabes sobre el patrón:\n"
        "...[recortado, 30 de 30 líneas; quedan las más nuevas]")
    assert res.messages[2] == msgs[2]


def test_un_hueco_de_una_sola_linea_no_se_toca():
    msgs = [{"role": "system", "content": "S" * 6_000},
            {"role": "system", "content": "[un hueco sin líneas que quitar] " * 40},
            {"role": "user", "content": "hola"}]
    res = budget.fit(msgs, None, 2_048)
    assert res.messages == msgs and res.clipped_pinned == 0 and not res.fits


def test_huecos_anclados_al_azar():
    """Persona, hechos y resumen de tamaños al azar: la persona nunca cambia, de
    cada hueco se quedan la cabecera y una cola, y si no cabe es que ni con los
    huecos en su suelo cabía."""
    import random

    rng = random.Random(1114)
    for _ in range(300):
        msgs: list[dict] = [{"role": "system", "content": "S" * rng.randint(10, 6_000)}]
        if rng.random() < 0.8:
            msgs.append({"role": "system", "content": _bloque_hechos(rng.randint(1, 40))})
        if rng.random() < 0.5:
            msgs.append({"role": "system", "content": "[resumen de la conversación previa]\n"
                         + "\n".join("r" * rng.randint(0, 120) for _ in range(rng.randint(0, 30)))})
        msgs += _turno(0, "z" * rng.randint(0, 9_000), cerrado=False)
        tools = _OCHO[:rng.randint(0, 8)] or None
        max_tokens = rng.choice([1_024, 2_048, 4_096])
        original = copy.deepcopy(msgs)

        res = budget.fit(msgs, tools, max_tokens, tool_cap=budget.local_tool_cap(max_tokens))
        out = res.messages
        head = sum(1 for m in msgs if m["role"] == "system")

        assert msgs == original
        assert out[0] is msgs[0] and len(out) == len(msgs)
        assert res.est_after == budget.est_tokens(out, tools)
        assert res.fits == (res.est_after <= max_tokens - budget.REPLY_RESERVE)
        for i in range(1, head):
            if out[i] != msgs[i]:
                lineas, quedan = msgs[i]["content"].split("\n"), out[i]["content"].split("\n")
                assert quedan[0] == lineas[0] and quedan[1].startswith("...[recortado, ")
                assert quedan[2:] == lineas[len(lineas) - len(quedan) + 2:]
        if not res.fits:
            minimo = msgs[:1] + [_hueco_minimo(m) for m in msgs[1:head]] + [
                {**m, "content": truncate_for_context(m["content"], budget.MIN_TOOL_CHARS)}
                if m["role"] == "tool" else m for m in msgs[head:]]
            assert budget.est_tokens(minimo, tools) > max_tokens - budget.REPLY_RESERVE
