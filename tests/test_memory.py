from crotolamo.core.memory import Conversation


def test_system_always_first():
    conv = Conversation("SYS", max_turns=10)
    conv.add_user("hola")
    conv.add_assistant("qué onda, patrón")
    msgs = conv.to_messages()
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_context_is_remembered():
    conv = Conversation("SYS")
    conv.add_user("recuerda el número 42")
    conv.add_assistant("anotado, patrón")
    conv.add_user("¿qué número te dije?")
    contents = [m["content"] for m in conv.to_messages()]
    assert any("42" in c for c in contents)


def test_sliding_window_drops_oldest():
    conv = Conversation("SYS", max_turns=2)
    for i in range(5):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    users = [m for m in conv.to_messages() if m["role"] == "user"]
    assert len(users) == 2
    # Quedan los más recientes.
    assert users[0]["content"] == "u3"
    assert users[1]["content"] == "u4"


def test_window_keeps_blocks_aligned():
    # Tras recortar no deben quedar mensajes 'tool' o 'assistant' huérfanos al frente.
    conv = Conversation("SYS", max_turns=1)
    conv.add_user("u0")
    conv.add_assistant("", tool_calls=[{"name": "x", "arguments": {}}])
    conv.add_tool_result("x", "ok")
    conv.add_assistant("listo")
    conv.add_user("u1")
    conv.add_assistant("a1")
    first_non_system = conv.to_messages()[1]
    assert first_non_system["role"] == "user"


def test_reset_clears_history_but_keeps_system():
    conv = Conversation("SYS")
    conv.add_user("algo")
    conv.reset()
    assert conv.to_messages() == [{"role": "system", "content": "SYS"}]


def test_compaction_summarizes_instead_of_dropping():
    calls = {"n": 0}

    def fake_summarizer(text):
        calls["n"] += 1
        return "RESUMEN"

    conv = Conversation("SYS", max_turns=2, compaction=True, summarizer=fake_summarizer)
    for i in range(6):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")

    msgs = conv.to_messages()
    assert msgs[0]["content"] == "SYS"
    # Hay un mensaje de resumen anclado (no se perdió el contexto viejo en seco).
    assert any("resumen" in m["content"].lower() for m in msgs)
    assert calls["n"] >= 1  # se llamó al summarizer
    # El historial sigue acotado (no crece sin límite).
    users = [m for m in msgs if m["role"] == "user"]
    assert len(users) <= 2


def test_compaction_keeps_single_growing_summary():
    conv = Conversation("SYS", max_turns=1, compaction=True, summarizer=lambda t: "S")
    for i in range(5):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    # Solo UN mensaje de resumen, por mucho que se compacte.
    summaries = [m for m in conv.to_messages() if "resumen" in m["content"].lower()]
    assert len(summaries) == 1


def test_no_compaction_drops_as_before():
    conv = Conversation("SYS", max_turns=2, compaction=False)
    for i in range(5):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    assert not any("resumen" in m["content"].lower() for m in conv.to_messages())


# --- T7: Conversation debe aguantar hilos (BrainThread + cualquier lector) ---

def test_conversation_soporta_escritores_concurrentes():
    """Dos hilos añadiendo y uno leyendo a la vez: nada de 'list changed size
    during iteration' y el conteo final consistente. El objetivo es 'no
    corrompe', no un orden determinista."""
    import threading

    from crotolamo.core.memory import Conversation

    conv = Conversation("SYS", max_turns=10_000)
    n = 200
    barrera = threading.Barrier(3)
    errores: list[Exception] = []

    def escritor(prefijo: str) -> None:
        try:
            barrera.wait(timeout=10)
            for i in range(n):
                conv.add_user(f"{prefijo}-{i}")
                conv.add_assistant(f"eco {prefijo}-{i}")
        except Exception as error:  # noqa: BLE001
            errores.append(error)

    def lector() -> None:
        try:
            barrera.wait(timeout=10)
            for _ in range(n):
                conv.to_messages()
                conv.history
        except Exception as error:  # noqa: BLE001
            errores.append(error)

    threads = [
        threading.Thread(target=escritor, args=("a",)),
        threading.Thread(target=escritor, args=("b",)),
        threading.Thread(target=lector),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)

    assert errores == []
    assert len(conv.history) == 2 * n * 2  # 2 hilos x n turnos x (user+assistant)
