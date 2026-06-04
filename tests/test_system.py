from crotolamo.tools import system


def test_disk_usage():
    out = system.disk_usage()
    assert "Disco" in out and "%" in out


def test_ram_usage():
    out = system.ram_usage()
    assert "RAM" in out


def test_system_status_combines():
    out = system.system_status()
    assert "Disco" in out and "RAM" in out


def test_list_processes_has_header():
    out = system.list_processes(limit=5)
    # El header de ps incluye la columna PID.
    assert "PID" in out.upper()
