import subprocess
import tempfile
from pathlib import Path

VOICE_MODEL = "/home/exitili/Documentos/chapi_assistant/voices/es_MX-ald-medium.onnx"


def speak(text: str) -> None:
    text = text.strip()

    if not text:
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio_path = Path(temp_audio.name)

    try:
        subprocess.run(
            [
                "python",
                "-m",
                "piper",
                "-m",
                VOICE_MODEL,
                "-f",
                str(audio_path),
                "--",
                text,
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        subprocess.run(
            [
                "ffplay",
                "-autoexit",
                "-nodisp",
                "-loglevel",
                "quiet",
                str(audio_path),
            ],
            check=False,
        )

    except subprocess.CalledProcessError as error:
        print("Error usando Piper:")
        print(error.stderr)

    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
