import time

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3:mini"


SYSTEM_PROMPT = """
Sos un asistente conversacional en español.
Respondé de forma breve, clara y amable.
"""


def ask_llama(message):
    """Envía un mensaje a Ollama y devuelve una respuesta segura para mostrar en pantalla."""

    if not message or not message.strip():
        return "Escribí un mensaje para comenzar."

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message.strip(),
            },
        ],
        "stream": False,
        "options": {
            "num_predict": 60,
            "temperature": 0.5,
        },
    }

    try:
        start_time = time.perf_counter()

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60,
        )

        elapsed = time.perf_counter() - start_time
        print(f"Tiempo Ollama: {elapsed:.2f} segundos")

        response.raise_for_status()

        data = response.json()
        return data.get("message", {}).get(
            "content",
            "No pude generar una respuesta en este momento.",
        )

    except requests.exceptions.ConnectionError:
        return "No pude conectarme con Ollama. Verificá que esté abierto ejecutando: ollama run phi3:mini"

    except requests.exceptions.Timeout:
        return "La IA tardó demasiado en responder. Probá con un mensaje más corto."

    except requests.exceptions.RequestException:
        return "Ollama respondió con un error. Verificá que el modelo phi3:mini esté funcionando."

    except Exception:
        return "Ocurrió un error inesperado al consultar la IA."
