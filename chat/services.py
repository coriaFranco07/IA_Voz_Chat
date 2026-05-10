import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3:mini"


SYSTEM_PROMPT = """
Sos un asistente emocional cálido y empático.
No diagnosticás enfermedades.
No reemplazás terapia profesional.
No recomendás medicación.
Usás escucha activa, preguntas simples y técnicas CBT básicas.
Respondé siempre en español, de forma breve, clara y humana.
"""


def ask_llama(message):
    """Envía un mensaje a Ollama y devuelve una respuesta segura para mostrar en pantalla."""

    if not message or not message.strip():
        return "Contame un poco cómo te estás sintiendo."

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
            "num_predict": 90,
            "temperature": 0.6,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60,
        )
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
