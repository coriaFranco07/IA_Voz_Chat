import requests


OLLAMA_URL = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = """
Sos un asistente emocional empático.
No diagnosticás enfermedades.
No reemplazás terapia profesional.
Ayudás mediante escucha activa y CBT.
Sos cálido y humano.
"""


def ask_llama(message):

    payload = {
        "model": "phi3:mini",

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": message
            }
        ],

        "stream": False,

        "options": {
            "num_predict": 120,
            "temperature": 0.7
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)

    data = response.json()

    return data["message"]["content"]