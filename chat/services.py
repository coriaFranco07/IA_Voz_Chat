import os
import time

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


SYSTEM_PROMPT = """
Sos un asistente de acompañamiento emocional en español.
No diagnosticás ni reemplazás atención profesional.
Respondé con empatía, breve y completo.
No hagas listas largas.
Terminá siempre con una pregunta simple para acompañar.
"""


def ask_gemini(message):
    """Envía un mensaje a Google Gemini y devuelve una respuesta segura para mostrar en pantalla."""

    if not message or not message.strip():
        return "Escribí un mensaje para comenzar."

    if not GEMINI_API_KEY:
        return "Falta configurar GEMINI_API_KEY en el archivo .env."

    try:
        start_time = time.perf_counter()

        genai.configure(api_key=GEMINI_API_KEY)

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

        response = model.generate_content(
            message.strip(),
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 180,
            },
        )

        elapsed = time.perf_counter() - start_time
        print(f"Tiempo Gemini: {elapsed:.2f} segundos | Modelo: {GEMINI_MODEL}")

        if not response.text:
            return "No pude generar una respuesta en este momento."

        return response.text.strip()

    except Exception as error:
        print(f"Error Gemini: {error}")
        return "Ocurrió un error al consultar Google Gemini. Probá cambiando GEMINI_MODEL en .env a gemini-2.0-flash o gemini-1.5-flash."
