import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


SYSTEM_PROMPT = """
Sos un asistente de acompañamiento emocional en español.
No diagnosticás ni reemplazás atención profesional.
Respondé con empatía, breve y completo.
No hagas listas largas.
Dá una respuesta cerrada de 4 a 7 oraciones.
Terminá siempre con una pregunta simple para acompañar.
"""


def _extract_text(response):
    """Extrae texto de Gemini de forma tolerante."""
    try:
        if response.text:
            return response.text.strip()
    except Exception:
        pass

    try:
        parts = response.candidates[0].content.parts
        text = "".join(part.text for part in parts if getattr(part, "text", None))
        return text.strip()
    except Exception:
        return ""


def _is_too_short(text):
    return len(text.strip()) < 80


def ask_gemini(message):
    """Envía un mensaje a Google Gemini y devuelve una respuesta segura para mostrar en pantalla."""

    if not message or not message.strip():
        return "Escribí un mensaje para comenzar."

    if not GEMINI_API_KEY:
        return "Falta configurar GEMINI_API_KEY en el archivo .env."

    try:
        start_time = time.perf_counter()

        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=message.strip(),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=320,
            ),
        )

        elapsed = time.perf_counter() - start_time
        print(f"Tiempo Gemini: {elapsed:.2f} segundos | Modelo: {GEMINI_MODEL}")

        response_text = _extract_text(response)

        if not response_text:
            return "No pude generar una respuesta en este momento."

        if _is_too_short(response_text):
            return (
                response_text
                + " Entiendo que esto puede sentirse pesado. No tenés que resolver todo ahora; podemos empezar por poner en palabras qué es lo que más te está afectando. ¿Querés contarme qué pasó hoy?"
            )

        return response_text

    except Exception as error:
        error_text = str(error)
        print(f"Error Gemini: {error_text}")

        if "429" in error_text or "quota" in error_text.lower():
            return "Google Gemini indicó que no hay cuota disponible para este modelo o API key. Probá otro modelo en .env, por ejemplo GEMINI_MODEL=gemini-2.5-flash-lite, o revisá la cuota del proyecto en Google AI Studio."

        if "404" in error_text or "not found" in error_text.lower():
            return "El modelo configurado no está disponible para esta API key. Probá cambiando GEMINI_MODEL en .env a gemini-2.5-flash o gemini-2.5-flash-lite."

        return "Ocurrió un error al consultar Google Gemini. Verificá tu API key, el modelo configurado y la conexión a internet."
