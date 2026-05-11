import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


SYSTEM_PROMPT = """
Sos un asistente especializado en acompañamiento emocional y contención psicológica en español. Tu objetivo es ayudar a la persona a sentirse escuchada, comprendida y acompañada con respuestas cálidas, humanas y claras. No realizás diagnósticos, no reemplazás terapia psicológica, psiquiátrica ni atención médica profesional, y nunca afirmás tener emociones reales.
Respondé siempre de forma breve, natural y empática, usando un máximo de 3 oraciones completas.
La primera oración debe validar emocionalmente a la persona sin exagerar ni dramatizar.
La segunda oración debe ofrecer una sugerencia práctica, simple y realista que pueda ayudar en el momento.
La tercera oración debe incluir una pregunta corta y abierta para continuar la conversación.
Mantené un tono humano, tranquilo, cercano y profesional. Evitá respuestas robóticas, genéricas o demasiado técnicas. No uses listas, viñetas ni respuestas largas. No dejes frases abiertas o incompletas.
Si la persona menciona autolesiones, suicidio, violencia o riesgo inmediato, respondé con contención emocional, recomendá buscar ayuda profesional o contactar emergencias/locales de asistencia, y evitá cualquier consejo peligroso.
Nunca juzgues, minimices ni discutas las emociones de la persona. Priorizá la escucha, la claridad y la seguridad emocional.
"""


FALLBACK_RESPONSE = (
    "Entiendo que eso puede sentirse muy pesado. "
    "Podés empezar por hacer una pausa corta y elegir una sola acción pequeña para ahora. "
    "¿Qué es lo que más te está costando en este momento?"
)


def _looks_incomplete(text):
    cleaned = text.strip()

    if not cleaned:
        return True

    return cleaned[-1] not in ".!?¿¡"


def _normalize_response(text):
    cleaned = text.strip()

    if len(cleaned) < 20 or _looks_incomplete(cleaned):
        return FALLBACK_RESPONSE

    return cleaned


def ask_openai(message):
    """Envía un mensaje a OpenAI y devuelve una respuesta segura para mostrar en pantalla."""

    if not message or not message.strip():
        return "Escribí un mensaje para comenzar."

    if not OPENAI_API_KEY:
        return "Falta configurar OPENAI_API_KEY en el archivo .env."

    try:
        start_time = time.perf_counter()

        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=message.strip(),
            temperature=0.2,
            max_output_tokens=160,
        )

        elapsed = time.perf_counter() - start_time
        print(f"Tiempo OpenAI: {elapsed:.2f} segundos | Modelo: {OPENAI_MODEL}")

        response_text = getattr(response, "output_text", "") or ""

        if not response_text:
            return "No pude generar una respuesta en este momento."

        return _normalize_response(response_text)

    except Exception as error:
        error_text = str(error)
        print(f"Error OpenAI: {error_text}")

        if "429" in error_text or "quota" in error_text.lower():
            return (
                "OpenAI indicó que no hay cuota disponible para este modelo o API key. "
                "Probá otro modelo en .env o revisá la cuota del proyecto en OpenAI."
            )

        if "404" in error_text or "not found" in error_text.lower():
            return (
                "El modelo configurado no está disponible para esta API key. "
                "Probá cambiando OPENAI_MODEL en .env."
            )

        if "api key" in error_text.lower() or "authentication" in error_text.lower():
            return (
                "La API key de OpenAI falta o no es válida. "
                "Verificá OPENAI_API_KEY en el archivo .env."
            )

        return "Ocurrió un error al consultar OpenAI. Verificá tu API key, el modelo configurado y la conexión a internet."