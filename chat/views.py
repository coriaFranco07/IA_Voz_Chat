from email.mime import message

from django.shortcuts import render
from .services import ask_openai


def ai_chat(request):
    response_text = None
    user_message = ""

    if request.method == "POST":
        user_message = request.POST.get("message", "").strip()

        if user_message:
            response_text = ask_openai(user_message)
        else:
            response_text = "Escribí algo para que pueda acompañarte."

    return render(
        request,
        "ai_chat.html",
        {
            "response": response_text,
            "message": user_message,
        },
    )
