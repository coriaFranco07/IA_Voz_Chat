from django.shortcuts import render
from .services import ask_llama


def ai_chat(request):

    response_text = None

    if request.method == "POST":

        message = request.POST.get("message")

        response_text = ask_llama(message)

    return render(request, "ai_chat.html", {
        "response": response_text
    })