from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def chat_view(request):
    return render(request, 'chat.html')