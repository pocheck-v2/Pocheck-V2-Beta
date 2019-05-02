from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
# Create your views here.

def index(request):

    template = loader.get_template('pocheck/index.html')
    context = {
        'latest_question_list' : "test",
    }
    return HttpResponse(template.render(context, request))

def play(request):

    return render(request, 'pocheck/play.html')