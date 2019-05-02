from django.conf.urls import url
from django.urls import path
from . import views

app_name = 'pocheck'
urlpatterns = [
    path('', views.index, name='index'),
    path('play/', views.play, name='play'),
]