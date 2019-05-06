from django.conf.urls import url
from django.urls import path
from . import views

app_name = 'pocheck'
urlpatterns = [
    path('', views.index, name='index'),
    path('play/', views.play, name='play'),
    path('livefe/', views.livefe, name='livefe'),
    # path('play/', views.play_pocheck, name='play_pocheck'),
]