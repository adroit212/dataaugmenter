
from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.home),
    re_path(r'^home', views.home),
    re_path(r'^data_history', views.data_history),
    re_path(r'^data_operation', views.data_operation),
]
