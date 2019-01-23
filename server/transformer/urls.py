from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.hello),
    url(r'^trans/$', views.trans)
]