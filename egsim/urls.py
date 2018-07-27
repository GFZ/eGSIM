"""eGSIM URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url  # added by default by django
from django.contrib import admin  # added by default by django
# from django.views.generic.base import RedirectView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from . import views


# for infor with trailing slashes:
# https://stackoverflow.com/questions/1596552/django-urls-without-a-trailing-slash-do-not-redirect

urlpatterns = [
    url(r'^admin/', admin.site.urls),  # added by default by django
    url(r'^$', views.index, name='index'),  # same as /home (see views.py)
    url(r'^(?P<menu>[a-zA-Z]+)/?$', views.main, name='main'),  # main page entry point
    url(r'^service/home/?$', views.home, name='home'),
    url(r'^service/trsel/?$', views.trsel, name='trsel'),
    url(r'^service/trellis/?$', views.trellis, name='trellis'),
    url(r'^service/gmdb/?$', views.gmdb, name='gmdb'),
    url(r'^service/residuals/?$', views.residuals, name='residuals'),
    url(r'get_init_params', views.get_init_params),
    url(r'get_tr_projects', views.get_tr_projects),
    url(r'get_gmdbs', views.get_gmdbs),
    # url(r'get_trellis_plots', views.TrellisPlotsView.as_view()),
    url(r'test_err', views.test_err),

    # REST (or alike) views:
    url(r'^query/trellis/?$', views.TrellisPlotsView.as_view(), name='trellis_api'),
    url(r'^query/gsims/?$', views.TrSelectionView.as_view(), name='trsel_api'),
    url(r'^query/magdistdata/?$', views.GmdbSelectionView.as_view(), name='magdistdata_api'),
    url(r'^query/residuals/?$', views.ResidualsView.as_view(), name='residuals_api'),

    # test views, TEMPORARY:
    url(r'^trellis_test/?$', views.trellis_test, name='main'),
]
