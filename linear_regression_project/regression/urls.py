from django.urls import path
from .views import linear_regression_view

urlpatterns = [
    path('linear_regression/', linear_regression_view, name='linear_regression')
]
