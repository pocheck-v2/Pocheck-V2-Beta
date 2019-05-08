from django.db import models

# Create your models here.
class PLAY_OR_NOT(models.Field):
    flag = models.BooleanField(default=False)