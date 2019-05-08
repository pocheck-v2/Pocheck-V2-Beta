from django.db import models

# Create your models here.
class F_name(models.Model):
    First_Name = models.CharField(max_length = 10)
    def __str__(self):
        return self.First_Name

class S_name(models.Model):
    first_name = models.ForeignKey(F_name, on_delete=models.CASCADE)
    second_name = models.CharField(max_length = 200)

    def __str__(self):
        return self.second_name