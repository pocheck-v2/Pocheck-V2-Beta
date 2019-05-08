from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
from django.http import StreamingHttpResponse
import cv2
import numpy as np
import threading
import sys
from .models import PLAY_OR_NOT
# sys.path.append('/home/pirl/Desktop/Pocheck-V2-Beta')

# import src
from Pocheck import RT_FMD_svmtree_hl

# Create your views here.

def index(request):
    # main()
    template = loader.get_template('pocheck/index.html')
    context = {
        'latest_question_list' : "test",
    }
    return HttpResponse(template.render(context, request))

def play(request):
    RT_FMD_svmtree_hl.main()
    template = loader.get_template('pocheck/index.html')
    context = {
        'latest_question_list': "test",
    }
    return HttpResponse(template.render(context, request))

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
            # cv2.imshow("", self.frame)
            # if cv2.waitKey(1) & 0xFF == 'q':
            #     break


def gen(camera):
    i = 0
    while True: #i <= 60:
        frame = camera.get_frame()
        print(type(frame))
        # i += 1
        '''
        yield(b'<!DOCTYPE html><html>')
        yield(b'<head><meta charset="utf-8"><title>POCHECK</title>')
        yield(b'</head>')
        yield(b'<body>')
        '''
        # yield(b'<img id="profileImage" src="data:image/jpg;base64,"' + frame + b'>')
        yield(b'<img id="ItemPreview" src="" />')
        # yield(camera.frame)
        yield(b'<script>')
        yield(b'document.getElementById("ItemPreview").src = "data:image/jpg;base64,"' + frame)
        yield(b'</script>')
        '''
        yield(b'hello')
        yield(b'<img id = "ItemPreview" src = "/static/pocheck/img/facial-recognition.png>"')
        yield(b'<body>')
        yield(b'</html>')
        '''


        '''
        yield(b'<script>')
        yield(b'document.getElementById("ItemPreview").src = "data:image/png;base64," + YourByte')
        yield(b'</script>')
        '''

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def livefe(request):
    if request.method == "POST":
        return render(request, "pocheck/index.html")
    else:

        return StreamingHttpResponse(gen(VideoCamera()), content_type='text/html; charset=utf-8')


