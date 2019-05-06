from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.template import loader
from django.http import StreamingHttpResponse
import cv2
import numpy as np
import threading

# Create your views here.

def index(request):

    template = loader.get_template('pocheck/index.html')
    context = {
        'latest_question_list' : "test",
    }
    return HttpResponse(template.render(context, request))

def play(request):

    livefe(request)
    return render(request, 'pocheck/play_pocheck.html')

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
            cv2.imshow("", self.frame)
            if cv2.waitKey(1) == 'q':
                break


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def livefe(request):
    try:
        # resp = StreamingHttpResponse(gen(VideoCamera()), content_type = "text/html")
        return render(request, 'pocheck/play_pocheck.html', {'video' : gen(VideoCamera())})
    except:
        pass