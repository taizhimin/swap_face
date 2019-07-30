import base64
import time

from django.shortcuts import render
from django.shortcuts import HttpResponse
import json
from . import FaceChanger


def index(request):
    ctx = {}
    if request.method == "GET":
        return render(request, 'index.html', ctx)
    if request.method == "POST":
        image1 = request.FILES.get("file1")
        timestamp1 = str(int(time.time()))
        file_url1 = 'static/img/1-%s.%s' % (timestamp1, 'jpg')
        destination1 = open(file_url1, 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk1 in image1.chunks():  # 分块写入文件
            destination1.write(chunk1)
        destination1.close()

        image2 = request.FILES.get("file2")
        timestamp2 = str(int(time.time()))
        file_url2 = 'static/img/2-%s.%s' % (timestamp2, 'jpg')
        destination2 = open(file_url2, 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk2 in image2.chunks():  # 分块写入文件
            destination2.write(chunk2)
        destination2.close()
        return HttpResponse(json.dumps(FaceChanger.swap_face(file_url1, file_url2)), content_type='application/json')
