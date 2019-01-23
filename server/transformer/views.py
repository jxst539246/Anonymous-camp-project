from django.http import JsonResponse, HttpResponse
from .sampler import transform_sampler
import json

def hello(request):
    return HttpResponse("Hello world ! ")


def trans(request):
    data = json.loads(request.body)
    timestamp = data['timestamp']
    image = data['image']
    file_name = transform_sampler.transform(image)
    return JsonResponse({'timestamp': timestamp, 'file_name': file_name})
