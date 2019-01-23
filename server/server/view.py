from django.http import JsonResponse, HttpResponse
from .sampler import transform_sampler


def hello(request):
    return HttpResponse("Hello world ! ")


def trans(request):
    timestamp = request.POST['timestamp']
    image = request.POST['image']
    file_name = transform_sampler.transform(image)
    return JsonResponse({'timestamp': timestamp, 'file_name': file_name})
