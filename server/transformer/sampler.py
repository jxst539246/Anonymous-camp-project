import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from .models import GeneratorResNet
import re
import base64
from io import BytesIO
from torch.autograd import Variable
import uuid


def base64_to_image(base64_str):
    # read image from a base64 str
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data).convert('RGB')
    return img


class Sampler():
    def __init__(self, use_cuda=True):
        # init
        self.G = GeneratorResNet()
        if use_cuda:
            self.G = self.G.cuda()
        transforms_ = [transforms.Resize((256, 256)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transformer = transforms.Compose(transforms_)
        self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor

    def load_check_point(self, check_point_path):
        # load data
        self.G.load_state_dict(torch.load(check_point_path))
        self.G.eval()
        for param in self.G.parameters():
            param.requires_grad = False

    def transform(self, image_str):
        # transform
        self.load_check_point('/home/menruimr/Anonymous-camp-project/server/server/final_models/2.0_G_AB_3.pth')
        img = base64_to_image(image_str)
        img = self.transformer(img)
        real_A = Variable(img.reshape(1, 3, 256, 256).type(self.Tensor))
        img_sample = self.G(real_A)
        file_name = 'output/' + str(uuid.uuid4()) + '.png'
        save_image(img_sample, file_name, normalize=True)
        return file_name


transform_sampler = Sampler(use_cuda=False)
