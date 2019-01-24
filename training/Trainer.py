from utils import *
import itertools
import numpy as np
import sys
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import GeneratorResNet, Discriminator, LargePatchDiscriminator, weights_init_normal


class Trainer():
    def __init__(self, opt):
        self.config = opt

        # Make output dirs
        os.makedirs('saved_models/%s' % (opt.model_name), exist_ok=True)
        os.makedirs('images/%s' % (opt.model_name), exist_ok=True)

        self.cuda = opt.gpu_id > -1

        # Gs and Ds
        self.G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
        self.G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)
        if opt.large_patch:
            self.D_A = LargePatchDiscriminator()
            self.D_B = LargePatchDiscriminator()
        else:
            self.D_A = Discriminator()
            self.D_B = Discriminator()

        # Patch
        if opt.large_patch:
            self.patch = (1, 64, 64)
        else:
            self.patch = (1, 16, 16)

        # Weight init
        self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)

        # Loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        if self.cuda:
            self.G_AB = self.G_AB.cuda()
            self.G_BA = self.G_BA.cuda()
            self.D_A = self.D_A.cuda()
            self.D_B = self.D_B.cuda()
            self.criterion_GAN = self.criterion_GAN.cuda()
            self.criterion_cycle = self.criterion_cycle.cuda()
            self.criterion_identity = self.criterion_identity.cuda()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
                                            lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,
                                                                lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                   opt.decay_epoch).step)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D,
                                                                lr_lambda=LambdaLR(opt.n_epochs, 0,
                                                                                   opt.decay_epoch).step)
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        # Loss weights
        self.lambda_cyc = 10
        self.lambda_id = opt.lambda_id * self.lambda_cyc

        # Buffers of previously generated samples
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Image transformations
        A_transforms_ = [transforms.CenterCrop((178, 178)),
                         transforms.Resize((300, 300)),
                         transforms.RandomCrop((256, 256)),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomAffine(degrees=opt.rotate_degree, fillcolor=(255, 255, 255)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        B_transforms_ = [transforms.Resize((360, 360)),
                         transforms.RandomCrop((256, 256)),
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomAffine(degrees=opt.rotate_degree, fillcolor=(255, 255, 255)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # Training data loader
        self.train_dataloader = DataLoader(
            ImageDataset("./data/", A_transforms_=A_transforms_, B_transforms_=B_transforms_),
            batch_size=1, shuffle=True, )
        # Test data loader
        self.val_dataloader = DataLoader(
            ImageDataset("./data/", A_transforms_=A_transforms_, B_transforms_=B_transforms_, mode='test'),
            batch_size=1)

    def train_epoch(self, epoch):
        prev_time = time.time()
        for i, batch in enumerate(self.train_dataloader):

            # Model input
            real_A = Variable(batch['A'].type(self.Tensor))
            real_B = Variable(batch['B'].type(self.Tensor))

            # Adversarial ground truths

            valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False)
            fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.patch))), requires_grad=False)

            #  Train Generators

            self.optimizer_G.zero_grad()

            # GAN loss
            fake_B = self.G_AB(real_A)
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            fake_A = self.G_BA(real_B)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            recov_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Identity loss

            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # Total loss
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
            loss_G.backward()
            self.optimizer_G.step()

            #  Train Discriminator

            self.optimizer_D.zero_grad()

            # Real loss
            loss_real = self.criterion_GAN(self.D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            self.optimizer_D.step()

            self.optimizer_D.zero_grad()
            loss_real = self.criterion_GAN(self.D_B(real_B), valid)
            fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
            loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            self.optimizer_D.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # Determine approximate time left
            batches_done = epoch * len(self.train_dataloader) + i
            batches_left = self.config.n_epochs * len(self.train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                (epoch, self.config.n_epochs,
                 i, len(self.train_dataloader),
                 loss_D.item(), loss_G.item(),
                 loss_GAN.item(), loss_cycle.item(),
                 loss_identity.item(), time_left))

            if batches_done % self.config.sample_interval == 0:
                # Sample a picture
                imgs = next(iter(self.val_dataloader))
                real_A = Variable(imgs['A'].type(self.Tensor))
                fake_B = self.G_AB(real_A)
                real_B = Variable(imgs['B'].type(self.Tensor))
                fake_A = self.G_BA(real_B)
                img_sample = torch.cat((real_A.data, fake_B.data,
                                        real_B.data, fake_A.data), 0)
                save_image(img_sample, 'images/%s/%s.png' % (self.config.model_name, batches_done), nrow=4, normalize=True)

        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()

        if self.config.checkpoint_interval != -1 and epoch % self.config.checkpoint_interval == 0:
            torch.save(self.G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (self.config.model_name, epoch))
