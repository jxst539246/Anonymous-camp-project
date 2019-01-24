from Trainer import Trainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--root', type=str, default='./data/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
    parser.add_argument('--sample_interval', type=int, default=100,
                        help='interval between sampling images from generators')
    parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id')
    parser.add_argument('--model_name', type=str, default='cycleGAN_white', help='model name')
    parser.add_argument('--rotate_degree', type=int, default=0, help='rotate degree')
    parser.add_argument('--lambda_id', type=float, default=0.5, help='lambda_id')
    parser.add_argument('--large_patch', type=bool, default=False, help='whether use large patch')
    opt = parser.parse_args()

    trainer = Trainer(opt)
    for epoch in range(opt.n_epochs):
        trainer.train_epoch(epoch)
