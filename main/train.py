import os
import sys
import argparse
from datetime import datetime

import torch
from tensorboardX import SummaryWriter

from utils.utils import train
from utils.logger import Logger

from models.gan import GAN
from losses.bce import BCELoss

from utils.dataloader import build_dataloader


def main():
    parser = argparse.ArgumentParser(description="Generate 汉字 via generative adversarial network.")

    # Dataset
    parser.add_argument("--size", type=int, default=32, help="Font size.")
    parser.add_argument("--from_unicode", type=int, help="Starting point of the unicode.")
    parser.add_argument("--to_unicode", type=int, help="Ending point of the unicode.")
    parser.add_argument("--font", type=str, required=True, help="Path to the font file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    # Optimization
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--gpu_ids", type=str, default='', help="GPUs for running this script.")
    parser.add_argument("--rand_dim", type=int, default=128, help="Dimension of the random vector.")
    parser.add_argument("--num_fakes", type=int, default=16,
                        help="Use num_fakes generated images to train the discriminator.")
    parser.add_argument("--flip_rate", type=float, default=0.8, help="Label flipping rate.")
    parser.add_argument("--g_lr", type=float, default=0.01, help="Learning rate for generator.")
    parser.add_argument("--d_lr", type=float, default=0.01, help="Learning rate for discriminator.")
    parser.add_argument("--factor", type=float, default=0.2, help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Threshold for measuring the new optimum, to only focus on significant changes. ")
    # Misc
    parser.add_argument("--log_dir", type=str, default="../run/", help="Where to save the log?")
    parser.add_argument("--log_name", type=str, required=True, help="Name of the log folder.")
    parser.add_argument("--show_freq", type=int, default=64, help="How frequently to show generated images?")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()
    assert args.show_freq > 0
    assert 0.0 <= args.flip_rate <= 1.0

    # Check before run.
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_dir = os.path.join(args.log_dir, args.log_name)

    # Setting up logger
    log_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
    sys.stdout = Logger(os.path.join(log_dir, log_file))
    print(args)

    for s in args.gpu_ids:
        try:
            int(s)
        except ValueError as e:
            print("Invalid gpu id:{}".format(s))
            raise ValueError

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    if args.gpu_ids:
        if torch.cuda.is_available():
            use_gpu = True
            torch.cuda.manual_seed_all(args.seed)
        else:
            use_gpu = False
    else:
        use_gpu = False

    torch.manual_seed(args.seed)

    dataloader, size = build_dataloader(args.batch_size, args.num_workers, use_gpu, args.font, args.size,
                                        args.from_unicode, args.to_unicode)
    model = GAN(args.num_fakes, args.rand_dim, size, use_gpu)
    criterion = BCELoss()
    d_optimizer = torch.optim.SGD(model.discriminator.parameters(), lr=args.d_lr, momentum=0.9)
    g_optimizer = torch.optim.SGD(model.generator.parameters(), lr=args.g_lr, momentum=0.9)
    d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode="min", factor=args.factor,
                                                             patience=args.patience, verbose=True,
                                                             threshold=args.threshold)
    g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode="min", factor=args.factor,
                                                             patience=args.patience, verbose=True,
                                                             threshold=args.threshold)

    optimizer = d_optimizer, g_optimizer
    scheduler = d_scheduler, g_scheduler

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    print("Start training...")
    start = datetime.now()
    with SummaryWriter(log_dir) as writer:
        for epoch in range(args.epochs):
            for i, param_group in enumerate(d_optimizer.param_groups):
                d_learning_rate = float(param_group["lr"])
                writer.add_scalar("d_lr_group_{0}".format(i), d_learning_rate, global_step=epoch)
            for i, param_group in enumerate(g_optimizer.param_groups):
                g_learning_rate = float(param_group["lr"])
                writer.add_scalar("g_lr_group_{0}".format(i), g_learning_rate, global_step=epoch)
            train(model, dataloader, criterion, optimizer, use_gpu, writer, epoch, scheduler, args.num_fakes,
                  args.flip_rate, args.show_freq)

    torch.save(model, os.path.join(log_dir, "latest.pth"))

    elapsed_time = str(datetime.now() - start)
    print("Finish training. Total elapsed time %s." % elapsed_time)


if __name__ == "__main__":
    main()
