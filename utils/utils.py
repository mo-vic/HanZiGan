import math

import torch
from torch._six import inf
from torchvision.utils import make_grid

import numpy as np
from tqdm import tqdm


def _grad_norm(parameters, norm_type=2):
    r"""Compute gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


def train(model, dataloader, criterion, optimizer, use_gpu, writer, epoch, scheduler, num_fakes, flip_rate, show_freq):
    all_acc = []
    all_d_loss = []
    all_g_loss = []

    d_optimizer, g_optimizer = optimizer
    d_scheduler, g_scheduler = scheduler

    for idx, data in tqdm(enumerate(dataloader), desc="Training Epoch {}".format(epoch)):
        # train discriminator
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        labels = torch.cat([torch.bernoulli(torch.ones((data.size(0), 1)) * flip_rate), torch.zeros((num_fakes, 1))],
                           dim=0)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data, mode='D')
        d_loss = criterion(outputs, labels)
        d_loss.backward()
        d_optimizer.step()

        all_d_loss.append(d_loss.item())
        acc = (torch.ge(outputs, 0.5).long().data == labels.long().data).double().mean()
        all_acc.append(acc.item())

        writer.add_scalar("train_d_grad_norm", _grad_norm(model.parameters()),
                          global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("train_d_loss", d_loss.item(), global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("train_acc", acc.item(), global_step=epoch * len(dataloader) + idx)

        # train generator
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        fake_images, outputs = model(mode='G')
        labels = torch.ones((num_fakes, 1))
        if use_gpu:
            labels = labels.cuda()
        g_loss = criterion(outputs, labels)
        g_loss.backward()
        g_optimizer.step()

        all_g_loss.append(g_loss.item())

        if idx % show_freq == 0:
            fake_images = make_grid(fake_images, nrow=round(math.sqrt(num_fakes)))
            writer.add_image("fake_images", fake_images, global_step=epoch * len(dataloader) + idx)
            real_images = make_grid(data, nrow=round(math.sqrt(data.size(0))))
            writer.add_image("real_images", real_images, global_step=epoch * len(dataloader) + idx)

        writer.add_scalar("train_g_grad_norm", _grad_norm(model.parameters()),
                          global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("train_g_loss", g_loss.item(), global_step=epoch * len(dataloader) + idx)

    writer.add_scalar("acc", np.mean(all_acc).item(), global_step=epoch)
    writer.add_scalar("d_loss", np.mean(all_d_loss).item(), global_step=epoch)
    writer.add_scalar("g_loss", np.mean(all_g_loss).item(), global_step=epoch)

    d_scheduler.step(np.mean(all_d_loss).item())
    g_scheduler.step(np.mean(all_g_loss).item())

    print("Epoch {}: total discriminator loss: {}".format(epoch, np.mean(all_d_loss).item()), end=',')
    print("total generator loss: {}, global accuracy:{}.".format(np.mean(all_g_loss), np.mean(all_acc)))
