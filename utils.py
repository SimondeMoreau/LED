import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

import sobel

def compute_loss(output, target):
    lambda_1 = 1
    lambda_2 = 1

    get_gradient = sobel.Sobel().cuda()
    cos = nn.CosineSimilarity(dim=1, eps=0)
    l1 = nn.L1Loss()

    ones = torch.ones(target.size(0), 1, target.size(2),target.size(3)).float().cuda()

    #compute the gradient of the depth maps
    depth_grad = get_gradient(target)
    output_grad = get_gradient(output)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(target)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(target)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(target)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(target)

    #compute the pseudonormal maps
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    #compute the loss logl1 on the depth maps
    loss_depth = torch.abs(torch.log(output) - torch.log(target)).mean()

    #compute the loss l1 on the gradient maps
    loss_dx = l1(output_grad_dx,depth_grad_dx)
    loss_dy = l1(output_grad_dy,depth_grad_dy)

    #compute the loss cosine similarity on the normal maps
    loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

    #compute the final loss
    loss = loss_depth + lambda_1 * loss_normal + lambda_2 *(loss_dx + loss_dy)

    return loss

def calculate_metrics(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def create_figures(data, target, output, max_depth):

    imgs = data.cpu().float().numpy().transpose(0,2,3,1)
    nb_display =  4 if len(imgs) > 4 else len(imgs)
    figure_input, axes = plt.subplots(1, nb_display, figsize = (20, 5))
    for i in range(0, nb_display):
        axes[i].imshow(imgs[i])
    plt.tight_layout()

    imgs = target.cpu().float().numpy().transpose(0,2,3,1)
    figure_target, axes = plt.subplots(1, nb_display, figsize = (20, 5))
    for i in range(0, nb_display):
        depth = axes[i].imshow(imgs[i]*max_depth,cmap="turbo")
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(depth, cax=cax)
    plt.tight_layout()

    imgs = output.cpu().float().numpy().transpose(0,2,3,1)
    figure_output, axes = plt.subplots(1, nb_display, figsize = (20, 5))
    for i in range(0, nb_display):
        depth = axes[i].imshow(imgs[i]*max_depth,cmap="turbo")
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(depth, cax=cax)
    plt.tight_layout()

    return figure_input, figure_target, figure_output

def get_args_train():
    parser = argparse.ArgumentParser(description='Train a Encoder-Decoder model on DriveSim dataset')
    parser.add_argument('dataset_root', type=str, help='The root directory of the dataset')
    parser.add_argument('dataset', type=str, help='Part of the dataset to use (LED or HB)')
    parser.add_argument('--max_depth', type=int, help='The maximum depth of the dataset', default=100)
    parser.add_argument('--batch_size', type=int, help='The batch size', default=32)
    parser.add_argument('--epochs', type=int, help='The number of epochs', default=70)
    parser.add_argument('--lr', type=float, help='The learning rate', default=1e-3)
    parser.add_argument('--checkpoint_dir', type=str, help='The directory to save the checkpoints', default='checkpoints')
    parser.add_argument('--experiment_name', type=str, help='The name of the experiment', default='')
    parser.add_argument('--log_dir', type=str, help='The directory to save the tensorbard logs', default='runs')
    parser.add_argument('--device', type=str, help='The device to use (cpu or cuda)', default='cuda')
    parser.add_argument('--checkpoint', type=str, help='The checkpoint to load', default=None)
    parser.add_argument('--saving_interval', type=int, help='The interval to save the model', default=10)
    return parser.parse_args()

def get_args_test():
    parser = argparse.ArgumentParser(description='Test a Encoder-Decoder model on DriveSim dataset')
    parser.add_argument('dataset_root', type=str, help='The root directory of the dataset')
    parser.add_argument('dataset', type=str, help='Part of the dataset to use (LED or HB)')
    parser.add_argument('experiment_name', type=str, help='The name of the experiment')
    parser.add_argument('--checkpoint', type=str, help='The checkpoint to load', default="best_model.pth")
    parser.add_argument('--max_depth', type=int, help='The maximum depth of the dataset', default=100)
    parser.add_argument('--checkpoint_dir', type=str, help='The directory to save the checkpoints', default='checkpoints')
    parser.add_argument('--device', type=str, help='The device to use (cpu or cuda)', default='cuda')
    parser.add_argument('--result_dir', type=str, help='The directory to save the output metrics', default='test_results')
    parser.add_argument('--batch_size', type=int, help='The batch size', default=32)
    return parser.parse_args()