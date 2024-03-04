import torch
from dataset.DriveSimDataset import DriveSimDataset
from model import UNet
import numpy as np
from torchvision.transforms import ToTensor
from torchvision import transforms
import torchvision
import os
from glob import glob
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
from utils import compute_loss, calculate_metrics, create_figures, get_args_train


#fix random seed
torch.manual_seed(0)
np.random.seed(0)

args = get_args_train()

print("Training on", args.dataset_root, "with", args.dataset, "dataset")
print("Max depth:", args.max_depth)
print("Batch size:", args.batch_size)
print("Epochs:", args.epochs)
print("Learning rate:", args.lr)
print("Checkpoint directory:", args.checkpoint_dir)
print("Log directory:", args.log_dir)
if(args.checkpoint is not None):
    print("Checkpoint:", args.checkpoint)


gen_type = args.dataset
max_depth = args.max_depth
if args.experiment_name != "":
    experiment_name =  args.experiment_name + '_' + gen_type + '_' + str(max_depth) + 'd_1'
else:
    experiment_name = gen_type + '_' + str(max_depth) + 'd_1'
print("Experiment name:", experiment_name)

# loading the dataset
drivesim_dir = os.path.join(args.dataset_root, gen_type)
train_dirs = glob(os.path.join(drivesim_dir,"gen15_*")) + glob(os.path.join(drivesim_dir,"china_*"))
val_dirs = glob(os.path.join(drivesim_dir,"hamburg_*"))
test_dirs = glob(os.path.join(drivesim_dir,"wuppertal_*"))

transform = transforms.Compose([ToTensor(), transforms.CenterCrop(640),transforms.Resize(320,antialias=True)])

dataset_train = DriveSimDataset(train_dirs, "ldr_color", "distance_to_image_plane",transform=transform, target_transform=transform, max_depth=max_depth,max_file=5000, max_depth_placeholder=max_depth)
dataset_val = DriveSimDataset(val_dirs, "ldr_color", "distance_to_image_plane", max_file=5000,transform=transform, target_transform=transform, max_depth=max_depth, max_depth_placeholder=max_depth)
dataset_test = DriveSimDataset(test_dirs, "ldr_color", "distance_to_image_plane", max_file=5000,transform=transform, target_transform=transform, max_depth=max_depth, max_depth_placeholder=max_depth)

ltrain = len(dataset_train)
ltest = len(dataset_test)
lval = len(dataset_val)
lsum = ltrain + ltest + lval

print("Dataset size:")
print("Train :",ltrain, "(",ltrain/lsum*100,"%)")
print("Test :",ltest, "(",ltest/lsum*100,"%)")
print("Validation :",lval, "(",lval/lsum*100,"%)")

batch_size = args.batch_size
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=16)

# device
if args.device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print("Device:", device)

epoch_start = 1

# model
model = UNet(n_channels=3, n_classes=1, bilinear=False)

if(args.checkpoint is not None):
    if not os.path.exists(args.checkpoint):
        print("Checkpoint not found")
        sys.exit(1)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_start = checkpoint['epoch'] + 1
    print("Checkpoint loaded from", args.checkpoint)
    print("Starting from epoch:", epoch_start)

model.to(device)

# optimizer
lr = args.lr

optimizer = optim.AdamW(model.parameters(), lr=lr)
if args.checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# creating tensorboard writer and checkpoint directory
writer = SummaryWriter(os.path.join(args.log_dir, experiment_name))
checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)

epochs = args.epochs

best_val_rmse = np.inf
best_epoch = 0

if args.checkpoint is not None:
    best_val_rmse = checkpoint['rmse'] if 'rmse' in checkpoint else np.inf
    best_epoch = checkpoint['epoch']

#train loop
for epoch in range(epoch_start,epochs+1):
    model.train()
    epoch_loss = 0

    loop = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(device), target.to(device)

        target = target / max_depth

        optimizer.zero_grad()

        output = model(data)
        output = torch.nn.functional.sigmoid(output)

        loss = compute_loss(output,target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        loop.set_description(f"Training Epoch [{epoch}/{epochs}]")
        loop.set_postfix(loss=loss.item(), epoch_loss=epoch_loss)

    writer.add_scalar('Training loss',epoch_loss,epoch)

    #validation loop 
    model.eval()
    val_loss = 0
    metrics_name = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
    metrics = {name: [] for name in metrics_name}

    with torch.no_grad():
        loop = tqdm(val_loader)
        for batch_idx, (data, target) in enumerate(loop):
            data, target = data.to(device), target.to(device)

            target = target / max_depth
            output = model(data)
            
            output = torch.nn.functional.sigmoid(output)
            
            output_unnorm = output * max_depth
            target_unnorm = target * max_depth

            loss = compute_loss(output,target)
            
            val_loss += loss.item()
            loop.set_description(f"Validation Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item(), validation_loss=val_loss)

            res_metrics = calculate_metrics(target_unnorm.cpu().float().numpy().flatten(), output_unnorm.cpu().float().numpy().flatten())

            for i, name in enumerate(metrics_name):
                metrics[name].append(res_metrics[i])

    # compute the mean of the metrics
    metrics = {name: np.mean(metrics[name]) for name in metrics_name}

    # log validation metrics
    writer.add_scalar('Validation loss', epoch_loss, epoch)
    for name in metrics_name:
        writer.add_scalar('Validation ' + name, metrics[name], epoch)
    
    # log example images from the validation set
    figure_input, figure_target, figure_output = create_figures(data, target, output, max_depth)
    writer.add_figure('Input', figure_input, epoch)
    writer.add_figure('Target', figure_target, epoch)
    writer.add_figure('Output', figure_output, epoch)

    rmse = metrics["rmse"]

    # save the best model
    if rmse < best_val_rmse:
        best_val_rmse = rmse
        best_epoch = epoch
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_loss': epoch_loss,
                'val_loss': val_loss,
                'rmse': rmse,
                }, os.path.join(checkpoint_dir, f"best_model.pth"))
            
        print("Best model saved at epoch", epoch)

    # save the model every saving_interval epochs
    if epoch % args.saving_interval == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_loss': epoch_loss,
                'val_loss': val_loss,
                'rmse': rmse,
                }, os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth"))
