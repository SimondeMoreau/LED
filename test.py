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
import torch.nn as nn
from tqdm import tqdm
import sys
from prettytable import PrettyTable
from utils import calculate_metrics, get_args_test

args = get_args_test()

dataset_to_test = args.dataset
experiment_name = args.experiment_name
max_depth = args.max_depth
drivesim_dir = os.path.join(args.dataset_root, dataset_to_test)

print("Testing on", args.dataset_root, "with", args.dataset, "dataset")
print("Experiment name:", experiment_name)
print("Max depth:", args.max_depth)
print("Batch size:", args.batch_size)
print("Checkpoint directory:", args.checkpoint_dir)
print("Checkpoint:", args.checkpoint)
print("Result directory:", args.result_dir)



checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)

if(not os.path.exists(checkpoint_path)):
    print("Checkpoint not found")
    sys.exit(1)

# loading the dataset
test_dirs = glob(os.path.join(drivesim_dir,"wuppertal_*"))
transform = transforms.Compose([ToTensor(), transforms.CenterCrop(640),transforms.Resize(320,antialias=True)])
dataset_test = DriveSimDataset(test_dirs, "ldr_color", "distance_to_image_plane", max_file=5000,transform=transform, target_transform=transform, max_depth=max_depth,max_depth_placeholder=max_depth)
ltest = len(dataset_test)
print("Test :",ltest)

batch_size = args.batch_size
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=16)

# device
if args.device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print("Device:", device)


model = UNet(n_channels=3, n_classes=1, bilinear=False)
#load the checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print("Checkpoint loaded from", checkpoint_path)
print("Training Epoch loaded:", checkpoint['epoch'])
model.to(device)

roi = [20,165,270,210]
mask_roi_np = np.zeros((320,320,1))
mask_roi_np[roi[1]:roi[3],roi[0]:roi[2]] = 1

#test loop torch
model.eval()

metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
metrics = {key: [] for key in metric}
metrics_roi = {key: [] for key in metric}
print("Testing...")
with torch.no_grad():
    loop = tqdm(test_loader)
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(device), target.to(device)

        target = target / max_depth
        output = model(data)
        
        output = torch.nn.functional.sigmoid(output)

        output_unnorm = output * max_depth
        target_unnorm = target * max_depth

        res = calculate_metrics(target_unnorm.cpu().numpy(), output_unnorm.cpu().numpy())
        for i,name in enumerate(metric):
            metrics[name].append(res[i])

        #roi LED
        mask_roi = torch.from_numpy(mask_roi_np).to(device)
        mask_roi = mask_roi.permute(2,0,1)
        mask_roi = mask_roi.expand_as(output).bool()

            
        output_unnorm_roi = output[mask_roi] * max_depth
        target_unnorm_roi = target[mask_roi] * max_depth

        res = calculate_metrics(target_unnorm_roi.cpu().numpy(), output_unnorm_roi.cpu().numpy())
        for i,name in enumerate(metric):
            metrics_roi[name].append(res[i])


metrics = {key: np.mean(value) for key, value in metrics.items()}
metrics_roi = {key: np.mean(value) for key, value in metrics_roi.items()}   

print("Experiment :",experiment_name)
print("Results for max_depth",max_depth)

print("Result full image")
pt = PrettyTable()
for key, value in metrics.items():
    pt.add_column(key, [value])
pt.float_format = "0.4"
print(pt)

print("Result ROI LED")
pt = PrettyTable()
for key, value in metrics_roi.items():
    pt.add_column(key, [value])
pt.float_format = "0.4"
print(pt)

test_results_dir = args.result_dir
os.makedirs(test_results_dir, exist_ok=True)
file_result = "results_" + experiment_name + "_testdataset_" + dataset_to_test + ".csv"
filename = os.path.join(test_results_dir, file_result)

header = "RMSE;Abs Rel;Log 10;RMSE log;Silog;Sq Rel;A1;A2;A3"

#write as csv with header
with open(filename, 'w') as f:
    f.write("Experiment;" + experiment_name + ";dataset_to_test;" + dataset_to_test +"\n")
    f.write("Full image\n")
    f.write(header + "\n")
    f.write(str(metrics["rmse"]) + ";" + str(metrics["abs_rel"]) + ";" + str(metrics["log_10"]) + ";" + str(metrics["rmse_log"]) + ";" + str(metrics["silog"]) + ";" + str(metrics["sq_rel"]) + ";" + str(metrics["a1"]) + ";" + str(metrics["a2"]) + ";" + str(metrics["a3"]) + "\n")
    f.write("ROI\n")
    f.write(header + "\n")
    f.write(str(metrics_roi["rmse"]) + ";" + str(metrics_roi["abs_rel"]) + ";" + str(metrics_roi["log_10"]) + ";" + str(metrics_roi["rmse_log"]) + ";" + str(metrics_roi["silog"]) + ";" + str(metrics_roi["sq_rel"]) + ";" + str(metrics_roi["a1"]) + ";" + str(metrics_roi["a2"]) + ";" + str(metrics_roi["a3"]) + "\n")

print("Results saved in", filename)