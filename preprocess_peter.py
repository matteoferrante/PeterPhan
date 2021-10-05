import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import SimpleITK as sitk

import os
from os.path import join as opj
import glob
from pathlib import Path
from torchio.transforms import HistogramStandardization
import torch

import tqdm

from scipy.stats import describe 
from scipy import stats
from skimage import filters

from argparse import ArgumentParser
import pprint

ap=ArgumentParser()
ap.add_argument("-i","--input",help="Path where all acquisition are placed",required=True)
ap.add_argument("-o","--output",help="Output Path")
ap.add_argument("-z","--z_norm",help="Wheter run z normalization or not, default=True",default=True)
ap.add_argument("-p","--plot",help="If True, it will plot the histogram before and after the normalization. It requires some time.",default=False)


args=ap.parse_args()
base_path=args.input
target_path=args.output

if target_path is None:
    print(f"[INFO] User doesn't define a target path, so output and intermediate files will be in the input directory")
    target_path=base_path


def read_image(file):
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(file)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    return image

def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.flatten()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)


    ##### START



print(f"[INFO] Scan {base_path} for repetitions and dicom files..")


devices=os.listdir(base_path)
#rep=[glob.glob(opj(base_path,device,"*\\")) for device in devices]

rep={dev:glob.glob(opj(base_path,dev,"*\\")) for dev in devices}

# just got all the filenames
rep_names={}
for k,v in rep.items():
    names=[]
    for i in v:
        name=i[str(i)[:-2].rfind('\\') + 1:]  #[:-2] is to crop \\ at the end

        names.append(name)
    rep_names[k]=names


files=[]



for k,v in rep.items():
    files+=v


print(f"[INFO] found {len(rep)} devices.")
pprint.pprint(rep)


print(f"[INFO] Training the histogram standardizer..")

t2_landmarks_path = Path(opj(target_path, "T2_landmarks.npy"))

t2_landmarks = HistogramStandardization.train(files)
torch.save(t2_landmarks, t2_landmarks_path)
landmarks_dict = {'t2': t2_landmarks}

print(f"[INFO] Dataset Generation")

datasets={}
for device in devices:
    subjects = []
    for image_path in rep[device]:
        subject = tio.Subject(t2=tio.ScalarImage(image_path))
        subjects.append(subject)
        dataset = tio.SubjectsDataset(subjects)
    datasets[device]=dataset


print(f"[INFO] Performing transformations")

rescale = tio.RescaleIntensity(out_min_max=(0, 1000))
histogram_transform = tio.HistogramStandardization(landmarks_dict)
znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)

if args.z_norm is not None:
    transform = tio.Compose([rescale,histogram_transform, znorm_transform])

else:
    transform = tio.Compose([rescale, histogram_transform])

###

transformed_datasets={}

for device,dataset in tqdm.tqdm(datasets.items()):
    transformed_imgs = [transform(sample) for sample in dataset]
    transformed_datasets[device]=transformed_imgs

print(f"[INFO] Saving outputs to {target_path}")

os.makedirs(opj(target_path,"output"),exist_ok=True)


for device,transformed_imgs in transformed_datasets.items():
    os.makedirs(opj(target_path,"output",device),exist_ok=True)
    for (i,t_img) in enumerate(transformed_imgs):

        name=rep_names[device][i][:-1]+".nii.gz"
        filename=opj(target_path,"output",device,name)
        print(filename)
        t_img.t2.save(filename)


print(f"[INFO] All files were saved to {target_path}")

info={}
for device in devices:
    info[device]= tuple(np.random.randint(256, size=3)/256)

if args.plot:
    print(f"[INFO] Computing before histogram kernel densities. This may require some time.")
    fig, axs = plt.subplots(1,2, figsize=(10, 5))

    ax1,ax2=axs
    for device,dataset in datasets.items():
        color = info[device]
        label=device
        for sample in dataset:
            img=sample.t2.data
            plot_histogram(ax1, img.flatten(), color=color)

    ax1.legend()
    ax1.set_title("Before Standardization")

    for device, dataset in transformed_datasets.items():
        color = info[device]
        label = device
        for sample in dataset:
            img = sample.t2.data.numpy()
            plot_histogram(ax2, img.flatten(), color=color)

    ax2.legend()
    ax2.set_title("After Standardization")

    filename=opj(target_path,"plot.png")
    plt.savefig(filename)
    plt.show()


