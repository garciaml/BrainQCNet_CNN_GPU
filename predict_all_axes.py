##### MODEL AND DATA LOADING
import time
start = time.time()

import torch
# torch.cuda.empty_cache()
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image

import re
import os

from helpers import makedir
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std

import argparse
import pandas as pd

from settings import img_size


# Defining arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-partid', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-masks', nargs=1, type=str, default='0')
parser.add_argument('-slicefile', nargs=1, type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(args.gpuid[0])
# specify the test image to be analyzed
test_image_dir = args.imgdir[0]
x_test_image_names = []
y_test_image_names = []
z_test_image_names = []
for root, dirs, files in os.walk(os.path.join(test_image_dir, "x")):
    for filename in files:
        x_test_image_names.append(filename)
x_test_image_names = [f for f in x_test_image_names if "png" in f]
for root, dirs, files in os.walk(os.path.join(test_image_dir, "y")):
    for filename in files:
        y_test_image_names.append(filename)
y_test_image_names = [f for f in y_test_image_names if "png" in f]
for root, dirs, files in os.walk(os.path.join(test_image_dir, "z")):
    for filename in files:
        z_test_image_names.append(filename)
z_test_image_names = [f for f in z_test_image_names if "png" in f]

test_image_paths = [os.path.join(test_image_dir, "x", test_image_name) for test_image_name in x_test_image_names] + \
                [os.path.join(test_image_dir, "y", test_image_name) for test_image_name in y_test_image_names] + \
                [os.path.join(test_image_dir, "z", test_image_name) for test_image_name in z_test_image_names]
slice_numbers = [(test_image_name.split("slice")[-1]).split(".png")[0] for test_image_name in x_test_image_names] + \
                [(test_image_name.split("slice")[-1]).split(".png")[0] for test_image_name in y_test_image_names] + \
                [(test_image_name.split("slice")[-1]).split(".png")[0] for test_image_name in z_test_image_names]
slice_numbers = [int(s) for s in slice_numbers]
axis_table = ["x"]*len(x_test_image_names) + ["y"]*len(y_test_image_names) + ["z"]*len(z_test_image_names)

if args.slicefile is not None:
    slicefile = args.slicefile[0]
    good_slices = pd.read_csv(slicefile, header=None)[0].tolist()
    print(good_slices)
    test_image_paths = [test_image_paths[i] for i in range(len(test_image_paths)) if slice_numbers[i] in good_slices]

# load the model
check_test_accu = False

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
load_model_name = args.model[0] #'10_18push0.7822.pth'

# participant identifier
partid = args.partid[0]

# predict masks
predict_masks = bool(int(args.masks[0]))

#model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

# creating the path to save results
save_analysis_path = os.path.join(test_image_dir)
makedir(save_analysis_path)

# creating log to save results
log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
#log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

cnn = torch.load(load_model_path)
cnn = cnn.cuda()
cnn_multi = torch.nn.DataParallel(cnn)

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

# loop here over all image paths
tables = []
for k, test_image_path in enumerate(test_image_paths):
    img_pil = Image.open(test_image_path).convert('RGB')
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    images_test = img_variable.cuda()
    logits = cnn_multi(images_test)
    # save each prediction and corresponding slice number in tables
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), slice_numbers[k], axis_table[k]))
        log(str(i) + ' ' + str(tables[-1]))

# get global repartition of preds for this axis
predictions = [t[0] for t in tables]
slices_masks = [t[1] for t in tables if t[0] > 0]
slices_axis = [t[2] for t in tables if t[0] > 0]
log("Slices predicted 1:" + str(slices_masks))
log("Corresponding axes: " + str(slices_axis))
log('Predicted median: ' + str(np.median(predictions)))
log('Predicted mean: ' + str(np.mean(predictions)))
label_pred_repartition = [len([el for el in predictions if el==0]), len([el for el in predictions if el==1])]
label_pred_repartition = [round(label_pred_repartition[0]/(label_pred_repartition[0] + label_pred_repartition[1]), 2), 
                        round(label_pred_repartition[1]/(label_pred_repartition[0] + label_pred_repartition[1]), 2)]
log("Repartition of predictions: " + str(label_pred_repartition[0]*100) + " percent of slices predicted label 0; " + \
    str(label_pred_repartition[1]*100) + " percent of slices predicted label 1.")    

end = time.time()
log("Processing time: " + str(end - start))

logclose()
torch.cuda.empty_cache()
