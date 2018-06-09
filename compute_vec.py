import os
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import datasets
import models
import h5py

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
#store the feature of perfect_500K dataset in xx_train[]
#store the image file name of perfect_500K dataset img_name[]
xx_train=[]
img_name=[]

# load model from pretrained resnet18
model = models.__dict__['resnet18'](low_dim=128)
checkpoint = torch.load('lemniscate_resnet18.pth.tar')
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#load perfect_500k dataset from /home/ivlab/JPGDATA
traindir = os.path.join('/home/ivlab','JPGDATA')
#traindir = os.path.join('datas','train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolderInstance(
    traindir,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset)

# compute the feature vector and save the features in xx_train[]
for i, (input, _, index) in enumerate(train_loader):
    # measure data loading time

    index = index.cuda(async=True)
    input_var = torch.autograd.Variable(input)

    # compute output
    feature = model(input_var)
    feature=feature.cpu()
    xx_train.append(feature.data.numpy())
    img_name.append(train_dataset.imgs[i][0])
    print train_dataset.imgs[i][0]

xx_train=np.array(xx_train)
img_name=np.array(img_name)

# save feature vectors and img_name to h5 format file
file = h5py.File ("data_feas.h5", "w")
file.create_dataset('img_data', data=xx_train)
file.create_dataset('img_index', data=img_name)
file.close ()