import os
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import datasets
import models
import h5py
import csv
# define the cosin similarity function
def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

# load the features of perfect_500k from h5 foramt file
h5f = h5py.File('data_feas.h5','r')
data_vectors = h5f['img_data'][:]
img_name = h5f['img_index'][:]
h5f.close()




hit=0
#load pretrained model
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model = models.__dict__['resnet18'](low_dim=128)
checkpoint = torch.load('lemniscate_resnet18.pth.tar')
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()
#load query data set from root/datas/val1
traindir = os.path.join('datas', 'val1')
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


# compute the cosin similarity with perfect_500k dataset for each feature vector in query data set
for i, (input, _, index) in enumerate(train_loader):
    #test_index=the image ID of query image
    test_index=train_dataset.imgs[i][0].split('/')[3].split('.')[0]
    print test_index
    # measure data loading time
    index = index.cuda(async=True)
    input_var = torch.autograd.Variable(input)

    # compute output
    feature = model(input_var)
    feature=feature.cpu()
    #dist store the similarity
    dist = []
    for index in range(data_vectors.shape[0]):
        op2=cos_sim(feature.data.numpy().reshape(128),data_vectors[index].reshape(128))
        #op2 = np.linalg.norm(feature.data.numpy() - data_vectors[index])
        dist.append(op2)
    rank_ID = np.argsort(dist)[::-1][:10]
    top_img = img_name[rank_ID]
    print (top_img[0])

    #plot the query result
    plt.figure(num=i, figsize=(8, 8))
    plt.figure(num='image_retireval', figsize=(8, 8))
    plt.subplot(4, 4, 1)
    plt.title('query image')
    img = Image.open(train_dataset.imgs[i][0])
    plt.imshow(img)
    index=2

    #compute the hit rate
    with open('datas/val.csv') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            if row['ID']==test_index:
                target_label=row['Training Image ID']
    print target_label

    for i in top_img:
        plt.subplot(4, 4, index)
        plt.title(index)
        img = Image.open(i)
        plt.imshow(img)
        index = index + 1
        if str(target_label)==str(i.split("/")[5].split(".")[0]):
            hit=hit+1
        print i.split("/")[5].split(".")[0]

    plt.show()
    print "hit:"+str(hit)

print hit





