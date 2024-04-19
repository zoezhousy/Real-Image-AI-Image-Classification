# %% [markdown]
# # **Image classification - Assignment 2**

# %%
import os
import cv2
import time
import numpy as np
from sklearn import datasets
from pprint import pprint
from collections import namedtuple, OrderedDict
from matplotlib import pyplot as plt
from skimage import feature
from skimage.feature import hog
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from sklearn.svm import SVC
import jsonlines # read annotation from annatation.jsonl
from torch.functional import Tensor
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# %% [markdown]
# ## **Data Preprocessing**

# %%
# load images
root = os.path.join('./','dataset/')
label_path = os.path.join(root,'annatation.jsonl')


class Sample:
    def __init__(self, idx=0, fname='', img=None, feat=None, label=None):
        self.idx = idx
        self.fname = fname
        self.img = img
        self.feat = feat
        self.label = label
        self.pred = None

# load annotation, file,
annotations = [] # annotation
filenames = [] # file name

if os.path.exists(label_path):
    # print('File exists')
    with jsonlines.open(label_path) as reader:
        # save annotation and file name into list
        for line in reader:
            filename = line['id']
            annotation = line['annotation']
            if annotation == 'aiart':
                anno = 1 # ai art: annotation = 1
            else:
                anno = 0 # real picture: annotation = 0
            filenames.append(filename)
            annotations.append(anno)
else:
    raise ValueError('Invalid label file path [%s]'%label_path)

filenames = [word + '.jpg' for word in filenames] # with .jpg
print("Annotation:", annotations)
print("File names: ",filenames)

# split annotation, ids into train dataset and val dataset
Anno_train, Anno_val, file_train, file_val = train_test_split(annotations, filenames, test_size=0.2, random_state=42)

# save 'train' or 'val' label attribute into the list 'labels'
labels = [None] * len(filenames) # empty list of labels (train/val)
for i in file_train:
    index = filenames.index(i)
    labels[index] = 'train'

for j in file_val:
    index = filenames.index(j)
    labels[index] = 'val'
print("Labels: ", labels)


# reference: Tutorial 7 tasks
samples = {'train': [], 'val': [], 'all':[]}
idx = 0
shape = np.empty((3, 3))
for fname, type, label in zip(filenames, annotations, labels):
    type = int(type)
    if idx % 4 == 0:
        plt.figure(figsize=(16, 4))
    plt.subplot(1,4,idx%4+1)
    plt.title(f'{fname} in G{type}({label})')
    fpath = os.path.join(root,fname)
    if not os.path.isfile(fpath):
        raise ValueError("%s not fould" % fpath)
    else:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB
        if idx == 0:
            H, W, C = img.shape
            shape = H, W, C
        else:
            img = cv2.resize(img, (W,H))
        
        plt.imshow(img)

        samples[label].append(Sample(idx, fname, img, None, type))
        samples['all'].append(samples[label][-1])
    idx +=1



# %% [markdown]
# # Preprocessing test dataset

# %%
# preprocessing the test dataset
# load images
realpic_root = os.path.join('./','testDataset/realpic/')
realpic_sample = []     # sample list for real pictures
filenames_realpic = []  # filename list

# for ai art images
aiart_root = os.path.join('./','testDataset/aiart/')
aiart_sample = []   # sample list for ai art
filenames_aiart = []    # filename list

H, W, C = shape

def get_sample_from_file(root, filenames, sample):
    files = os.listdir(root)
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.isfile(file_path):  # Check if it's a file
            filenames.append(file)
        else:
            raise ValueError('Invalid label file path [%s]'%file_path)

    idx = 0
    label = ''
    type = 0
    if root == realpic_root:
        type = 0
        label = 'test_realpic'
    elif root == aiart_root:
        type = 1
        label = 'test_aiart'
    
    for fname in filenames:
        
        if idx % 4 == 0:
            plt.figure(figsize=(16, 4))
        plt.subplot(1,4,idx%4+1)
        plt.title(f'{fname} in G{type}({label})')
        fpath = os.path.join(root,fname)
        if not os.path.isfile(fpath):
            raise ValueError("%s not fould" % fpath)
        else:
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB
            img = cv2.resize(img, (W,H))
            plt.imshow(img)
            sample.append(Sample(idx, fname, img, None, type))
            
        idx += 1

get_sample_from_file(realpic_root, filenames_realpic, realpic_sample)
get_sample_from_file(aiart_root, filenames_aiart, aiart_sample)



# %% [markdown]
# ## Feature Extraction

# %%
## extract features
# This code contains feature extraction functions using SIFT and gray histogram, 
# as well as distance calculation functions for further calculation. 
# It then processes samples by extracting features and flattening them. 
# The code seems to be part of a computer vision project for image classification.

sift = cv2.SIFT_create()

def get_feat(img):
    return hog(img, channel_axis=2)
    # return sift_feat(img)
    # return gray_histogram(img)
    

def calc_distance(x, y):
    return L2_distance(x, y)
    # return L2_distance_sift(x, y)


def sift_feat(img):
    kps, des =  sift.detectAndCompute(img, None)
    responses = [kp.response for kp in kps]
    order = np.argsort(responses)[::-1]
    return np.array(des[order[:30]])

def gray_histogram(img: np.array, norm: bool = True) -> np.array:
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = np.array([len(img[img == i]) for i in range(256)])
    if norm:
        return hist / np.size(img)
    return hist



def L2_distance(x, y):
    return ((x - y) ** 2).sum() ** 0.5

def L2_distance_sift(x, y):
    dist = ((x[:, None] - y[None, :])**2).sum(axis=-1).min(axis=-1)
    dist.sort()
    return dist[:15].mean()


for sample in samples['train']:
    sample.feat = get_feat(sample.img).flatten()

for sample in samples['val']:
    sample.feat = get_feat(sample.img).flatten()

for sample in samples['all']:
    sample.feat = get_feat(sample.img).flatten()

for sample in aiart_sample:
    sample.feat = get_feat(sample.img).flatten()

for sample in realpic_sample:
    sample.feat = get_feat(sample.img).flatten()



# %% [markdown]
# ## Visualizing feature space

# %%
## visualize images in feature space using t-SNE with IDs and Labels for image

sns.set(rc={'figure.figsize':(8,6)})
palette = sns.color_palette("bright", 4)

features = [sample.feat for sample in (samples['train'] + samples['val'])]
labels = [sample.label for sample in (samples['train'] + samples['val'])]
ids = [sample.fname for sample in (samples['train'] + samples['val'])]


features = StandardScaler().fit_transform(features) 

tsne = TSNE()
X_embedded = tsne.fit_transform(features)
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], markers=ids, hue=labels, legend='full', palette=palette)
# for i, (x, y) in enumerate(X_embedded):
#     plt.text(x, y, ids[i])


# %% [markdown]
# ### 1. kNN classifiers

# %%
# for each image in the testing list, search its NNs
# display in each row the target image X to classify, the NNs, label assigned to X, and the ground truth label

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

def kNN(test_sample, train_samples, k=3):
    distances = []
    for sample in train_samples:
        distance = calc_distance(test_sample.feat, sample.feat)
        distances.append((distance, sample))
    distances = sorted(distances, key=lambda x: x[0])[:k]
    label_count = {}
    plt.figure(figsize=((k+1)*4, 4))
    plt.subplot(1, k+1, 1)
    plt.title(f'{test_sample.fname} in G{test_sample.label}(val)')
    plt.imshow(test_sample.img)
    for i, (distance, sample) in enumerate(distances):
        plt.subplot(1, k+1, i+2)
        plt.title(f'{sample.fname} in G{sample.label}(train)')
        plt.imshow(sample.img)
        label_count[sample.label] = label_count.get(sample.label, 0) + 1
        max_label, max_count = -1, 0  
    for label, count in label_count.items():
        if count > max_count:
            max_label, max_count = label, count
    test_sample.pred = max_label
    

def kNNClassifier(training_samples, testing_samples):
    correct = 0
    # print(len(testing_samples))
    for sample in testing_samples:
        kNN(sample, training_samples)
        if sample.label == 1:
            if sample.pred == 1:
                print(sample.fname, 'with label \"AI art\" is predicted as \"AI art\"')
                correct += 1
            else:
                print(sample.fname,'with label \"AI art\" is predicted as \"Real Picture\"')
        else:
            if sample.pred == 1:
                print(sample.fname, 'with label \"Real Picture\" is predicted as \"AI art\"')
            else:
                print(sample.fname,'with label \"Real Picture\" is predicted as \"Real Picture\"')
                correct += 1
    

    return correct / len(testing_samples)



print("Accuracy of kNN classifier: {:.0%}".format(kNNClassifier(samples['train'], samples['val'])))

# using unseen pictures(real picture)
# test with real senarios
print("Accuracy of kNN classifier for real picture samples: {:.0%}".format(kNNClassifier(samples['all'], realpic_sample)))

# test with AI picture
print("Accuracy of kNN classifier for real picture samples: {:.0%}".format(kNNClassifier(samples['all'], aiart_sample)))




# %% [markdown]
# ### 2. SVM Model

# %%
# train with SVM
model = SVC(kernel='rbf')

def svmClassifier(training_samples, testing_samples):
    train_samples = [sample.feat for sample in training_samples]
    train_labels = [sample.label for sample in training_samples]
    model.fit(train_samples, train_labels)

    # test with SVM
    test_samples = [sample.feat for sample in testing_samples]
    results = model.predict(test_samples)

    correct = 0
    for sample, result in zip(testing_samples, results):
        sample.pred = result
        if sample.label == 1:
            if sample.pred == 1:
                print(sample.fname, 'with label \"AI art\" is predicted as \"AI art\"')
                correct += 1
            else:
                print(sample.fname,'with label \"AI art\" is predicted as \"Real Picture\"')
        else:
            if sample.pred == 1:
                print(sample.fname, 'with label \"Real Picture\" is predicted as \"AI art\"')
            else:
                print(sample.fname,'with label \"Real Picture\" is predicted as \"Real Picture\"')
                correct += 1
    scores = cross_val_score(model, train_samples, train_labels, cv=3)  # 5-fold cross-validation
    print(f'Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}%')
    return correct / len(results) # return accuracy

# display the results (target images with the predicted labels and ground truth label)
print("Accuracy of SVM classifier: {:.0%}".format(svmClassifier(samples['train'], samples['val'])))

# display the results of real picture
print("Accuracy of SVM classifier-realpic: {:.0%}".format(svmClassifier(samples['all'], realpic_sample)))

# results of ai art
print("Accuracy of SVM classifier-aiart: {:.0%}".format(svmClassifier(samples['all'], aiart_sample)))


# %% [markdown]
# ### **3. SDG Classifier**

# %%
def SGDClassifier_sample(training_samples,testing_samples):
    # use SDG classifier
    train_samples = [sample.feat for sample in training_samples]
    train_labels = [sample.label for sample in training_samples]

    # StandardScaler() aims to normalize input dataset via (x-mean)/var
    # SGDClassifier(): max_iter (iteration times); tol is used for early stop; when learning rate is constant, eta0 is seen as learning rate; log loss means logistic regression
    # make_pipeline: conduct StandardScaler() first, then create classifier


    classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=50000, tol=1e-3, learning_rate='constant', eta0=0.1, loss='log_loss'))


    # call classifier for trainingS
    classifier.fit(train_samples, train_labels)

    test_samples = [sample.feat for sample in testing_samples]
    test_labels = [sample.label for sample in testing_samples]

    # call classifier for predict
    results = classifier.predict(test_samples)

    correct = 0
    for sample, result in zip(testing_samples, results):
        sample.pred = result
        if sample.label == 1:
            if sample.pred == 1:
                print(sample.fname, 'with label \"AI art\" is predicted as \"AI art\"')
                correct += 1
            else:
                print(sample.fname,'with label \"AI art\" is predicted as \"Real Picture\"')
        else:
            if sample.pred == 1:
                print(sample.fname, 'with label \"Real Picture\" is predicted as \"AI art\"')
            else:
                print(sample.fname,'with label \"Real Picture\" is predicted as \"Real Picture\"')
                correct += 1
    
    
    return correct / len(results)

# display the results (target images with the predicted labels and ground truth label)
print("Accuracy of SDG classifier: {:.0%}".format(SGDClassifier_sample(samples['train'], samples['val'])))
print("Accuracy of SDG classifier for real picture: {:.0%}".format(SGDClassifier_sample(samples['all'], realpic_sample)))
print("Accuracy of SDG classifier for AI art: {:.0%}".format(SGDClassifier_sample(samples['all'], aiart_sample)))

# %% [markdown]
# ## **2. A simple Neural Network**

# %%
def simpleNNClassifier(training_samples, testing_samples):
    train_samples = [sample.feat for sample in training_samples]
    train_labels = [sample.label for sample in training_samples]

    input_size = len(train_samples[0])
    output_size = 1
    learning_rate = 1

    # To define a neural network class by pytorch, you have to inhert nn.Module class.
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()

            # input_size corresponds to input feature dimension.
            # output_size coressponds to class number. 
            self.linear = nn.Linear(input_size, output_size)

            #Then, we use sigmoid activation function to gain the probability.
            # when probability is larger than 0.5, we treat it as positive 1. Else, we treat it as negative 0.
            self.sigmoid = nn.Sigmoid()

        # forward function is inherted from parent's class. x denotes the input feature. 
        def forward(self, x):
            y_pred = self.linear(x)
            y_pred = self.sigmoid(y_pred)
            return y_pred


    x_train = torch.tensor(train_samples).to(torch.float32)
    y_train = torch.tensor(train_labels).reshape(-1, 1).to(torch.float32)

    test_samples = [sample.feat for sample in testing_samples]
    test_labels = [sample.label for sample in testing_samples]
    x_test = torch.tensor(test_samples).to(torch.float32)

    # create model
    model = SimpleNN()

    # # check if CUDA is available
    # if torch.cuda.is_available():
    #     model = model.to('cuda')

    # create loss function. BCE is binary cross entropy loss
    criterion = nn.BCELoss()

    # create optimizer. 1st parameter: the parameters will be optimized; 2nd: learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training
    # set training flag 
    model.train()
    num_epochs = 50000
    for epoch in range(num_epochs):
        # if torch.cuda.is_available():
        #     inputs = Variable(x_train).cuda()
        #     target = Variable(y_train).cuda()
        # else:
        inputs = Variable(x_train)
        target = Variable(y_train)
    
        # forward() function
        out = model(inputs)

        # calculate loss
        loss = criterion(out, target)

        # clear gradient
        optimizer.zero_grad()

        # backward propagation
        loss.backward()

        # Updating parameters via SGD
        optimizer.step()
    
        if (epoch+1) % 10000 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'
                .format(epoch+1, num_epochs, loss.item()))

    # testing
    model.eval()
    results = model(Variable(x_test))

    # display the results
    correct = 0
    for sample, result in zip(testing_samples, results):
        sample.pred = 1 if result > 0.5 else 0
        if sample.label == 1:
            if sample.pred == 1:
                print(sample.fname, 'with label \"AI art\" is predicted as \"AI art\"')
                correct += 1
            else:
                print(sample.fname,'with label \"AI art\" is predicted as \"Real Picture\"')
        else:
            if sample.pred == 1:
                print(sample.fname, 'with label \"Real Picture\" is predicted as \"AI art\"')
            else:
                print(sample.fname,'with label \"Real Picture\" is predicted as \"Real Picture\"')
                correct += 1
    return correct / len(results)

# display the results (target images with the predicted labels and ground truth label)
print("Accuracy of 1-layer Simple Neural Network classifier: {:.0%}".format(simpleNNClassifier(samples['train'], samples['val'])))
print("Accuracy of 1-layer Simple Neural Network classifier for real picture: {:.0%}".format(simpleNNClassifier(samples['all'], realpic_sample)))
print("Accuracy of 1-layer Simple Neural Network classifier for AI arts pics: {:.0%}".format(simpleNNClassifier(samples['all'], aiart_sample)))

# %% [markdown]
# ## **3. Multi-Layer Neural Network**

# %%
def MultiNNClassifier(training_samples, testing_samples):
    train_samples = [sample.feat for sample in training_samples]
    train_labels = [sample.label for sample in training_samples]

    input_size = len(train_samples[0])
    output_size = 1
    learning_rate = 1

    x_train = torch.tensor(train_samples).to(torch.float32)
    y_train = torch.tensor(train_labels).reshape(-1, 1).to(torch.float32)

    class SimpleNN_3layer(nn.Module):
        def __init__(self):
            super(SimpleNN_3layer, self).__init__()
            self.linear1 = nn.Linear(input_size, int(input_size/2))
            self.act1 = nn.ReLU()
            self.linear2 = nn.Linear(int(input_size/2), output_size)
            self.act2 = nn.Sigmoid()
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.act1(x)
            x = self.linear2(x)
            y_pred = self.act2(x)
            return y_pred

    model = SimpleNN_3layer()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training and testing
    model.train()
    num_epochs = 50000
    for epoch in range(num_epochs):
        # if torch.cuda.is_available():
        #     inputs = Variable(x_train).cuda()
        #     target = Variable(y_train).cuda()
        # else:
        inputs = Variable(x_train)
        target = Variable(y_train)
    
        # forward
        out = model(inputs)
        loss = criterion(out, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (epoch+1) % 10000 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'
                .format(epoch+1, num_epochs, loss.item()))

    test_samples = [sample.feat for sample in testing_samples]
    test_labels = [sample.label for sample in testing_samples]
    x_test = torch.tensor(test_samples).to(torch.float32)
    model.eval()
    results = model(Variable(x_test))

    # display the results
    correct = 0
    for sample, result in zip(samples['val'], results):
        sample.pred = 1 if result > 0.5 else 0
        if sample.label == 1:
            if sample.pred == 1:
                print(sample.fname, 'with label \"AI art\" is predicted as \"AI art\"')
                correct += 1
            else:
                print(sample.fname,'with label \"AI art\" is predicted as \"Real Picture\"')
        else:
            if sample.pred == 1:
                print(sample.fname, 'with label \"Real Picture\" is predicted as \"AI art\"')
            else:
                print(sample.fname,'with label \"Real Picture\" is predicted as \"Real Picture\"')
                correct += 1
    return correct / len(results)

# display the results (target images with the predicted labels and ground truth label)
print("Accuracy of 3-layer Simple Neural Network classifier: ", "{:.0%}".format(MultiNNClassifier(samples['train'], samples['val'])))
print("Accuracy of 3-layer Simple Neural Network classifier for real picture: ", "{:.0%}".format(MultiNNClassifier(samples['all'], realpic_sample)))
print("Accuracy of 3-layer Simple Neural Network classifier for AI art: ", "{:.0%}".format(MultiNNClassifier(samples['all'], aiart_sample)))




# %% [markdown]
# ## **Deep Neural Network**

# %%
# define device, gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset construction
# This time we don't have to extract the features. Deep neural networks usually take the images as the input directly.

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
def img_transform(img_rgb, transform=None):
    """
    transform images
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("there is no transform")

    img_t = transform(Image.fromarray(img_rgb))
    return img_t


# define a classifier following the network
class classification_head(nn.Module):
	def __init__(self,in_ch,num_classes):
		super(classification_head,self).__init__()
		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
		self.fc = nn.Linear(in_ch,num_classes)

	def forward(self, x):
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x


# define ResNet model. 
# a simple tips here: you can follow the execute sequence in forward() to understand what a network is.  
# For original ResNet, its final layer will output 1000 class number. Here, we change it for our task.
class Net(nn.Module):
	def __init__(self, num_class,pretrained=True):
		super(Net,self).__init__()
		model = models.resnet50(pretrained=pretrained)
		self.backbone =  nn.Sequential(*list(model.children())[:-2]) #remove the last Avgpool and Fully Connected Layer
		self.classification_head = classification_head(2048, num_class)
										
	def forward(self,x):
		x = self.backbone(x)
		output = self.classification_head(x)
		return output


# creat a model
model = Net(1)

# fix the weights of ResNet except the last layer. This is because the training set is small. 
for p in model.backbone.parameters():
  p.requires_grad = False

model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

model.train()

def deepNNClassification(training_samples, testing_samples):
    # load data
    train_imgs = [img_transform(sample.img, inference_transform) for sample in training_samples]
    train_imgs = torch.stack(train_imgs, dim=0)

    test_imgs = [img_transform(sample.img, inference_transform) for sample in testing_samples]
    test_imgs = torch.stack(test_imgs, dim=0)

    # training
    train_samples = [sample.feat for sample in training_samples]
    train_labels = [sample.label for sample in training_samples]
    x_train = torch.tensor(train_samples).to(torch.float32)
    y_train = torch.tensor(train_labels).reshape(-1, 1).to(torch.float32)
    num_epochs = 200
    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            inputs = Variable(train_imgs).cuda()
            target = Variable(y_train).cuda()
        else:
            inputs = Variable(train_imgs)
            target = Variable(y_train)
    
        # forward
        out = model(inputs)
        loss = criterion(out, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (epoch+1) % 20 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'
                .format(epoch+1, num_epochs, loss.item()))

    # testing DNN
    model.eval()
    if torch.cuda.is_available():
        test_imgs = Variable(test_imgs).cuda()
    results = model(Variable(test_imgs))

    # display the results
    correct = 0
    for sample, result in zip(testing_samples, results):
        sample.pred = 1 if result > 0.5 else 0
        if sample.label == 1:
            if sample.pred == 1:
                print(sample.fname, 'with label \"AI art\" is predicted as \"AI art\"')
                correct += 1
            else:
                print(sample.fname,'with label \"AI art\" is predicted as \"Real Picture\"')
        else:
            if sample.pred == 1:
                print(sample.fname, 'with label \"Real Picture\" is predicted as \"AI art\"')
            else:
                print(sample.fname,'with label \"Real Picture\" is predicted as \"Real Picture\"')
                correct += 1
    return correct / len(results)

# display the results (target images with the predicted labels and ground truth label)
print("Accuracy of Deep Neural Network classifier: ", "{:.0%}".format(deepNNClassification(samples['train'], samples['val'])))
print("Accuracy of Deep Neural Network classifier for real picture: ", "{:.0%}".format(deepNNClassification(samples['all'], realpic_sample)))
print("Accuracy of Deep Neural Network classifier for AI art: ", "{:.0%}".format(deepNNClassification(samples['all'], aiart_sample)))





# %% [markdown]
# # **Pretrained DCNN - VGG**

# %%
# load a pretrained DCNN (e.g., VGG)
class VGGFeature(nn.Module):
    def __init__(self, pretrained=True, layer=28):
        super().__init__()
        self.net = models.vgg16(pretrained).features.eval()
        self.layer = layer
        self.requires_grad_(False)

    def forward(self, x):
        for idx, layer in enumerate(self.net):
            x = layer(x)
            if idx == self.layer:
                return x

force_cpu = False

if torch.cuda.is_available() and not force_cpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('We are using device', device)

VGG = VGGFeature().to(device)

# %% [markdown]
# feed image into network and take feature maps out

# %%
# extract the feature vectors using pretrained DCNN
for sample in samples['all']:
    img = np.ascontiguousarray(sample.img.transpose(2, 0, 1)) # HWC -> CHW
    img = torch.tensor(img, dtype=torch.float32, device=device)[None] # np.array -> torch.tensor & CHW -> NCHW
    sample.feat = VGG(img)

# %%
# prepare and display the ranked list by showing in each row the query and the most relevant images (with similarities indicated)
cos = nn.CosineSimilarity(dim=1)
# np.array(image)
all_feats = torch.cat([sample.feat.view(1, -1) for sample in samples['val']], dim=0)
for idx, sample in enumerate(samples['val']):
    dists = cos(sample.feat.view(1, -1).expand_as(all_feats), all_feats)
    simlarity, orders = torch.sort(dists, descending=True)
    
    plt.figure(figsize=(18,4))
    plt.subplot(1, 6, 1)
    plt.title(f'query: {sample.fname}')
    plt.imshow(sample.img)
    
    for i, order in enumerate(orders[1:6]):
        plt.subplot(1, 6, i+2)
        result = samples['val'][order]
        plt.title(f'{result.fname} - %.3f' % simlarity[i+1])
        plt.imshow(result.img)

# %% [markdown]
# visualising by t-SNE

# %%
all_feats = [np.array(sample.feat.cpu().view(1, -1)[0]).flatten() for sample in samples['val']]
labels = [sample.label for sample in samples['val']]

all_feats = StandardScaler().fit_transform(all_feats) 
tsne = TSNE(perplexity=3)
X_embedded = tsne.fit_transform(np.array(all_feats))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)

# %% [markdown]
# summarize

# %%
# use maxpooling to prepare the feature vectors
# redo the retrieval and display the results
for idx, sample in enumerate(samples['all']):
    sample.feat_vec = sample.feat.max(dim=3)[0].max(dim=2)[0]
    # sample.feat_vec = sample.feat.mean(dim=(2,3))

all_feats = torch.cat([sample.feat_vec.view(1, -1) for sample in samples['val']], dim=0)
for idx, sample in enumerate(samples['val']):
    dists = cos(sample.feat_vec.view(1, -1).expand_as(all_feats), all_feats)
    simlarity, orders = torch.sort(dists, descending=True)
    
    plt.figure(figsize=(18,4))
    plt.subplot(1, 6, 1)
    plt.title(f'query: {sample.fname}')
    plt.imshow(sample.img)
    
    for i, order in enumerate(orders[1:6]):
        plt.subplot(1, 6, i+2)
        result = samples['val'][order]
        plt.title(f'{result.fname} - %.3f'%simlarity[i+1])
        plt.imshow(result.img)

# %% [markdown]
# check t-SNE

# %%
# check with t-NSE again

all_feats = [np.array(sample.feat_vec.cpu()[0]) for sample in samples['val']]
labels = [sample.label for sample in samples['val']]

tsne = TSNE(perplexity=3)
X_embedded = tsne.fit_transform(np.array(all_feats))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)

# %% [markdown]
# # **Fine-tuned method**

# %%
# build the networks
class SiameseNet(nn.Module):
    def __init__(self, in_features=512, mid_features=256, out_features=128):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ('Input', nn.Linear(in_features, mid_features)),
            ('Act', nn.Sigmoid()),
            ('Output', nn.Linear(mid_features, out_features)),
        ]))
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        feat_x = self.net(x)
        feat_y = self.net(y)
        return self.cos_sim(feat_x, feat_y)

Net = SiameseNet().to(device)
optimizer = optim.Adam(Net.parameters(), lr=1e-3, betas=(0.9, 0.999))
criterion = nn.MSELoss()
num_iters = 100
batch_size = 32
n_train = len(samples['train'])
train_inputs = torch.cat([sample.feat_vec for sample in samples['train']], dim=0).to(device)
train_labels = torch.tensor([sample.label for sample in samples['train']]).to(device)

# print(train_inputs.shape, train_labels.shape)
# training
for it in range(num_iters):
    idx_x = torch.randint(n_train, size=(batch_size,), device=device)
    idx_y = torch.randint(n_train, size=(batch_size,), device=device)
    input_x = train_inputs[idx_x]
    input_y = train_inputs[idx_y]
    target = (train_labels[idx_x] == train_labels[idx_y]).to(torch.float32) * 2 - 1
    output = Net(input_x, input_y)
    # print(output, target)
    loss = criterion(output, target)
    if it % 10 == 0:
        print(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
# show image with feature
all_feats = torch.cat([sample.feat_vec.view(1, -1) for sample in samples['val']], dim=0)
Net.eval()

for idx, sample in enumerate(samples['val']):
    dists = Net(sample.feat_vec.view(1, -1).expand_as(all_feats), all_feats)
    simlarity, orders = torch.sort(dists, descending=True)
    
    plt.figure(figsize=(18,4))
    plt.subplot(1, 6, 1)
    plt.title(f'query: {sample.fname}')
    plt.imshow(sample.img)
    
    for i, order in enumerate(orders[1:6]):
        plt.subplot(1, 6, i+2)
        result = samples['val'][order]
        plt.title(f'{result.fname} - %.3f'%simlarity[i+1])
        plt.imshow(result.img)

# %%
torch.cuda.empty_cache()


