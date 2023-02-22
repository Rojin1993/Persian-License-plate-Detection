import copy
import time
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
import torchvision
import numpy as np
import random
import cv2 as cv
import imutils
import os

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# variables
train_b_s = 20
valid_b_s = 10
test_b_s = 5
num_epochs = 20

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constant for classes
classes = ('a','b','ch','d','ein','f', 'g', 'ghaf', 'ghein', 'h2','hj', 'j', 'k', 'kh','l','m','n','p','r','s','sad','sh','t','ta','th','v','y','z','za','zad','zal','zh')

# functions for Augmentation
def blur(image):
    x = random.randrange(1, 5, 2)
    blur = cv.GaussianBlur(image, (x, x), cv.BORDER_DEFAULT)
    return blur

def rotate_image(image, angle):
    rotated = imutils.rotate(image, angle)
    return rotated

def oversample_images(main_dataset):
    new_images = torch.Tensor(len(main_dataset)*3, 3, 90, 90)
    new_labels = []
    counter = 0
    for i, (img, target) in enumerate(main_dataset):
        new_images[counter, :, :, :] = img
        new_labels.append(target)
        counter += 1

        img = np.array(img)
        image = rotate_image(img, random.randint(-10, 10))
        image = torch.from_numpy(image)
        new_images[counter, :, :, :] = image
        new_labels.append(target)
        counter += 1

        image = blur(img)
        image = torch.from_numpy(image)
        new_images[counter, :, :, :] = image
        new_labels.append(target)
        counter += 1

    new_labels1 = torch.Tensor(new_labels)
    new_labels1 = new_labels1.int()

    return new_images, new_labels1


# function of training images
def train_epoch(model, de, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    out_train = []
    labels_train = []
    model.train()
    m = nn.Softmax(dim=1)
    # m = nn.LogSoftmax(dim=1)
    m = m.to(de)
    for images, labels in dataloader:
        images = images.to(de)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(de)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output = model(images)
            loss = loss_fn(m(output), labels)
            loss.backward()
            optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)

        for i in predictions:
            out_train.append(i)
        for j in labels:
            labels_train.append(j)
        train_correct += (predictions == labels).sum().item()

    # Build confusion matrix
    labels_train1 = torch.tensor(labels_train, device='cpu')
    out_train1 = torch.tensor(out_train, device='cpu')
    cf_matrix = confusion_matrix(labels_train1, out_train1)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                             columns=[i for i in classes])
    print(df_cm)

    return train_loss, train_correct

# function of implementing trained model on validation data
def valid_epoch(model, de, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    out_valid = []
    labels_valid = []
    model.eval()
    m = nn.Softmax(dim=1)
    # m = nn.LogSoftmax(dim=1)
    m = m.to(de)
    for images, labels in dataloader:
        labels = labels.type(torch.LongTensor)
        images, labels = images.to(de), labels.to(de)

        with torch.set_grad_enabled(False):
            output = model(images)
            loss = loss_fn(m(output), labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)

        for i in predictions:
            out_valid.append(i)
            # n += 1
        for j in labels:
            labels_valid.append(j)
        val_correct += (predictions == labels).sum().item()

    # Build confusion matrix
    labels_valid1 = torch.tensor(labels_valid, device='cpu')
    out_valid1 = torch.tensor(out_valid, device='cpu')
    cf_matrix = confusion_matrix(labels_valid1, out_valid1)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    print(df_cm)

    return valid_loss, val_correct

# function of implementing chosen model on test data
def test_epoch(model, de, dataloader, loss_fn):
    test_loss, test_correct = 0.0, 0
    out_test = []
    labels_test = []
    model.eval()
    m = nn.Softmax(dim=1)
    # m = nn.LogSoftmax(dim=1)
    m = m.to(de)
    for images, labels in dataloader:
        labels = labels.type(torch.LongTensor)
        images, labels = images.to(de), labels.to(de)

        with torch.set_grad_enabled(False):
            output = model(images)
            loss = loss_fn(m(output), labels)
        test_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)

        for i in predictions:
            out_test.append(i)
        for j in labels:
            labels_test.append(j)
        test_correct += (predictions == labels).sum().item()

    # Build confusion matrix
    labels_test1 = torch.tensor(labels_test, device='cpu')
    out_test1 = torch.tensor(out_test, device='cpu')
    cf_matrix = confusion_matrix(labels_test1, out_test1)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    print(df_cm)

    return test_loss, test_correct

# definition of train, test and validation dataset
train_horoof = torchvision.datasets.ImageFolder(root='C:/Users/rozhi/Downloads/Compressed/train-horoof/train',
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Resize([90, 90]),
                                                       torchvision.transforms.Normalize(mean=0, std=1)
                                                   ])
                                                   )
Data1, targets1 = oversample_images(train_horoof)
train_horoof1 = TensorDataset(Data1, targets1)

test_horoof = torchvision.datasets.ImageFolder(root='C:/Users/rozhi/Downloads/Compressed/test-horoof/test',
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Resize([90, 90]),
                                                       torchvision.transforms.Normalize(mean=0, std=1)
                                                   ])
                                                   )
Data2, targets2 = oversample_images(test_horoof)
test_horoof1 = TensorDataset(Data2, targets2)

valid_horoof = torchvision.datasets.ImageFolder(root='C:/Users/rozhi/Downloads/Compressed/valid-horoof/valid',
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Resize([90, 90]),
                                                       torchvision.transforms.Normalize(mean=0, std=1)
                                                   ])
                                                   )
Data3, targets3 = oversample_images(valid_horoof)
valid_horoof1 = TensorDataset(Data3, targets3)

# definition of model and make changes on it's classifier
model = torchvision.models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 32)
model = model.to(d)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


train_loss, train_correct = 0.0, 0
valid_loss, valid_correct = 0.0, 0
test_loss, test_correct = 0.0, 0

best_model_wts = copy.deepcopy(model.state_dict())
since = time.time()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

Train_size = len(train_horoof1)
Valid_size = len(valid_horoof1)
Test_size = len(test_horoof1)

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_horoof1, batch_size=train_b_s, shuffle=True,
                                               generator=torch.Generator().manual_seed(0), num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=valid_horoof1, batch_size=valid_b_s, shuffle= True,
                                               generator=torch.Generator().manual_seed(0), num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_horoof1, batch_size=test_b_s,
                                               generator=torch.Generator().manual_seed(0), num_workers=0)


best_acc = 0.0
best_epoch = 0

# training for number of num_epochs
for epoch in range(num_epochs):
    print('confusion matrix of train data in epoch {}:'.format(epoch + 1))
    train_loss, train_correct = train_epoch(model, d, train_loader, criterion, optimizer)
    print('confusion matrix of valid data in epoch {}:'.format(epoch + 1))
    valid_loss, valid_correct = valid_epoch(model, d, valid_loader, criterion)
    train_loss = train_loss / Train_size
    train_acc = train_correct / Train_size * 100
    valid_loss = valid_loss / Valid_size
    valid_acc = valid_correct / Valid_size * 100
    if valid_acc >= best_acc:
        best_epoch = epoch
        best_acc = valid_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print("Epoch:{}/{} : Training Loss:{:.3f}, Valid Loss:{:.3f}   Training Acc {:.2f} %, Valid Acc {:.2f} % ".format(
            epoch + 1,
            num_epochs,
            train_loss,
            valid_loss,
            train_acc,
            valid_acc))
    print('==' * 20)

    print(
        'best accuracy of valid dataset in epoch{}: {}'.format(best_epoch + 1, best_acc))
    print('*_' * 10)

# implementing model on test data with it's best weights
model.load_state_dict(best_model_wts)
test_loss, test_correct = test_epoch(model, d, test_loader, criterion)
test_loss = test_loss / Test_size
test_acc = test_correct / Test_size * 100
print('in Test Dataset: Accuracy = {}, Loss ={}'.format(test_acc,test_loss))

time_elapsed = time.time() - since
print('Training and Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
PATH = 'R:/horoof'
torch.save(model, os.path.join(PATH, 'Augdensenet-characters90.pth'))

