import json
import torchvision 
from torchvision import transforms, datasets
import torch 
from torch.utils.data import DataLoader
import torchvision.models as models
from collections import OrderedDict
from torch import nn, optim
import numpy as np
import argparse
import os
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plot 

#check if GPU is available 

check_gpu = torch.cuda.is_available()

if not check_gpu:
    print('Running on CPU, switch to GPU if possible ...')
else:
    print('Training on GPU ...')

#create args 

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers', help='directory containing dataset')
    parser.add_argument('--gpu', type=bool, default='True', action='store', help='True to use gpu')
    parser.add_argument('--arch', type=str, default='alexnet', help='model architecture')
    parser.add_argument('--epochs', type=int, default='20', dest='epochs', help='number of epochs')
    parser.add_argument('--hidden_units', type=int, default=[1024], help='train model')
    parser.add_argument('--learning_rate', type=float, dest='lr', default='0.01')
    parser.add_argument('--save_model', type=str, action='store', default='checkpoint.pth', help='save the trained model')
    
    args = parser.parse_args()
    return args
 
#transform data 

def transform(train_dir, valid_dir, test_dir):
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    training_transform = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    return training_transform, validation_transform, testing_transform

#Load datasets with ImageFolder

def train(training_transform, validation_transform, testing_transform):
    
    train_data = datasets.ImageFolder(train_dir, transform=training_transform)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transform)
    testing_data = datasets.ImageFolder(test_dir, transform = testing_transform)
    
    return train_data, validation_data, testing_data

#Define dataloaders 

def load_data(train_data, validation_data, testing_data):    
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True) 
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 32)
    testing_loader = torch.utils.data.DataLoader(testing_data, batch_size = 32)
    
    return training_loader, validation_loader, testing_loader
   

def pretrained_model(arch, gpu):
    
    
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        print('loading AlexNet ... ')
    else:
        print('AlexNet is the only way')
        model = models.alexnet(pretrained=True)
     
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    print(model)
    
def neural_network(model, gpu, hidden_units):
    

    if hidden_units == None:
        hidden_units = [1024, 256]
    if input_size == None:
        input_size = 9216
    output_size = 102
    
    if 'gpu' == True:
        model = model.to('cuda')
    else:
        print('turn on gpu')
    
    model.classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_units[0])),
                      ('relu1', nn.ReLU()),
                      ('dropout1', nn.Dropout(p=0.1)),
                      ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                      ('relu2', nn.ReLU()),
                      ('dropout2', nn.Dropout(p=0.1)),
                      ('fc3', nn.Linear(hidden_units[1], output_size)),
                      ('output', nn.LogSoftmax(dim=1))]))

    classifier = model.classifier
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.paramters(), lr=0.01)
    model.cuda()
    
    return model 
    
    
                          
    

def train_model(epochs, training_loader, validation_loader, gpu, model, optimizer, criterion):
    if epochs == None:
        epochs = 10
        print('iterating through 10 epochs')
    
    steps = 0
    print_every = 5
    train_losses, validation_losses = [], []
    
    for e in range(epochs, gpu):
        running_loss = 0
        for images, labels in training_loader: 
            steps += 1
            if gpu == 'True':
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = images.to('cpu'), labels.to('cpu')
        

            optimizer.zero_grad()

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

            with torch.no_grad:
                for images, labels in validation_loader:
                    if gpu == 'True':
                        images, labels = images.to('cuda'), labels.to('cuda')
                    else:
                        images, labels = images.to('cpu'), labels.to('cpu')

                    log_ps = model.forward(images)
                    batch_loss = criterion(log_ps, labels)
                    validation_loss += batch_loss.item()
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

    
            train_losses.append(running_loss/len(training_loader)),
            validation_losses.append(validation_loss/len(validation_loader))  

    
            print("Epoch: {}/{}..".format(e+1, epochs),
                        "Training loss: {:.3f}..".format(running_loss/len(training_loader)),
                        "Validation loss: {:.3f}..".format(validation_loss/len(validation_loader)),
                        "Test accuracy: {:.3f}".format(accuracy/len(validation_loader)))
            running_loss = 0
            model.train() 
        
    return model

                        
def test_model(model, testing_loader, gpu, criterion):
    test_loss = 0
    accuracy = 0
    model.to('cuda')

    
    with torch.no_grad():
        for images, labels in testing_loader:
            if gpu == 'True':
                images, labels = images.to('cuda'), labels.to('cuda')
            log_ps = model.forward(images) 
            batch_loss = criterion(log_ps, labels) 
            test_loss += batch_loss.item()   
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))


        print("Test Loss: {:.3f}.. ".format(test_loss/len(testing_loader)),
        "Test Accuracy: {:.3f}".format(accuracy/len(testing_loader)))


def save(model, train_datasets, save_model):

    model.class_to_idx = train_data.class_to_idx  
    checkpoint = {'input_size': input_size,
            'hidden_layer1': hidden_units[0],
            'hidden_layer2': hidden_units[1],
            'output_size': output_size,
            'epochs': epochs,
            'arch': 'alexnet',
            'classifier': classifier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'class_to_idx': train_data.class_to_idx} 

    return torch.save(checkpoint, class_to_idx, save_model)

def load_checkpoint(cat_to_name, arch):
    checkpoint = torch.load(cat_to_name)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print('alexnet is the only way')
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_index = checkpoint['class_to_idx']
    model.classifier.epochs = checkpoint['epochs']
        
    return model

    model = load_checkpoint('checkpoint.pth')
    print(model)
    


def main():
    args = parse_args()
    
    cuda = torch.device('cuda')
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    
    model = pretrained_model(args.arch, args.gpu)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model = neural_network(model, args.gpu, args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr = args.lr)
    
    training_model = train_model(args.epochs, training_loader, validation_loader, args.gpu, model, optimizer, criterion)
    
    testing_model = test_model(training_model, testing_loader, args.gpu, criterion)  
    
    save_model = save(training_model, train_data, args.save_model)
    
    load_model = load_checkpoint(model)
    
    print('model is trained!')
                     
                     
if __name__ == '__main__':
    main()
    