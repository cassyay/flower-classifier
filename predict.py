import torchvision 
from torchvision import transforms, datasets
import torch 
from torch.utils.data import DataLoader
import torchvision.models as models
from collections import OrderedDict
from torch import nn, optim
import numpy as np
from PIL import Image
import json
import train
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type=str, default='flowers/test/17/image_03830.jpg', help='predicted classification for this image')
    parser.add_argument('--gpu', type=bool, action='store', default=True, help='True turns on GPU')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='trained model')
    parser.add_argument('--topk', type=int, default=3, help='list the number of top predictions')
    parser.add_argument('--category_to_name', type=str, default='cat_to_name.json', help='json file containing category to label') 
    
    args = parser.parse_args()
    return args

def process_image(image):
   
    load_data = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()
                                   ])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    load_data = transform(load_data).float()
    np_image = np.array(load_data)
    np_image = (np.ndarray.transpose(np_image, (1, 2, 0)) - mean)/std     
    np_image = np.ndarray.transpose(np_image, (2, 0, 1))               
    
    
    return np_image
        
def predict(np_image, model, gpu, topk):  

    if gpu == True:
        model = model.to('cuda')  
    
    img_torch = process_image(np_image)  
    img_torch = torch.from_numpy(img_torch)
    
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if gpu == True:
        with torch.no_grad():
            output = model.forward(img_torch.cuda()) 

    transform = F.softmax(output.data,dim=1) 

    return transform.topk(topk)

    probs, classes = predict(np_image, model, topk)
    return probs, classes

def sanity_check(image): 
    index = int(path.split('/')[1])
    imagepath = test_dir + path
    image = process_image(imagepath)

    plot = imshow(image, ax = plt)
    plot.axis('off')
    plot.title(cat_to_name[str(index)])
    plot.show()

    axes = predict(imagepath, model)

    yaxis = [cat_to_name[str(i)] for i in np.array(axes[1][0].cpu())]
    y_pos = np.arange(len(yaxis))
    xaxis = np.array(axes[0][0].cpu().numpy())   

    plt.barh(y_pos, xaxis)
    plt.xlabel('probability')
    plt.yticks(y_pos, yaxis)
    plt.title('probability of flower classification')

    plt.show()

def main():
        args = parse_args()
        
        process = process_image(args.image)
        
        prediction = predict(np_image, model, args.gpu, args.topk)
        
        check = sanity_check(args.image)
        
        
if __name__ == '__main__':
        main()
