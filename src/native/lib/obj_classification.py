
import os
import sys
import json
import cv2
import numpy as np
from io import open
from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.models import SqueezeNet

class ImgClassificationSqueezeNet:

    def __init__(self, label_file_path, weights_file_path):
        #self.model = squeezenet1_1(pretrained=True)
        if type(weights_file_path) != type('str'):
            weights_file_path = weights_file_path.decode("utf-8")
        if type(label_file_path) != type('str'):
            label_file_path = label_file_path.decode("utf-8")
        is_existed = os.path.exists(weights_file_path)
        if not is_existed:
            print("Cannot find weights file...")
            return -1
        print(label_file_path)
        self.model = SqueezeNet(version=1.1)
        self.model.load_state_dict(torch.load(weights_file_path))
        self.model.eval()
        self.transformation = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        is_existed = os.path.exists(label_file_path)
        if not is_existed:
            print("Error: no labels file found ...");
            return -1;

        labels = json.load(open(label_file_path))
        self.class_map = {int(key):value for (key, value) in labels.items()}
        print('In python: initialize successfully ...')

    def predict(self, image):
        print("In python: predict function ...")
        model = self.model
        transformation = self.transformation
        class_map = self.class_map
        # Preprocess
        image_tensor = transformation(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        
        if torch.cuda.is_available():
            print("Using GPU ...")
            image_tensor.cuda()

        # Turn the input into a Variable
        input = Variable(image_tensor)
        
        # Predict the class of the image
        output = model(input)
        index = output.data.numpy().argmax()
        prediction = class_map[index]
        print("In python: predict done!")
        return prediction

    def run(self, img_data):
        cv2_img = cv2.cvtColor(img_data[0],cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img)
        res = self.predict(pil_img)
        return res

def test(image_path):
    #image = Image.open(image_path)
    src_img = cv2.imread(image_path)
    #cv2_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
    #pil_img = Image.fromarray(cv2_img)
    label_file_path = 'labels.json'
    weights_file_path = 'squeezenet1_1-f364aa15.pth'
    classifier = ImgClassificationSqueezeNet(label_file_path, weights_file_path)
    #res = classifier.predict(pil_img)
    res = classifier.run(src_img)

    print('Class:', res)

if __name__=='__main__':
    path = sys.argv[1]
    test(path)
