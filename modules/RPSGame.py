import cv2
from PIL import Image
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from MakeDataset import *
from DataLoaders import *
import torch
import torch.nn as nn

def findKeyFromValue(hashMap, value):
    """
    Retrieves the key from a hash map, assuming all values are also unique.
    
    Keyword Arguments:
    hashMap: the hash map to retrieve the key from.
    value: the value to find in the hash map.
    
    Returns the key corresponding to the value.
    """
    
    key_list = list(hashMap.keys())
    val_list = list(hashMap.values())
    
    position = val_list.index(value)
    
    return key_list[position]
    
def conformImgToModel(img):
    """
    Converts (300 x 300 x 3) array to CNN expectation of (1 x 3 x 300 x 300) input tensor
    
    Keyword Arguments:
    img: The image to be transformed
    
    Returns a (1 x 3 x 300 x 300) image tensor.
    """
    
    # Fix the color channel from BGR to RGB since the model was trained on RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Remove the background since the model was trained on images with backgrounds removed
    img = removeBackground(img)
    
    # Model expects tensor input, convert from numpy to tensor
    img = torch.tensor(img) 
    
    # Tensor is (300 x 300 x 3) because of the numpy array, switch to (3 x 300 x 300)
    img = img.permute(2, 0, 1)
    
    # Model expects a batch size dimension, simply add a dummy dimension
    img = img.unsqueeze(0)
    
    return img

def makePrediction(img, model):
    """
    Trained CNN makes a prediction on the current image.
    
    Keyword Arguments:
    img: a numpy array of the current image (300 x 300)
    model: the pre-trained PyTorch CNN created in network-trainer.ipynb
    
    Returns a string representing the model's class prediction.
    """
    
    softmax = nn.Softmax(dim = 1)
    img = conformImgToModel(img)
    
    # Dummy transform to get label_mappings
    train_transform = defineDataTransform("train")
    label_mappings = createDataLoader("data/rps-train", train_transform, get_label_mappings = True)
    
    _, prediction_as_num = torch.max(softmax(model(img.float())), dim = 1)
       
    prediction = findKeyFromValue(label_mappings, prediction_as_num)
   
    return prediction
    
def executeGame(model):
    """
    Main function to play rock, paper, scissors.
    
    Keyword Arguments:
    model: the pre-trained PyTorch CNN created in network-trainer.ipynb
    """
    
    # Initialize the webcam object
    camera = cv2.VideoCapture(0)
    
    # Visual settings for webcam display
    coordinates, fontScale, thickness, font, color = (5, 18), 0.45, 2, cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)
    
    while True:
        success, img = camera.read()
        img = rescaleImg(img)
        
        #Pass the current frame into the model to retrieve a class prediction and probability
        prediction = makePrediction(img, model)
        
        img = cv2.putText(img, f"Prediction: {prediction}", coordinates, font, fontScale, color, thickness)
        cv2.imshow("Recording Window", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()