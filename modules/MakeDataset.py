import cv2
from PIL import Image
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation

def arrayToImg(arr, path, file_name):
    """
    Converts a numpy array to an image and saves it at the provided path.
    
    Keyword Arguments:
    arr: the numpy array to be converted.
    path: the file path to save the image.
    file_name: name of the image.
    """
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) #Conversion from OpenCV's default BGR to RGB
    img = Image.fromarray(arr)
    img.save(f'{path}{file_name}.png')
    
def rescaleImg(img, dim = (300, 300)):
    """
    Rescales the OpenCV image to be 300 x 300 pixels.

    Keyword Arguments
    img: the image to be transformed.
    dim: the (m x n) pixel dimensions to be converted to.
    """
    return cv2.resize(img, dim)

def removeBackground(img):
    
    segmentor = SelfiSegmentation()
    img = segmentor.removeBG(img, (255, 255, 255), threshold = 0.55)
    
    return img

def createDirectories():
    """
    Creates the necessary folders for storing rock, paper, and scissors examples.
    """
    
    directories = [
        "data",
        "data/rps-train",
        "data/rps-test",
        "data/rps-train/rock",
        "data/rps-train/paper",
        "data/rps-train/scissors",
        "data/rps-test/rock",
        "data/rps-test/paper",
        "data/rps-test/scissors"
    ]
    
    try:
        for k in range(len(directories)):
            os.makedirs(directories[k])
    except:
        raise Exception("Training folders already exist.")

def recordExamples(record_type, path, n_examples):
    """
    Given the path to the appropriate directory, saves webcam example images.

    Keyword Arguments:
    record_type: a string that can be one of "rock" "paper" or "scissors."
    path: a string representing the path to which the examples images are saved.
    n_examples: an int representing the number of examples to record.
    """

    camera = cv2.VideoCapture(0)
    coordinates, fontScale, thickness, font, color = (5, 18), 0.45, 2, cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)
    all_matrices = []
    
    result = input(f"Type 'y' to begin recording {record_type} examples: ")
    
    if result == "y":
        count = 0
        while count < n_examples:
            success, img = camera.read()
            img = rescaleImg(img)
            all_matrices.append(img)
            
            img = cv2.putText(img, f"n_examples: {count + 1}", coordinates, font, fontScale, color, thickness)
            cv2.imshow("Recording Window", img)
            
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        camera.release()
        cv2.destroyAllWindows()
        
        print("Saving images...")
        for idx, image in enumerate(all_matrices):
            image = removeBackground(image)
            arrayToImg(image, path, str(idx))
        
def createDataSet(flag, n_imgs_per_class = {"train": 500, "test": 150}):
    """
    Automatically creates a dataset of images per class, using the user's
    images as examples.
    
    Keyword Arguments
    flag: a string that can be one of "train" or "test" depending on which dataset is being generated.
    n_imgs_per_class: hashmap that can be used to customize train-test split if specified.
    """
    
    try:
        n_examples = n_imgs_per_class[flag]
        paths = {
            "rock": f"data/rps-{flag}/rock/",
            "paper": f"data/rps-{flag}/paper/",
            "scissors": f"data/rps-{flag}/scissors/"
        }
    except:
        raise Exception("Flag can only be one of \"train\" or \"test.\"")
        
    if flag == "train":
        createDirectories()
    
    for key, value in paths.items():
        recordExamples(key, value, n_examples)