import torch
from torchvision import transforms, datasets

def defineDataTransform(flag, transform = None):
    """
    Defines a series of transformations to apply to all images in the dataset.
    
    Keyword Arguments:
    flag: a string which is either "train" or "test." Applies transforms based on input.
    transforms: an array of sequential transform operations. If not supplied, default transforms for
    train and test will be used.
    
    Returns a pipeline defined by transforms.Compose(transforms).
    """
    
    if transform is not None:
        return transforms.Compose(transform)
    
    if flag == "train":
        return transforms.Compose([transforms.RandomRotation(30), 
                                   transforms.RandomResizedCrop(224), 
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])
    elif flag == "test":
        return transforms.Compose([transforms.RandomRotation(30), 
                                   transforms.RandomResizedCrop(224), 
                                   transforms.ToTensor()])
    else:
        raise Exception("Unknown flag input. Can only be train or test.")
    

def createDataLoader(path, transform, batch_size):
    """
    Creates a dataloader for the train, validation, or test data.
    
    Keyword Arguments:
    path: string representing the path from current directory to root folder of data.
    transform: a transforms.Compose() to apply to all data in the loader.
    batch_size: an int representing the batch size for the machine learning model.
    """
    
    data = datasets.ImageFolder(path, transform = transform)
    
    return torch.utils.data.DataLoader(data, batch_size = batch_size)