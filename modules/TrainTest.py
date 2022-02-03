import torch.nn as nn
import torch
from sklearn import metrics

def validationLoss(model, val_loader, criterion):
    validation_loss = 0
    model.eval()
    
    for batch in val_loader:
        features, labels = batch
        preds = model(features)
        loss = criterion(preds, labels)
        
        validation_loss += loss.item()
    
    return validation_loss

def trainNetwork(model, train_loader, val_loader, n_epochs, optimizer, criterion = nn.CrossEntropyLoss()):
    """
    Runs a training loop on the provided model.
    
    Keyword Arguments:
    model: an instance of the user-defined neural network.
    train_loader: PyTorch DataLoader to iterate through training batches.
    val_loader: PyTorch DataLoader to conduct validation loss
    n_epochs: number of epochs to run the training loop.
    optimizer: optimizer used for gradient descent.
    criterion: loss function used to compute gradients.
    """
    
    for epoch in range(n_epochs):
        total_loss = 0
        model.train() # Setting the model to train mode because validationLoss sets to eval mode
        
        for batch in train_loader:
            features, labels = batch
            
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        
        epoch_val_loss = validationLoss(model, val_loader, criterion)
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss:.3f}, Validation Loss {epoch_val_loss:.3f}')
        
def evaluateNetwork(model, dataloader):
    """
    Prints precision, accuracy, and recall of a model's evaluation.
    
    Keyword Arguments:
    model: an instance of the neural network to be evaluated.
    dataloader: PyTorch DataLoader that contains the test data.
    """
    
    softmax = nn.Softmax(dim = 1)
    model.eval()
    
    for batch in dataloader:
        features, labels = batch
        _, preds = torch.max(softmax(model(features)), dim = 1)
        print(metrics.confusion_matrix(labels, preds))
        print(metrics.classification_report(labels, preds, digits = 5))
        
def loadModel(model, model_name):
    """
    Loads a saved model in the saved_models directory.
    
    Keyword Arguments:
    model: an instance of the neural network to be loaded.
    model_name: a string representing the name of the saved model to be loaded.
    """
    
    model.load_state_dict(torch.load(f'saved_models/{model_name}'))