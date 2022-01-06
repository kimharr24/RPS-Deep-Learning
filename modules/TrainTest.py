import torch.nn as nn

def trainNetwork(model, dataloader, n_epochs, optimizer, criterion = nn.CrossEntropyLoss()):
    """
    Runs a training loop on the provided model.
    
    Keyword Arguments:
    model: an instance of the user-defined neural network.
    dataloader: PyTorch DataLoader to iterate through batches.
    n_epochs: number of epochs to run the training loop.
    optimizer: optimizer used for gradient descent.
    criterion: loss function used to compute gradients.
    """
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            features, labels = batch
            
            optimizer.zero_grad()
            preds = model(features.float())
            loss = criterion(preds, labels)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Epoch {epoch + 1}, Loss: {total_loss}, Final Batch Loss: {loss.item()}')