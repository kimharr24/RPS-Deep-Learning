# RPS-Neural-Network

## General Repository Information
The objective of this project is to develop a game of rock paper scissors. Users can execute the main run file `run.ipynb` which will automatically generate a custom image dataset, train a convolutional neural network (CNN) to distinguish between the user's rock, paper, scissors, and build a webcam-based game where users can compete against the computer in rock paper scissors. **Important:** At the top of the main run file, users should change the parameter in `sys.path.append()` to be the location of the **modules** directory in relation to their machine.

## Customizable Dataset 
To train the CNN to recognize the differences between the user's rock, paper, and scissors, the following code is executed in the main run file.
```
createDataSet("train")
createDataSet("test")
```
The terminal will ask to demonstrate rock, paper, and scissors, intelligently saving hundreds of example images and constructing nested directories stemming from the main git repository to store data.

## Hyperparameters and Training
Ranging from batch size to regularization and number of epochs to train, users can easily specify hyperparameters in the main run file. Afterwards, by running
```
model = CNN()
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = regularization)
trainNetwork(model, train_loader, n_epochs, optimizer)
```
the run file will automatically train a CNN with the provided hyperparameters and image dataset.

## Evaluating The Model
Once the model has finished training, users should run
```
model = CNN()
loadModel(model, f'{model_name}_{batch_size}BS_{learning_rate}LR_{n_epochs}E')
evaluateNetwork(model, test_loader)
```
to evaluate how effective their model is at distinguishing between rock, paper, and scissors. An average weighted F1 score close to 1.00 is desirable. After tuning the model, users can now use the CNN to play a game of rock paper scissors.