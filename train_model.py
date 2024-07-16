from data_loaders import CreateDataLoaders
from neural_network import NeuralNetwork

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os as os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score



def train_model(no_epochs):

    if torch.cuda.is_available():
        device = torch.device('cuda')
        #print("Using CUDA")
    else:
        device = torch.device('cpu')
        #print("Using CPU")

    batch_size = 32
    model = NeuralNetwork().to(device)
    data_loaders = CreateDataLoaders(batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss_function = nn.BCELoss().to(device) # Alternative is nn.MSELoss()

    # nn.BCELoss needs to receive the outputs of Sigmoid activation as its input, meaning: [0,1] range.
    # nn.BCEWithLogitsLoss will need the logits as inputs instead of outputs of Sigmoid, since it will apply sigmoid internally.
    # nn.CrossEntropyLoss needs at the minimum two outputs (classes) per sample so if you want to use this for binary classification
    # then you need to change the neural network so that it has two output nodes (one for label 0, one for label 1, each with a probability(sigmoid)).
    # nn.CrossEntropyLoss: 
    # predictions: tensor([[0.2228, 0.2523, 0.9712, 0.7887, 0.2820], [0.7778, 0.4144, 0.8693, 0.1355, 0.3706], [0.0823, 0.5392, 0.0542, 0.0153, 0.8475]])
    # == 2-D tensor
    # 0.9712 matches class 2, but for the second label 0 has a lower prob than 2 so not good, then for 4 0.84 matches 4 indeed with the highest prob.
    # labels: tensor([2, 0, 4]) == 1-D tensor
    # Cross Entropy Loss:
    # tensor(1.2340) == 0-D (number) tensor

    losses = []

    stopped_early = False

    for epoch in range(no_epochs):

        model.train() 

        #for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
        for batch in data_loaders.train_loader:
	
            inputs = batch["input"].to(device) 
            labels = batch["label"].to(device)
            outputs = model(inputs) # outputs is a tensor with size torch.Size([5, 1]) like tensor([[0.], [0.], [0.], [0.], [0.]])

            labels = labels.unsqueeze(1) # torch.Size([5]) like tensor([0., 0., 0., 0., 0.]) to torch.Size([5, 1]) lie tensor([[0.], [0.], [0.], [0.], [0.]])

            loss = loss_function(outputs, labels) # loss is single number tensor

            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model=model, data_loader=data_loaders.valid_loader, loss_function=loss_function, device=device)

        if epoch > 0:
            if result > losses[-1]:
                previous_epoch = epoch - 1
                print(f"Stopping early - epoch {epoch} gave higher loss. Loading model from epoch {previous_epoch}.")
                file = "./saved/" + f"saved_model_{previous_epoch}.pkl"
                model.load_state_dict(torch.load(file))
                stopped_early = True
                break
            
            else:
                losses.append(result)

                file = "./saved/" + f"saved_model_{epoch}.pkl"
                torch.save(model.state_dict(), file, _use_new_zipfile_serialization=False)
        else:
            losses.append(result)

            os.mkdir("./saved/")
            file = "./saved/" + f"saved_model_{epoch}.pkl"
            torch.save(model.state_dict(), file, _use_new_zipfile_serialization=False)

    if not stopped_early:
        print("Trained using all epochs - no early stopping.")
    
    #print(losses)
    #plt.plot(losses)
    #plt.title('Eval loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.show()

    test(model=model, data_loader=data_loaders.test_loader, device=device)

def evaluate(model, data_loader, loss_function, device):

    model.eval() 

    losses = []

    with torch.no_grad():

        for batch in data_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs) 

            labels = labels.unsqueeze(1) 

            loss = loss_function(outputs, labels) 

            losses.append(loss.detach().numpy()) # Store the losses but not in GPU RAM.

    np_losses = np.asarray(losses) 
    np_losses_mean = np_losses.mean()
    np_losses_mean_asfloat32 = np_losses_mean.item()
    return np_losses_mean_asfloat32          # The mean loss over all the batches in the epoch.

def test(model, data_loader, device):

    model.eval()
    
    with torch.no_grad():

        labels_list = []
        predictions_list = []

        for batch in data_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)

            labels_list.extend(labels.detach().tolist())
            predictions_list.extend(outputs.detach().squeeze(-1).round().tolist())

        precision = precision_score(labels_list, predictions_list)
        recall = recall_score(labels_list, predictions_list)
        accuracy = accuracy_score(labels_list, predictions_list)
        f1 = f1_score(labels_list, predictions_list)
        tn, fp, fn, tp = confusion_matrix(labels_list, predictions_list).ravel()
        print("Actual")
        print("  ", " 1", " 0")
        print(" 1", tp, fp)
        print(" 0", fn, tn)
        print("Accuracy: ",accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 :", f1)

if __name__ == '__main__':
    train_model(no_epochs=10)


