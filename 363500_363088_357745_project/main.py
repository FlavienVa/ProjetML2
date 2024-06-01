from datetime import datetime
import matplotlib.pyplot as plt


import argparse

import numpy as np
from torchinfo import summary
import torch

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import os

def run_experiments(args):
    def train_and_evaluate(lr, max_iters):
        args.lr = lr
        args.max_iters = max_iters
        acc, f1 = main(args)
        return acc, f1

    lrs = [1e-3, 1e-2, 1e-1]
    max_iters_list = [10, 20, 30]

    results = []
    for lr in lrs:
        for max_iters in max_iters_list:
            acc, f1 = train_and_evaluate(lr, max_iters)
            results.append((lr, max_iters, acc, f1))
            print(f"lr: {lr}, max_iters: {max_iters} - Validation Accuracy: {acc:.3f}, Validation F1 Score: {f1:.3f}")

    plot_results(results, lrs, max_iters_list)

def plot_results(results, lrs, max_iters_list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for lr in lrs:
        accs = [result[2] for result in results if result[0] == lr]
        f1s = [result[3] for result in results if result[0] == lr]
        axes[0].plot(max_iters_list, accs, label=f'lr={lr}')
        axes[1].plot(max_iters_list, f1s, label=f'lr={lr}')

    axes[0].set_xlabel('Max Iterations')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Validation Accuracy vs. Max Iterations')
    axes[0].legend()

    axes[1].set_xlabel('Max Iterations')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Score vs. Max Iterations')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data_path)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    print(xtrain.shape, xtest.shape, ytrain.shape)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    mean = np.mean(xtrain)
    std = np.std(xtrain)
    xtrain = normalize_fn(xtrain, means= mean, stds= std)
    xtest = normalize_fn(xtest, means= mean, stds=std)

    print(xtrain.shape, xtest.shape, ytrain.shape)


    # Make a validation set
    if not args.test:
        # Split the training data into training and validation sets
        random_index = np.arange(xtrain.shape[0])
        np.random.shuffle(random_index)
        xtrain, xvalid = xtrain[random_index[:int(0.8*xtrain.shape[0])]], xtrain[random_index[int(0.8*xtrain.shape[0]):]]
        ytrain, yvalid = ytrain[random_index[:int(0.8*ytrain.shape[0])]], ytrain[random_index[int(0.8*ytrain.shape[0]):]]
        
    mean = np.mean(xtrain, axis=0)
    std = np.std(xtrain, axis=0)
    print(xtrain.shape, xtest.shape, ytrain.shape, xvalid.shape)
    if not args.test:
        xvalid = normalize_fn(xvalid,means=mean,stds=std)
    
    ### WRITE YOUR CODE HERE
        #print("Using PCA")
    if args.device == "cuda":
        if (torch.cuda.is_available()):
            print("Device use: CUDA")
            device = torch.device('cuda')
        else:
            print("ERROR specified device unusable -> CPU")
            device = torch.device('cpu')
    elif args.device == "mps":
        if (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            print("Device use: MPS")
            device = torch.device('mps')
        else:
            print("ERROR device specified unusable -> CPU")
            device = torch.device('cpu')
    else: 
        print("Device use: CPU")
        device = torch.device('cpu')
    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        xvalid = pca_obj.reduce_dimension(xvalid)


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)
        xvalid = append_bias_term(xvalid)

        model = MLP(input_size= xtrain.shape[1] ,n_classes= 10, device=device) ### WRITE YOUR CODE HERE
        
    if args.nn_type == "cnn" :
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        xvalid = xvalid.reshape(-1, 1, 28, 28)

        model = CNN(input_channels=1 , n_classes=10)

    if args.nn_type == "transformer":
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        xvalid = xvalid.reshape(-1, 1, 28, 28)
        print(xtrain.shape)
        model = MyViT(xtrain.shape[1:], device=device)


    if args.load == True and args.path != None:
        model.load_state_dict(torch.load(args.path)) 
    

    summary(model)
    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=device)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
    print("preds_train" , np.shape(preds_train))

    # Predict on unseen data
    preds = method_obj.predict(xtest)
    print("preds" , np.shape(preds))
    predsvalid = method_obj.predict(xvalid)
    print("predsvalid", np.shape(predsvalid))

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(predsvalid, yvalid)
    macrof1 = macrof1_fn(predsvalid, yvalid)
    print(f"\nValidation set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    print(preds.shape)
    
    np.save("predictions", preds)

    if args.save == True:
        current_time = datetime.now().isoformat(timespec="minutes")
        torch.save(model.state_dict(), f"trained_model/{model.__class__.__name__}-{macrof1:.5f}-{current_time}")
    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    ## acc = accuracy_fn(preds, xtest)
    ## macrof1 = macrof1_fn(preds, xtest)
    ## print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    return acc, macrof1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp", help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=20, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--save', type=bool, default=False, help="save the trained model in the directory trained_model")
    parser.add_argument('--load', type=bool, default=False, help="load the trained model in the directory trained_model")
    parser.add_argument('--path', type=str, default=None, help="the path to load/save the model")
    parser.add_argument('--experiment', action='store_true', help="run hyperparameter experiments")
    args = parser.parse_args()

    if args.experiment:
        run_experiments(args)
    else:
        main(args)