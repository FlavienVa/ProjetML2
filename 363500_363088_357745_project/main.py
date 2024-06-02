from datetime import datetime
import matplotlib.pyplot as plt


import argparse

import numpy as np
from torchinfo import summary
import torch

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import (
    normalize_fn,
    append_bias_term,
    accuracy_fn,
    macrof1_fn,
    get_n_classes,
)
import os


def run_experiments(args):
    def train_and_evaluate(lr, max_iters, xtrain, ytrain, xvalid, yvalid, xtest):
        args.lr = lr
        args.max_iters = max_iters
        acc_test, f1_test, acc_valid, f1_valid = main(
            args,
            first=first,
            xtest=xtest,
            xtrain=xtrain,
            ytrain=ytrain,
            xvalid=xvalid,
            yvalid=yvalid,
        )
        return acc_test, f1_test, acc_valid, f1_valid

    xtrain, xtest, ytrain = load_data(args.data_path)

    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    random_index = np.arange(xtrain.shape[0])
    np.random.shuffle(random_index)
    xtrain, xvalid = (
        xtrain[random_index[: int(0.8 * xtrain.shape[0])]],
        xtrain[random_index[int(0.8 * xtrain.shape[0]) :]],
    )
    ytrain, yvalid = (
        ytrain[random_index[: int(0.8 * ytrain.shape[0])]],
        ytrain[random_index[int(0.8 * ytrain.shape[0]) :]],
    )
    mean = np.mean(xtrain)
    std = np.std(xtrain)
    xtrain = normalize_fn(xtrain, means=mean, stds=std)
    xtest = normalize_fn(xtest, means=mean, stds=std)
    xvalid = normalize_fn(xvalid, means=mean, stds=std)

    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        xvalid = pca_obj.reduce_dimension(xvalid)

    lrs = [1e-3, 1e-2, 1e-1]
    max_iters_list = [
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ]
    first = True
    last_iter = 0
    results = []
    for lr in lrs:
        first = True
        last_iter = 0
        for max_iters in max_iters_list:
            acc_test, f1_test, acc_valid, f1_valid = train_and_evaluate(
                lr, max_iters - last_iter, xtrain, ytrain, xvalid, yvalid, xtest
            )
            last_iter = max_iters
            first = False
            results.append((lr, max_iters, acc_test, f1_test, acc_valid, f1_valid))

    plot_results(results, lrs, max_iters_list)


def plot_results(results, lrs, max_iters_list):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for lr in lrs:
        accs_test = [result[2] for result in results if result[0] == lr]
        f1s_test = [result[3] for result in results if result[0] == lr]
        accs_valid = [result[4] for result in results if result[0] == lr]
        f1s_valid = [result[5] for result in results if result[0] == lr]
        axes[0].plot(max_iters_list, accs_test, label=f"lr={lr} accuracy")
        axes[0].plot(max_iters_list, f1s_test, marker="o", label=f"lr={lr} f1 score")
        axes[1].plot(max_iters_list, accs_valid, label=f"lr={lr} accuracy")
        axes[1].plot(max_iters_list, f1s_valid, marker="o", label=f"lr={lr} f1 score")

    axes[0].set_xlabel("Max Iterations in epoch")
    axes[0].set_ylabel("Accuracy and F1 score in %")
    axes[0].set_title("Test data set metrics in function of Max Iterations")
    axes[0].legend()

    axes[1].set_xlabel("Max Iterations in epoch")
    axes[1].set_ylabel("Accuracy and F1 score in %")
    axes[1].set_title("Valid data set metrics in function of Max Iterations")
    axes[1].legend()
    fig.suptitle("CNN", fontsize=25)
    plt.tight_layout()
    plt.show()


def main(
    args, first=False, xtrain=None, xvalid=None, xtest=None, ytrain=None, yvalid=None
):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    if not args.experiment:
        ## 1. First, we load our data and flatten the images into vectors
        xtrain, xtest, ytrain = load_data(args.data_path)
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        # print(xtrain.shape, xtest.shape, ytrain.shape)

        ## 2. Then we must prepare it. This is were you can create a validation set,
        #  normalize, add bias, etc.

        # print(xtrain.shape, xtest.shape, ytrain.shape)

        # Make a validation set
        if not args.test:
            # Split the training data into training and validation sets
            random_index = np.arange(xtrain.shape[0])
            np.random.shuffle(random_index)
            xtrain, xvalid = (
                xtrain[random_index[: int(0.8 * xtrain.shape[0])]],
                xtrain[random_index[int(0.8 * xtrain.shape[0]) :]],
            )
            ytrain, yvalid = (
                ytrain[random_index[: int(0.8 * ytrain.shape[0])]],
                ytrain[random_index[int(0.8 * ytrain.shape[0]) :]],
            )

        mean = np.mean(xtrain)
        std = np.std(xtrain)
        xtrain = normalize_fn(xtrain, means=mean, stds=std)
        xtest = normalize_fn(xtest, means=mean, stds=std)
        # print(xtrain.shape, xtest.shape, ytrain.shape, xvalid.shape)
        if not args.test:
            xvalid = normalize_fn(xvalid, means=mean, stds=std)

    ### WRITE YOUR CODE HERE
    # print("Using PCA")
    if args.device == "cuda":
        if torch.cuda.is_available():
            print("Device use: CUDA")
            device = torch.device("cuda")
        else:
            print("ERROR specified device unusable -> CPU")
            device = torch.device("cpu")
    elif args.device == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("Device use: MPS")
            device = torch.device("mps")
        else:
            print("ERROR device specified unusable -> CPU")
            device = torch.device("cpu")
    else:
        print("Device use: CPU")
        device = torch.device("cpu")
    # Dimensionality reduction (MS2)
    if args.use_pca and not args.experiment:
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

        model = MLP(
            input_size=xtrain.shape[1], n_classes=10, device=device
        )  ### WRITE YOUR CODE HERE

    if args.nn_type == "cnn":
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        xvalid = xvalid.reshape(-1, 1, 28, 28)

        model = CNN(input_channels=1, n_classes=10)

    if args.nn_type == "transformer":
        xtrain = xtrain.reshape(-1, 1, 28, 28)
        xtest = xtest.reshape(-1, 1, 28, 28)
        xvalid = xvalid.reshape(-1, 1, 28, 28)
        print(xtrain.shape)
        model = MyViT(xtrain.shape[1:], device=device)

    if args.load == True and args.path != None:
        model.load_state_dict(torch.load(args.path))

    if args.experiment and not first:
        model.load_state_dict(torch.load(args.path))

    summary(model)
    # Trainer object
    method_obj = Trainer(
        model,
        lr=args.lr,
        epochs=args.max_iters,
        batch_size=args.nn_batch_size,
        device=device,
    )

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
    # print("preds_train" , np.shape(preds_train))

    # Predict on unseen data
    preds = method_obj.predict(xtest)
    # print("preds" , np.shape(preds))
    predsvalid = method_obj.predict(xvalid)
    # print("predsvalid", np.shape(predsvalid))

    ## Report results: performance on train and valid/test sets
    acc_train = accuracy_fn(preds_train, ytrain)
    macrof1_train = macrof1_fn(preds_train, ytrain) * 100
    print(f"\nTrain set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.3f}%")

    acc_valid = accuracy_fn(predsvalid, yvalid)
    macrof1_valid = macrof1_fn(predsvalid, yvalid) * 100
    print(
        f"\nValidation set: accuracy = {acc_valid:.3f}% - F1-score = {macrof1_valid:.3f}%"
    )

    np.save("predictions", preds)

    if args.experiment:
        torch.save(model.state_dict(), args.path)

    if args.save == True:
        current_time = datetime.now().isoformat(timespec="minutes")
        torch.save(
            model.state_dict(),
            f"trained_model/{model.__class__.__name__}-{macrof1_valid:.5f}-{current_time}",
        )
    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    ## acc = accuracy_fn(preds, xtest)
    ## macrof1 = macrof1_fn(preds, xtest)
    ## print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    return acc_train, macrof1_train, acc_valid, macrof1_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--nn_type",
        default="mlp",
        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'",
    )
    parser.add_argument(
        "--use_pca", action="store_true", help="use PCA for feature reduction"
    )
    parser.add_argument(
        "--pca_d", type=int, default=100, help="the number of principal components"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="save the trained model in the directory trained_model",
    )
    parser.add_argument(
        "--load",
        type=bool,
        default=False,
        help="load the trained model in the directory trained_model",
    )
    parser.add_argument(
        "--path", type=str, default=None, help="the path to load/save the model"
    )
    parser.add_argument(
        "--experiment", action="store_true", help="run hyperparameter experiments"
    )
    args = parser.parse_args()

    if args.experiment:
        run_experiments(args)
    else:
        main(args)
