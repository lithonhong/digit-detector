# digit-detector

This repository contains a handwritten digit detector web app that utilises a feedforward neural network and backpropagation.
The neural network is trained with the [MNIST database](https://web.archive.org/web/20200430193701/http://yann.lecun.com/exdb/mnist/).

This repository is based on [Chapter 1 of Neural Network and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) by Michael Nielson.
(I unfortunately did not follow through with the rest of the book due to the mathematical prerequisite for its content; would love to revisit it another day though!)

## Usage

Test out this digit detector [here](https://lithonhong.github.io/digit-detector/)!

Alternatively, if you wish to run the neural network locally, feel free to clone this repository and run `training.ipynb` to re-train the neural network.

## Specification

The neural network was written and trained in Python, using only `numpy` for the core infrastructure.
The aim of this project was to understand how feedforward neural networks work without relying on existing modules.
The weights and biases of the neural network are then exported to a JSON file, such that 

As mentioned in the summary, this project uses a feedforward neural network and backpropagation.
A training period of 30 epoches and a learning rate ($\eta$) of 0.01 is used.
The structure of the neural network is as follows:

* One input layer of 784 nodes, corresponding to the 28x28 image given
* One hidden layer of 30 nodes
* One output layer of 10 nodes, corresponding to the 10 possible digits

A mini-batch size of 10 is used to implement mini-batch stochastic gradient descent (SGD).
The backpropagation algorithm was copied from the tutorial as Nielson invited the readers to "assume it works as claimed" before it was formally introduced in the following chapter.

Unlike the original MNIST database, which contains 60,000 training images, only 50,000 of which is used for training.
The remaining 10,000 images were categorised into a separate validation dataset to verify the accuracy of the neural network.

## Files

| File | Function |
|------|----------|
| index.html | Contains HTML to render the website. |
| nnet.json | Stores the weights and biases of a neural network obtained from `training.ipynb`. Results will vary every time `training.ipynb` is ran. |
| nnet.py | Comprises the module for the neural network. |
| README.md | You are here! |
| script.js | Handles user input from the web app. |
| styles.css | Contains CSS to style the website. |
| training.ipynb | Trains the neural network by imported data from MNIST. |


## Improvements

Seeing as the neural network still fails to recognise some of my own handwriting despite a 94.50% accuracy from the validation set, this project definitely has room for improvement.
I propose the following ideas that would potentially help:

* Reduce the size of the validation set to allow more data to be used for training
* Rescale the input image to comply with MNIST standards
* Modify the hidden layer(s)

I have also yet to style the web app with CSS.