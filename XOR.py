import random
from Perceptron import Perceptron

if __name__ == '__main__':
    num_train = 10
    training_examples = []
    training_labels = []
    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        training_labels.append(1.0 if (training_examples[i][0] > 0.75) != (training_examples[i][1] > 0.75) else 0.0)
    notA = Perceptron(1,bias=-0.75)
    notB = Perceptron(1,bias=-0.75)
    AnotB = Perceptron(2, bias = -1.75)
    BnotA = Perceptron(2,bias = -1.75)
    orGate = Perceptron(2,bias = -0.5)
    print(training_examples)
    print(training_labels)
