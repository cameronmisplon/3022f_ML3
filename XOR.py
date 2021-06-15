import random
from Perceptron import Perceptron

if __name__ == '__main__':
    num_train = 100
    learning_rate = 0.1
    training_examples = []
    values_for_each_perceptron = [[],[],[],[],[]]
    valid_labels = []
    notA_labels = []
    notB_labels =[]
    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        values_for_each_perceptron[0].append([training_examples[i][0]])
        values_for_each_perceptron[1].append([training_examples[i][1]])
        notA_labels.append(1.0 if training_examples[i][0] >= 0.75 else 0.0)
        notB_labels.append(1.0 if training_examples[i][1] >= 0.75 else 0.0)
        valid_labels.append(1.0 if (training_examples[i][0]>=0.75) != (training_examples[i][1]>=0.75) else 0.0)
    notA = Perceptron(1,bias=-0.75)
    notB = Perceptron(1,bias=-0.75)
    AnotB = Perceptron(2, bias = -1.75)
    BnotA = Perceptron(2,bias = -1.75)
    orGate = Perceptron(2,bias = -0.5)
    l =0
    valid_percentage =0
    while valid_percentage < 0.98:
        l+=1
        print('------ Iteration '+str(l)+ ' ------')
        notA.train(values_for_each_perceptron[0],notA_labels,learning_rate)
        notB.train(values_for_each_perceptron[1],notB_labels,learning_rate)
        AnotB_labels = []
        BnotA_labels =[]
        values_for_each_perceptron[2] = []
        values_for_each_perceptron[3] = []
        for j in range(num_train):
            AnotB_labels.append(1.0 if training_examples[j][0]>=0.75 and training_examples[j][1]<0.75 else 0.0)
            BnotA_labels.append(1.0 if training_examples[j][1]>=0.75 and training_examples[j][0]<0.75 else 0.0)
            values_for_each_perceptron[2].append([training_examples[j][0],notB.activate(values_for_each_perceptron[1][j])])
            values_for_each_perceptron[3].append([training_examples[j][1],notA.activate(values_for_each_perceptron[0][j])])
        AnotB.train(values_for_each_perceptron[2],AnotB_labels,learning_rate)
        BnotA.train(values_for_each_perceptron[3],BnotA_labels,learning_rate)
        values_for_each_perceptron[4] = []
        for k in range(num_train):
            temp1 = AnotB.activate(values_for_each_perceptron[2][k])
            temp2 = BnotA.activate(values_for_each_perceptron[3][k])
            values_for_each_perceptron[4].append([temp1,temp2])
        orGate.train(values_for_each_perceptron[4],valid_labels,learning_rate)
        valid_percentage=orGate.validate(values_for_each_perceptron[4],valid_labels,verbose=False)
        print(valid_percentage)
        if l==100:
            break
