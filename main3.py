import numpy as np
import random
from sklearn.datasets import make_blobs
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split


class Neuron():
    def __init__(self, input_count):
        self.weights=[]
        self.input_count=input_count
        for i in range(0, input_count):
            self.weights.append(np.random.uniform(0, 1))


    def get_cost(self, inputs):
        cost = 0
        for i in range(self.input_count):
            cost += (inputs[i] - self.weights[i]) ** 2
        cost = cost ** 0.5
        return cost

    def change_weight(self, inputs, learning_kf):
            for i in range(len(inputs)):
                self.weights[i]=self.weights[i]+(learning_kf*(inputs[i]-self.weights[i]))

    def get_place(self):
        return self.weights



class Kohonen():
    neurons_list = []


    def __init__(self, count_neurons, input_count=2,lr=1,d=0.001):
        self.count_neurons = count_neurons
        self.input_count = input_count
        self.create_neurons(count_neurons, input_count)
        self.lr=lr
        self.d=d

    def create_neurons(self, count_neurons, input_count):
        for i in range(count_neurons):
            self.neurons_list.append(Neuron(input_count))


    def fit(self, X):
        i=0
        epoch=0
        while self.lr>0:
           self.train(X[i])
           i+=1
           if i==len(X):
                i=0
                epoch+=1
           if epoch%50==0 and epoch!=0:
                self.lr-=self.d


    def train(self,x):
        results=[]
        for i in self.neurons_list:
            results.append(i.get_cost(x))
        a=np.argmin(results)
        self.neurons_list[a].change_weight(x,self.lr)


    def predict(self, x):
        res=[]
        for i in self.neurons_list:
            res.append(i.get_cost(x))
        a=np.argmin(res)
        return a

    def test(self,X,Y):
        good=0
        all=0
        for i in range(len(X)):
            res=self.predict(X[i])
            if res==Y[i]:
                good+=1
            all+=1
        print("Accuracy={}%".format((good/all)*100))
                
    def plot(self):
        pass


    def get_location(self):
        List=[]
        for i in self.neurons_list:
            List.append(i.get_place())
        return List


if __name__ == '__main__':
    X, Y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=3)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

    model = Kohonen(3)
    model.fit(X_train)
    p=model.get_location()

    model.test(X_test,y_test)
    plt.scatter([x[0] for x in p],[x[1] for x in p],color='r')
    plt.scatter([x[0] for x in X], [x[1] for x in X])
    plt.show()




