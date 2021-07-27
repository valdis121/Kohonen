import numpy as np 
from sklearn.datasets import make_blobs
from sklearn import preprocessing
import matplotlib.pyplot as plt 
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
            self.weights[i] = self.weights[i] + (learning_kf * (inputs[i] - self.weights[i]))

    def get_place(self):
        return self.weights



class Kohonen():
    neurons_list = []
    bool_plot=False
    plot1=None
    plot2 = None
    fig=None
    ax=None


    def __init__(self, count_neurons, input_count=2,lr=0.3,d=0.1):
        self.count_neurons = count_neurons
        self.input_count = input_count
        self.create_neurons(count_neurons, input_count)
        self.lr=lr
        self.d=d

    def create_neurons(self, count_neurons, input_count):
        for i in range(count_neurons):
            self.neurons_list.append(Neuron(input_count))

    def color(self, X):
        
        colors = ["b", "g", "k", "y", "m", "tab:pink"]
        dictor = dict()
        for i in colors:
            dictor[i]=list()
        for i in X:
            temp = self.predict(i)

            color = colors[temp]
            dictor[color].append(i)

        return dictor
    
    
    def fit(self, X):
        i=0
        epoch=0
        while self.lr>0:
            self.train(X[i])
            self.plot(X)


            i+=1
            if i==len(X):
                i=0
                epoch+=1
            if epoch%1==0 and epoch!=0:
                self.lr-=self.d



        plt.ioff()
        plt.show()


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



    def plot(self, X):
        if self.bool_plot==True:
            p=self.get_location()
            x=[x[0] for x   in p]
            y=[x[1] for x in p]
            plt.clf()
            plt.title("priznakovy priestor")
            plt.xlabel("xlabel")
            plt.ylabel("ylabel")
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            
            klasters = self.color(X)
            keys = ["b", "g", "k", "y", "m", "tab:pink"]
            
            for i in range(0, 3):
                key = keys[i]
                plt.scatter([x[0] for x in klasters[key]], [x[1] for x in klasters[key]],marker = 's',color= key)
                
            plt.scatter(x, y, color='r', s=120)


            plt.pause(0.2)


        else:
            plt.ion()
            self.bool_plot=True
            self.fig, self.ax = plt.subplots()
            p=self.get_location()
            plt.title("priznakovy priestor")
            plt.xlabel("xlabel")
            plt.ylabel("ylabel")
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            
            klasters = self.color(X)
            keys = ["b", "g", "k", "y", "m", "tab:pink"]
            
            for i in range(0, len(klasters)):
                key = keys[i]

                self.plot1=self.ax.scatter([x[0] for x in klasters[key]], [x[1] for x in klasters[key]],marker = 's',color= key)
                 
            self.plot2=self.ax.scatter([x[0] for x in p],[x[1] for x in p],color='r', marker = 'o', s = 120)




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
