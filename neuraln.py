import numpy as np

class NeuralNetwork:

    def __init__(self,size):
        self.size = size
        self.weight= []
        self.bias = []

        
        for i in range(len(size)-1):
            # Initialize weights with small random values and biases with zeros
            # Using Xavier initialization for weights to help with training stability
            self.weight.append(np.random.randn(size[i],size[i+1])*np.sqrt(2/(size[i]+size[i+1])))
            self.bias.append(np.zeros((1,size[i+1])))
    
    
    def forward(self,input_data):
        self.forward_values = []
        self.activation_values = []
        self.activation_values.append(input_data)

        for i in range(len(self.size)-2):
            self.forward_values.append(np.dot(self.activation_values[-1],self.weight[i]) + self.bias[i])
            self.activation_values.append(np.tanh(self.forward_values[-1]))

        self.forward_values.append(np.dot(self.activation_values[-1],self.weight[-1]) + self.bias[-1])
        self.activation_values.append(self.forward_values[-1])  # Output layer (no activation)

   

    def backward(self,target,learning_rate):
        m=len(target)
        delta=[0]*(len(self.size)-1)
        for i in range(len(self.size)-2,-1,-1):
            if i==len(self.size)-2:
                error = (self.activation_values[-1] - target)**2/m
                delta[i] = 2*(self.activation_values[-1] - target)/m
                dw=np.dot(self.activation_values[i].T,delta[i])
                # the bias update is computed by summing the delta across all samples in the batch(mean of the delta) for keeping the shape [1, number of neurons in the layer].
                db=np.sum(delta[i],axis=0,keepdims=True)
                self.bias[i] -= learning_rate * db
                self.weight[i] -= learning_rate * dw
            else:
                 delta[i]=np.dot(delta[i+1],self.weight[i+1].T)*(1-self.activation_values[i]**2)
                 dw=np.dot(self.activation_values[i-1].T,delta[i])
                 db=np.sum(delta[i],axis=0,keepdims=True)
                 self.bias[i] -= learning_rate * db
                 self.weight[i] -= learning_rate * dw