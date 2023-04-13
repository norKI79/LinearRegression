#This script generates some random data and then shows the linear model being fitted progressively

#ChatGPT: 
# prompt1: create a simple linear regression model of some random size data (2-100)
# set of 1 feature and an output value display the line 
# for w and b for each recalulation and also display the data points on the plot
#
# prompt2: i want plt.show() to close and refresh after 1 second

import numpy as np
import matplotlib.pyplot as plt
import time


# Generate random data 10-100 points
n = np.random.randint(10, 100)
x = np.random.rand(n)

m=np.random.random()
b=np.random.random()
y = (m*x)+b+np.random.randn(n)

# Define initial weights and learning rate
w = 0.1
b = 0.1
lr = 0.1

# Define function to update weights for each epoch
def train(x, y, w, b, lr):
    y_pred = w * x + b
    error = y_pred - y
    w -= lr * np.dot(error, x) / n
    b -= lr * np.mean(error)
    return w, b, np.dot(error,x)/n

# Train model for 50 epochs and display line and points after each epoch

epochs = 50
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,0.9])
plt.ylim([y.min(),y.max()])

line=None
e_label=None
x_pos=250
y_pos=0
fig_manager = plt.get_current_fig_manager()
backend = plt.get_backend()
if backend == 'TkAgg':
    fig_manager.window.wm_geometry("+"+str(x_pos)+"+200")
elif backend == 'WXAgg':
    fig_manager.window.SetPosition((x_pos, y_pos))
else:
    # This works for QT and GTK
    # You can also use window.setGeometry
    fig_manager,manager.window.move(x_pos, y_pos)

for i in range(epochs):
    
    w, b, error = train(x, y, w, b, lr)
    y_pred = w * x + b
    
    if line:
        line.remove()
    line, = plt.plot(x, y_pred,c='red')
    
    epoch_info = 'Epoch: {}, w: {:.2f}, b: {:.2f}, Error:{:.2f}'.format(i+1, w, b,error)
    if e_label:
        e_label.remove()
    e_label=plt.text(x.min(),y.max()+0.01,epoch_info, fontsize=10) 
    
    plt.draw()
    plt.pause(0.21)
    
