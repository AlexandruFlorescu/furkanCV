import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def sigmoid(x):
	# compute the sigmoid
	return 1/(1+np.exp(-x))

def dsigmoid(x):
	# compute the derivative sigmoid
	return (x*(1-x))

# read dataset
inputs = pd.read_csv('dataset.csv')[['i1','i2','i3','i4','i5']].values
outputs = pd.read_csv('dataset.csv')[['o1','o2','o3']].values

# read data
por = pd.read_csv("student-por.csv")
mat = pd.read_csv("student-mat.csv")
mat['Class'] = "math"
por['Class'] = "port"

# merge and clean data
data = pd.concat([mat,por],axis = 0)
data = data.drop(columns = ['school', 'sex', 'address', 'Mjob', 'Fjob', 'reason', 'guardian', 'Class'], index=1)

# format to numeric
data.loc[data['famsize'] == 'LE3', 'famsize'] = 0
data.loc[data['famsize'] == 'GT3', 'famsize'] = 1

data.loc[data['Pstatus'] == 'T', 'Pstatus'] = 1
data.loc[data['Pstatus'] == 'A', 'Pstatus'] = 0

data.loc[data['schoolsup'] == 'yes', 'schoolsup'] = 1
data.loc[data['schoolsup'] == 'no', 'schoolsup'] = 0

data.loc[data['famsup'] == 'yes', 'famsup'] = 1
data.loc[data['famsup'] == 'no', 'famsup'] = 0

data.loc[data['paid'] == 'yes', 'paid'] = 1
data.loc[data['paid'] == 'no', 'paid'] = 0

data.loc[data['activities'] == 'yes', 'activities'] = 1
data.loc[data['activities'] == 'no', 'activities'] = 0

data.loc[data['nursery'] == 'yes', 'nursery'] = 1
data.loc[data['nursery'] == 'no', 'nursery'] = 0

data.loc[data['higher'] == 'yes', 'higher'] = 1
data.loc[data['higher'] == 'no', 'higher'] = 0

data.loc[data['internet'] == 'yes', 'internet'] = 1
data.loc[data['internet'] == 'no', 'internet'] = 0

data.loc[data['romantic'] == 'yes', 'romantic'] = 1
data.loc[data['romantic'] == 'no', 'romantic'] = 0
# 

# select 85% as train data
train = sorted(random.sample(range(0,len(data)), 85*len(data)/100)) 
test = sorted(list(set(range(0,len(data))) - set(train)))

# split train/test data
trainInputs = np.array([X[t] for t in train])
trainOutputs = np.array([Y[t] for t in train])
testInputs = np.array([X[t] for t in test])
testOutputs = np.array([Y[t] for t in test])


# initialize metaparameters
epochs_count = 100000
reg = 1
learning_rate = 0.2

### I didn't use this due to getting higher accuracies with more epochs
### it is implemented, nonetheless
error_threshold = 0.2
###

# initialize errors list for plotting
errors = []

np.random.seed(1) 
# initialize hidden layer
w1 = np.random.random((5,4)) 
b1 = np.random.random(4)
# initialize output layer
w2 = np.random.random((4,3)) 
b2 = np.random.random(3)

err2 = [1]
epoch = 0
# do the training
for epoch in xrange(epochs_count):
# while np.mean(np.abs(err2)) > error_threshold:
	# forwardpropagate
	h1 = sigmoid(np.dot(trainInputs, w1) +b1)
	h2 = sigmoid(np.dot(h1,w2) +b2)
	
	# backpropagate layer 2
	err2 = trainOutputs - h2
	dh2 = err2*dsigmoid(h2)
	
	if epoch % 1000 == 0:
		print("Error: " + str(np.mean(np.abs(err2))))
	# epoch+=1
	errors.append(np.mean(np.abs(err2)))

	# backpropagate layer 1 
	err1 = dh2.dot(w2.T)
	dh1 = err1*dsigmoid(h1)

	# update weights
	w2 += learning_rate*reg*h1.T.dot(dh2)
	b2 += learning_rate*reg*sum(dh2)
	w1 += learning_rate*reg*trainInputs.T.dot(dh1)
	b1 += learning_rate*reg*sum(dh1)

h1 = sigmoid(np.dot(testInputs, w1)+b1)
h2 = sigmoid(np.dot(h1,w2)+b2)
testErr = testOutputs - h2
print testOutputs, h2
print("Test:"+str(np.mean(np.abs(testErr))))

plt.plot(errors, linewidth=2.0)
plt.ylabel('error')
plt.xlabel('epochs')
plt.show()