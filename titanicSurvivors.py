import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from copy import deepcopy
from sklearn import preprocessing

# a.read json file
data = json.load(open('titanic.json'))

# b.create dataset
ages =      []
fares =     []
siblings =  []
locations = []
sexes =		[]
survivors =	[]
for it in data:
	if it['Age'] != '' and it['Fare'] != '' and it['Embarked']!='' and it['Sex']!='' and it['Survived']!='':
		# get ages
		ages.append(float(it['Age']))
		
		# get fares
		fares.append(float(it['Fare']))
		
		# get total siblings
		sibs = 0
		if it['SiblingsAndSpouses'] != '':
			sibs += int(it['SiblingsAndSpouses'])
		if it['ParentsAndChildren'] != '':
			sibs += int(it['ParentsAndChildren'])
			siblings.append(sibs)
		
		# get embarking locations 
		if it['Embarked'] == 'C':
			locations.append(1)
		elif it['Embarked'] == 'Q':
			locations.append(2)
		else:
			locations.append(3)

		#  get sex
		if it['Sex'] == 'male':
			sexes.append(0.0)
		else:
			sexes.append(1.0)
		# get survived status
		survivors.append(float(it['Survived']))

# ensamble the dataset
dataset = zip(ages, fares, siblings, locations, sexes, survivors)


dt = [[data[i] for i in range(0,6)] for data in dataset]
df = pd.DataFrame(dt, columns=['age','fare','siblings','location','sex','survivors'])


# c. cluster analysis
# plot dendogram
# sns.set(color_codes=True)
# dendo = sns.clustermap(df, standard_scale=1, metric="correlation")
# plt.show()

# in order to display the dendogram I have normalized the data to [0,1] range
# from the diagram we can notice that there are 5 main clusters that have 
# a reasonable proximity to eachother and so, I consider this to be the optimal
# number of clusters to choose in order to draw relevant conclusions about the data
# Furthemore, these clusters are aproximately covering most of the data in the dataset
# and they only leave small noise gaps that are barely noticeable in our diagram.
# A lower ammount of clusters would have been unconclusive, due to the hight distances
# between the items, while a higher ammount of clusters would allow for too much 'noise' 
# to be represented in the dataset.

# d. k-means implementation
# d.1 data normalization
# scales on [0,1] using MinMax method
scaler = preprocessing.MinMaxScaler()
dv = scaler.fit_transform(df)

# d.2 euclidian distance
def dist(a, b,  ax=1):
    return np.linalg.norm(a - b, axis=ax)

k = 5

# initialize centroids
Cage = np.random.random_sample(size=k)
Cfare = np.random.random_sample(size=k)
Csibling = np.random.random_sample(size=k)
Clocation = np.random.choice([0,0.5,1], size=k)
Csex = np.random.randint(0, 1, size=k)
Csurvivor = np.random.randint(0, 1, size=k)
C = np.array(list(zip(Cage, Cfare, Csibling, Clocation, Csex, Csurvivor)), dtype=np.float32)

# old centroid
oldC = np.zeros(C.shape)
clusters = np.zeros(len(dv))

# 10 iterations
for it in xrange(11):
	
	if(it==0):
		# plot points
		for p in dv:
			if p[3] == 0: # uses location
				clr = 'y'
			elif p[3] == 0.5:
				clr = 'k'
			else:
				clr = 'r'
			if p[5] == 0:
				mkr = 'x'
			else:
				mkr = 'o'
			plt.scatter(p[0], (p[1]+p[2])/2, c=clr, marker=mkr)
		# plot centroids
		for c in C:
			plt.scatter(c[0], (c[1]+c[2])/2, c='g', marker='*', s=200)
		plt.show()#savefig('loc0.png')
		# &&
		for p in dv:
			if p[4] == 0: # uses sex
				clr = 'r'
			else:
				clr = 'b'
			if p[5] == 0:
				mkr = 'x'
			else:
				mkr = 'o'
			plt.scatter(p[0], (p[1]+p[2])/2, c=clr, marker=mkr)
		# plot centroids
		for c in C:
			plt.scatter(c[0], (c[1]+c[2])/2, c='g', marker='*', s=200)
		plt.show()#savefig('sex0.png')


	if(it==5):
		# plot points
		for p in dv:
			if p[3] == 0: # uses location
				clr = 'y'
			elif p[3] == 0.5:
				clr = 'k'
			else:
				clr = 'r'
			if p[5] == 0:
				mkr = 'x'
			else:
				mkr = 'o'
			plt.scatter(p[0], (p[1]+p[2])/2, c=clr, marker=mkr)
		# plot centroids
		for c in C:
			plt.scatter(c[0], (c[1]+c[2])/2, c='g', marker='*', s=200)
		plt.show()#savefig('loc5.png')


		for p in dv:
			if p[4] == 0: # uses sex
				clr = 'r'
			else:
				clr = 'b'
			if p[5] == 0:
				mkr = 'x'
			else:
				mkr = 'o'
			plt.scatter(p[0], (p[1]+p[2])/2, c=clr, marker=mkr)
		# plot centroids 
		for c in C:
			plt.scatter(c[0], (c[1]+c[2])/2, c='g', marker='*', s=200)
		plt.show()#savefig('sex5.png')

	if(it==10 or dist(oldC, C, None) == 0):
		# plot points
		for p in dv:
			if p[3] == 0: # uses location
				clr = 'y'
			elif p[3] == 0.5:
				clr = 'k'
			else:
				clr = 'r'
			if p[5] == 0:
				mkr = 'x'
			else:
				mkr = 'o'
			plt.scatter(p[0], (p[1]+p[2])/2, c=clr, marker=mkr)
		# plot centroids
		for c in C:
			plt.scatter(c[0], (c[1]+c[2])/2, c='g', marker='*', s=200)
		plt.show()#savefig('loc10.png')


		for p in dv:
			if p[4] == 0: # uses sex
				clr = 'r'
			else:
				clr = 'b'
			if p[5] == 0:
				mkr = 'x'
			else:
				mkr = 'o'
			plt.scatter(p[0], (p[1]+p[2])/2, c=clr, marker=mkr)
		# plot centroids
		for c in C:
			plt.scatter(c[0], (c[1]+c[2])/2, c='g', marker='*', s=200)
		plt.show()#savefig('sex10.png')
	
	# if convergence, exit loop
	if dist(oldC, C, None) == 0:
		it=11

	# Assign each value to it's closest cluster
	for i in xrange(len(dv)):
		distances = dist(dv[i], C)
		cluster = np.argmin(distances)
		clusters[i] = cluster

	# Storing old centroids
	oldC = deepcopy(C)

	# Finding the new centroids
	for i in xrange(k):
		points = [dv[j] for j in xrange(len(dv)) if clusters[j] == i]
		C[i] = np.mean(points, axis=0)

# print clusters
# at this point we have the 5 clusters computed
# looking at the clusters we can notice that:
# a) fare and siblings have high numerical correlation (0,0.20)
# so we can perhaps consider merging the features with a mean
# b) sex stays in the binary range 0,1 so we can represent this
# on the diagram with different icons
# c) location is approximatively binary, so we can represent it
# on another plot, using different icons 
# d) as such, we are left with two dimensions to represent xy axes,
# two feature to represent shape (on two different plots)
# and a feature to represent the color