from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 1.5 8

df = pd.read_csv('mote17.csv')
# df.plot(x = 'epoch', y=['temperature', 'humidity', 'voltage', 'light']) #
# plt.show()
df = df.drop('datetime', axis=1)
df = df.drop('moteid', axis=1)
df = df.drop('voltage', axis=1)
df = df[df['light']<1000]
X = df.values

# X = StandardScaler().fit_transform(X)
X = MinMaxScaler().fit_transform(X)
# print (X)

# mp = 1.25
# minPoints = len(X)*mp/100
minPoints = 350

dbscan = DBSCAN(eps=0.1, min_samples=minPoints).fit(X)
cs = [0]*len(set(dbscan.labels_))
for lb in dbscan.labels_:
	cs[lb] += 1

print (dbscan.labels_)
print (set(dbscan.labels_))
print (cs)

#3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for index,value in enumerate(X):
	if dbscan.labels_[index] == 0:
		ax.scatter(value[0], value[1], value[2], marker='o', color='r')
	elif dbscan.labels_[index] == 1:
		ax.scatter(value[0], value[1], value[2], marker='s', color='b')
	elif dbscan.labels_[index] == 2:
		ax.scatter(value[0], value[1], value[2], marker='D', color='g')
	elif dbscan.labels_[index] == 3:
		ax.scatter(value[0], value[1], value[2], marker='P', color='c')
	elif dbscan.labels_[index] == -1:
		ax.scatter(value[0], value[1], value[2], marker='8', color='m')
# 	elif dbscan.labels_[index] == 5:
# 		ax.scatter(value[0], value[1], value[2], marker='^', color='y')

ax.set_xlabel('temperature')
ax.set_ylabel('humidity')
ax.set_zlabel('light')
plt.show()
