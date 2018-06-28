from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

# 1.5 8

df = pd.read_csv('mote17.csv')
# df.plot(x = 'epoch', y=['temperature', 'humidity', 'voltage', 'light']) #
# plt.show()
df = df.drop('datetime', axis=1)
df = df.drop('voltage', axis=1)
df = df[df['light']<1000]

X = df.values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['temperature'], df['humidity'], df['light'])
ax.set_xlabel('temperature')
ax.set_ylabel('humidity')
ax.set_zlabel('light')

# df.plot(x='temperature', y='humidity', z='light', kind ='scatter')
plt.show()