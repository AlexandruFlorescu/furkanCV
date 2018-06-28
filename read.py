import pandas as pd
import pprint
import glob
import os

import matplotlib.pyplot as plt
from operator import itemgetter

pp = pprint.PrettyPrinter(depth=6)
import redis

r = redis.Redis(
    host='127.0.0.1',
    port='6379')

rootdir = os.path.dirname(os.path.realpath(__file__))
print rootdir
alldata = {}

for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		extension = file.split('.')[-1]
		if(extension == 'txt'): #or extension == 'csv'
			print subdir, dirs, files
			df = pd.read_csv(file, names=['empty', 'date_time', 'lastP', 'last_size', 'Acc Volume',	'bidP', 'askP','eight','nine','ten','eleven','twelve'])[:-1]

			df['date_time'] = pd.to_datetime(df.date_time)

			df = df.set_index(df.date_time)

			minlastP = pd.DataFrame()
			minlastP['lastP'] = df['lastP']
			minlastP['lastP'] = df['lastP'].resample('S').min()
			minlastP = minlastP.groupby(minlastP.index).first()
			print minlastP

			maxlastP = pd.DataFrame()
			maxlastP['lastP'] = df['lastP']
			maxlastP['lastP'] = df['lastP'].resample('S').max()
			maxlastP = maxlastP.groupby(maxlastP.index).first()

			fig, axmin = plt.subplots()

			axmax = axmin.twinx()
			axavg = axmin.twinx()
			print minlastP.index
			axmin.plot(minlastP.index , minlastP['lastP'], 'g-')
			axmax.plot(maxlastP.index , maxlastP['lastP'], 'b-')
			plt.show()


print df.head()

r.set('key', maxlastP)
value = r.get('key')
print value
