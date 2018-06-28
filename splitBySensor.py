s1 = [17, 50]
s2 = [3,14,16,19,39]
s3 = [3,13,14,16,19,24,34,53]
s4 = [7,24,41,44,50]
s5 = [3,7,11,14,16,22,27,29,34,41]
s6 = [14,19,24,25,36]

import pandas as pd
import matplotlib.pyplot as plt
# # sensors series
# r1 = open('ss1.csv', 'w')
# r1.write('datetime,moteid,temperature,humidity,light,voltage\n')

# r2 = open('ss2.csv', 'w')
# r2.write('datetime,moteid,temperature,humidity,light,voltage\n')

# r3 = open('ss3.csv', 'w')
# r3.write('datetime,moteid,temperature,humidity,light,voltage\n')

# r4 = open('ss4.csv', 'w')
# r4.write('datetime,moteid,temperature,humidity,light,voltage\n')

# r5 = open('ss5.csv', 'w')
# r5.write('datetime,moteid,temperature,humidity,light,voltage\n')

# r6 = open('ss6.csv', 'w')
# r6.write('datetime,moteid,temperature,humidity,light,voltage\n')
# print (' here')
# with open('inteldata.csv') as f:
# 	lines = f.read().split('\n')

# 	for line in lines[1:-1]:
# 		print (line)
# 		moteid = int(float(line.split(',')[1]))
# 		print (moteid)
# 		if moteid in s1:
# 			r1.write(line + '\n')
# 		if moteid in s2:
# 			r2.write(line + '\n')
# 		if moteid in s3:
# 			r3.write(line + '\n')
# 		if moteid in s4:
# 			r4.write(line + '\n')
# 		if moteid in s5:
# 			r5.write(line + '\n')
# 		if moteid in s6:
# 			r6.write(line + '\n')



# sensors separation
# 'epoch,temperature,humidity,light,voltage\n'
with open('inteldata.csv') as f:
	lines = f.read().split('\n')

	# for moteid in range(1,55):
	r = open('mote17.csv', 'a')
	r.write('datetime,moteid,temperature,humidity,light,voltage\n')
	
	for line in lines[1:-1]:
		# sline = line.split(',')
		# newline = str(sline[0]) + str(sline[1]) + ',' + ','.join(sline[3:7]) + '\n'
		moteid = str(int(float(line.split(',')[1])))
		# print (newline, moteid)
		if(moteid=='17' and int(float(line.split(',')[-2])) <1000 ):
			r = open('mote'+moteid+'.csv', 'a')
			r.write(line+'\n')

# plot sensor by epoch
# df = pd.read_csv('sensors1.csv')
# df.plot(x='epoch', y=['temperature','humidity','voltage'])
# plt.show()


# filter epoch
# with open('raw_data2.csv') as f:
# 	lines = f.read().split('\n')
# 	for line in lines[1:-1]:
# 		epochid = int(float(line.split(',')[1]))
# 		if(epochid == 47):
# 			print( line)
# 			r = open('epoch'+str(epochid)+'.csv', 'a')
# 			r.write(line+'\n')

# # plot epoch
# df = pd.read_csv('epoch47.csv')

# df.plot(x='moteid', y=['temperature', 'humidity', 'voltage'])
# plt.show()