import numpy as np
import pandas as pd

def handleX(list1):
	#print(list1)
	list2 = []
	
	list2.append(list1[0])
	listOfZeros1 = [0]*2
	listOfZeros1[int(list1[1]) - 1] = 1
	list2.extend(listOfZeros1)
	
	listOfZeros2 = [0]*7
	listOfZeros2[int(list1[2])] = 1
	list2.extend(listOfZeros2)
	listOfZeros3 = [0]*4
	listOfZeros3[int(list1[3])] = 1
	list2.extend(listOfZeros3)
	'''
	listOfZeros2 = [0]*4
	temp = int(list1[2])
	if temp == 0 or temp > 4:
		listOfZeros2[3] = 1
	else:
		listOfZeros2[temp - 1] = 1
	list2.extend(listOfZeros2)
	
	listOfZeros3 = [0]*3
	temp = int(list1[3])
	if temp == 0 or temp > 3:
		listOfZeros3[2] = 1
	else:
		listOfZeros3[temp - 1] = 1
	list2.extend(listOfZeros3)
	'''
	list2.append(list1[4])
	listOfZeros4 = [0]*11
	listOfZeros4[int(list1[5]) + 2] = 1
	list2.extend(listOfZeros4)
	listOfZeros5 = [0]*11
	listOfZeros5[int(list1[6]) + 2] = 1
	list2.extend(listOfZeros5)
	listOfZeros6 = [0]*11
	listOfZeros6[int(list1[7]) + 2] = 1
	list2.extend(listOfZeros6)
	listOfZeros7 = [0]*11
	listOfZeros7[int(list1[8]) + 2] = 1
	list2.extend(listOfZeros7)
	listOfZeros8 = [0]*11
	listOfZeros8[int(list1[9]) + 2] = 1
	list2.extend(listOfZeros8)
	listOfZeros9 = [0]*11
	listOfZeros9[int(list1[10]) + 2] = 1
	list2.extend(listOfZeros9)
	list2.extend(list1[11:])
	'''
	list2.extend(list1[0:5])
	list2.extend(list1[11:])
	'''
	#print(list1)
	#print(list2)
	#exit()
	return list2

def getTrainX(path):
	tempX = []
	trainX = []
	data = []
	with open(path, 'rb') as fp:
		data = fp.read().split(b'\r\n')[1:]
	dataLen = int(len(data))
	#print(dataLen)		#20000
	for i in range(dataLen):
		temp = data[i].split(b',')
		temp = handleX(temp)
		tempX.append(temp)
		#print(temp)
		#print('---')
	tempX = np.concatenate((tempX,np.ones((len(tempX),1))),axis=1)
	trainX = np.asarray(tempX).astype(float)
	#print(trainX)
	#print(trainX.shape)	#(20000, 31)
	return trainX

def getTrainY(path):
	trainY = []
	data = []
	with open(path, 'rb') as fp:
		data = fp.read().split(b'\r\n')[1:]
	trainY = np.asarray(data).astype(float).reshape(len(data), 1)
	#print(trainY.shape)			#(20000,)
	trainY[trainY == 0] = -1.0
	return trainY

def getTestX(path):
	tempX = []
	testX = []
	data = []
	with open(path, 'rb') as fp:
		data = fp.read().split(b'\r\n')[1:]
	dataLen = int(len(data))
	for i in range(dataLen):
		temp = data[i].split(b',')
		temp = handleX(temp)
		tempX.append(temp)
	tempX = np.concatenate((tempX,np.ones((len(tempX),1))),axis=1)
	testX = np.asarray(tempX).astype(float)
	#print(testX.shape)			#(10000, 31)
	return testX

def saveAnswer(name, ans):
	output = open(name,'w')
	print('id,Value',file=output)
	for i in range(len(ans)):
		print('id_%d,%d'%(i, int(ans[i])), file=output)
	return

def getDTkind(path):
	data = []
	table = []
	with open(path, 'rb') as fp:
		data = fp.read().split(b'\r\n')[1:]
	dataLen = int(len(data))
	for i in range(dataLen):
		temp = data[i].split(b',')[1:4]
		temp = temp[0]+temp[1]+temp[2]
		if temp not in table:
			table.append(temp)
	table = np.asarray(table).astype(float)
	print(np.sort(table))
	return

if __name__ == '__main__':
	dataPath = './data'
	'''
	trainX = getTrainX(dataPath + '/train_x.csv')
	trainY = getTrainY(dataPath + '/train_y.csv')
	for i in range(len(trainX)):
		if (trainX[i, 3] == 1 or trainX[i, 10] == 1) and trainY[i] == 1:
			print(i)
		elif trainY[i] == 1:
			print("->%d"%(i)) 
	exit()
	'''
	trainX = getTrainX(dataPath + '/train_x.csv')
	trainY = getTrainY(dataPath + '/train_y.csv')
	testX = getTestX(dataPath + '/test_x.csv')
	'''
	N = len(trainX)
	count = 0
	for i in range(N):
		#if trainX[N - i - 1, 3] == 1.0 or trainX[N - i - 1, 10] == 1.0 or trainX[N - i - 1, 8] == 1.0 or trainX[N - i - 1, 9] == 1.0:
			#print('%d\t%d'%(i, trainY[i]))
		if trainY[i] == 1:
			count += 1
	print(count)
	exit()
	N = len(testX)
	for i in range(N):
		if testX[N - i - 1, 3] == 1.0 or testX[N - i - 1, 10] == 1.0 or testX[N - i - 1, 8] == 1.0 or testX[N - i - 1, 9] == 1.0:
			#print('%d\t%d'%(i, trainY[i]))
			count += 1
	print(count)
	exit()
	'''
	np.save('trainX.npy', trainX)
	#np.save('trainY.npy', trainY)
	np.save('testX.npy', testX)
	
	#trainX = getDTkind(dataPath + '/train_x.csv')