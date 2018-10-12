import sys
import numpy as np
import math
#import matplotlib.pyplot as plt

upper = 120
lower = 2
wLen = 9
#bound = [lower, 9, 13, 16, 20, 23, 26, 30, 35, 42, upper]	#B
#bound = [lower,5,25,50,53,55,57,60,63,65,67,70, upper]	#B2
#boundclass = len(bound) - 1 #10
'''
def countBound(data):
	print(data.shape)
	wLen = len(data[0])
	
	group = 15
	avg = []
	for i in range(len(data)):
		avg.append(np.sum(data[i][0:wLen])/wLen)
	avg.sort()
	avg = np.asarray(avg).astype(int)
	groupSize = int(len(avg)/group)
	for i in range(group):
		print(avg[groupSize*i])
	print('---')
	print(avg)
	
	print(data[0])
	pm2dot5 = []
	pm10 = []
	for iter, val in enumerate(data):
		for i in range(wLen):
			if i < 9:
				if val[i] < 2 or val[i] > 120:
					print(iter)
				pm2dot5.append(val[i])
			elif i < 18:
				if val[i] < 2 or val[i] > 150:
					print(iter)
				pm10.append(val[i])
	#pm2dot5 = list(set(pm2dot5))
	#pm10 = list(set(pm10))
	#pm2dot5.sort()
	#pm10.sort()
	#print(pm2dot5)
	#print(pm10)
	#plt.hist(pm10, bins=200)
	#plt.show()
	exit()

def getTrainDataPm2dot5and10(path):
	trainData = []
	tempX = []
	tempY = []
	trainX = []
	trainY = []
	pm2dot5 = []
	pm10 = []
	with open(path, 'rb') as fp:
		trainData = fp.read().split(b'\r\n')[1:]
	dataLen = int(len(trainData)/18)
	#print(dataLen) 240
	for i in range(dataLen):
		pm2dot5 += trainData[i*18+9].split(b',')[3:]
		pm10 += trainData[i*18+8].split(b',')[3:]
	#print(len(pm2dot5)) 5760
	#exit(0)
	for i in range(len(pm2dot5)-9):
		if i % 20*24 > 20*24-9:
			continue
		tempX.append(pm2dot5[i:i+9] + pm10[i:i+9])
		tempY.append(pm2dot5[i+9])
	#print(trainX[0], trainY[0])
	for i in range(len(tempX)):
		able = 1

		for j in range(9):
			if able == 0:
				break
			if float(tempX[i][j]) < lower or float(tempX[i][j]) > upper:
				able = 0
		if float(tempY[i]) < lower or float(tempY[i]) > upper:
			able = 0
		
		if able == 1:
			trainX.append(tempX[i])
			trainY.append(tempY[i])
	trainX = np.concatenate((trainX,np.ones((len(trainX),1))),axis=1)
	trainX = np.asarray(trainX).astype(float)
	trainY = np.asarray(trainY).astype(float)
	trainY = trainY.reshape((len(trainY), 1))
	#countBound(trainX)
	return trainX, trainY
'''
def getTrainDataPm2dot5(path):
	trainData = []
	tempX = []
	tempY = []
	trainX = []
	trainY = []
	pm2dot5 = []
	with open(path, 'rb') as fp:
		trainData = fp.read().split(b'\r\n')[1:]
	dataLen = int(len(trainData)/18)
	#print(dataLen) 240
	for i in range(dataLen):
		pm2dot5 += trainData[i*18+9].split(b',')[3:]
	#print(len(pm2dot5)) 5760
	#exit(0)
	for i in range(len(pm2dot5)-9):
		if i % 20*24 > 20*24-9:
			continue
		tempX.append(pm2dot5[i:i+9])
		tempY.append(pm2dot5[i+9])
	#print(trainX[0], trainY[0])
	for i in range(len(tempX)):
		able = 1
		for j in range(9):
			if able == 0:
				break
			if float(tempX[i][j]) < lower or float(tempX[i][j]) > upper:
				able = 0
		if float(tempY[i]) < lower or float(tempY[i]) > upper:
			able = 0
		if able == 1:
			trainX.append(tempX[i])
			trainY.append(tempY[i])
	trainX = np.concatenate((trainX,np.ones((len(trainX),1))),axis=1)
	trainX = np.asarray(trainX).astype(float)
	trainY = np.asarray(trainY).astype(float)
	trainY = trainY.reshape((len(trainY), 1))
	#countBound(trainX)
	return trainX, trainY

def getTestDataPm2dot5(path):
	testData = []
	tempX = []
	testX = []
	pm2dot5 = []
	pm10 = []
	with open(path, 'rb') as fp:
		testData = fp.read().split(b'\r\n')
	dataLen = int(len(testData)/18)
	#print(dataLen)	260
	for i in range(dataLen):
		pm2dot5 += testData[i*18+9].split(b',')[2:]
	#print(len(pm2dot5))	2340
	for i in range(int(len(pm2dot5)/9)):
		tempX.append(pm2dot5[i*9:(i+1)*9])
	tempX = np.concatenate((tempX,np.ones((len(tempX),1))),axis=1)
	testX = np.asarray(tempX).astype(float)
	return testX
'''
def getTestDataPm2dot5and10(path):
	testData = []
	tempX = []
	testX = []
	pm2dot5 = []
	pm10 = []
	with open(path, 'rb') as fp:
		testData = fp.read().split(b'\r\n')
	dataLen = int(len(testData)/18)
	#print(dataLen)	260
	for i in range(dataLen):
		pm2dot5 += testData[i*18+9].split(b',')[2:]
		pm10 += testData[i*18+8].split(b',')[2:]
	#print(len(pm2dot5))	2340
	for i in range(int(len(pm2dot5)/9)):
		tempX.append(pm2dot5[i*9:(i+1)*9] + pm10[i*9:(i+1)*9])
	tempX = np.concatenate((tempX,np.ones((len(tempX),1))),axis=1)
	testX = np.asarray(tempX).astype(float)
	return testX
'''
def trainW(trainX, trainY):
	wLen = len(trainX[0])
	lr = 1
	w = np.zeros((wLen, 1))
	w[-1] = 1
	w_lr = np.zeros((wLen, 1))
	iteration = 10000
	for i in range(iteration):
		w_grad = np.zeros((wLen, 1))
		w_grad = w_grad - 2 * np.dot(trainX.T, (trainY - np.dot(trainX, w)))
		w_lr = w_lr + np.power(w_grad, 2)
		w = w - lr/np.sqrt(w_lr) * w_grad
	return w
'''
def trainWWithBound(trainX, trainY):
	wLen = len(trainX[0])
	lr = 1
	iteration = 10000
	w = np.zeros((boundclass, wLen, 1))
	w[:,-1,0] = 1
	w_lr = np.zeros((boundclass, wLen, 1))
	#trainXB = np.zeros((10, 1))
	#trainYB = np.zeros((10, 1))
	trainXB = []
	trainYB = []
	for i in range(boundclass):
		trainXB.append([])
		trainYB.append([])
	
	for i in range(len(trainX)):
		mean = np.sum(trainX[i][0:9])/9
		for j in range(boundclass):
			if mean <= bound[j+1]:
				#print(mean)
				#np.append(trainXB[j], trainX[i], axis=0)
				#np.append(trainYB[j], trainY[i], axis=0)
				trainXB[j].append(trainX[i])
				trainYB[j].append(trainY[i])
				break
	#print(len(trainXB[0]))
	for j in range(boundclass):
		print(j)
		tempX = np.array(trainXB[j])
		tempY = np.array(trainYB[j])
		for i in range(iteration):
			w_grad = np.zeros((wLen, 1))
			w_grad = w_grad - 2 * np.dot(tempX.T, (tempY - np.dot(tempX, w[j])))
			w_lr[j] = w_lr[j] + np.power(w_grad, 2)
			w[j] = w[j] - lr/np.sqrt(w_lr[j]) * w_grad
	return w

def countErr(w, trainX, trainY):
	err = np.sqrt(np.sum(np.power((np.dot(trainX, w) - trainY), 2))/len(trainX))
	return err

def loo(trainX, trainY):
	testX
	for pickOutIndex in range
'''
if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Need w name')
		exit(0)
	trainDataPath = '../../ML_Report/hw1/data/train.csv'
	testDataPath = '../../ML_Report/hw1/data/test.csv'
	#trainX, trainY = getTrainDataPm2dot5and10(trainDataPath)
	trainX, trainY = getTrainDataPm2dot5(trainDataPath)
	#countBound(trainX)
	#loo(trainX, trainY)
	#testX = getTestData(testDataPath)
	#exit(0)
	#w = trainWWithBound(trainX, trainY)
	w = trainW(trainX, trainY)
	#print(w.shape)
	np.save(sys.argv[1], w)
	#w = trainW(trainX[0:-100], trainY[0:-100])
	#print(countErr(w, trainX, trainY))
	