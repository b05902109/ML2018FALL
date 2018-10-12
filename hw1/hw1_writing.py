from hw1_train import *
import numpy as np
import matplotlib.pyplot as plt

upper = 122
lower = 0

def problem1(trainX, trainY):
	wLen = len(trainX[0])
	lr_array = [0.0001, 0.001, 0.01, 0.1]
	iteration = 300
	for lr in lr_array:
		#print(lr)
		err = []
		w = np.ones((wLen, 1))
		w[-1] = 1
		w_lr = np.zeros((wLen, 1))
		#err.append(countErr(w, trainX, trainY))
		for i in range(iteration):
			w_grad = np.zeros((wLen, 1))
			w_grad = w_grad - 2 * np.dot(trainX.T, (trainY - np.dot(trainX, w)))
			#'''
			w_lr = w_lr + np.power(w_grad, 2)
			w = w - lr/np.sqrt(w_lr) * w_grad
			#'''
			#w = w - lr * w_grad
			err.append(countErr(w, trainX, trainY))
		plt.plot(range(iteration), err, label="learn rate %f"%(lr))
		#print(len(err))
	plt.legend(loc='upper right')
	plt.xlabel('iteration')
	plt.ylabel('RMSE')
	plt.show()

def getTrainDataAll(trainDataPath):
	trainData = []
	trainX = []
	trainY = []
	All = []
	for i in range(18):
		All.append([])
	with open(trainDataPath, 'rb') as fp:
		trainData = fp.read().split(b'\r\n')[1:]
	dataLen = int(len(trainData)/18)
	for i in range(dataLen):
		for j in range(18):
			if j == 10:
				continue
			All[j] += trainData[i*18+j].split(b',')[3:]
	for i in range(len(All[0])-9):
		if i % 20*24 > 20*24-9:
			continue
		temp = []
		for j in range(18):
			if j == 10:
				continue
			temp += All[j][i:i+9]
		trainX.append(temp)
		trainY.append(All[9][i+9])
	trainX = np.concatenate((trainX,np.ones((len(trainX),1))),axis=1)
	trainX = np.asarray(trainX).astype(float)
	trainY = np.asarray(trainY).astype(float)
	trainY = trainY.reshape((len(trainY), 1))
	#print(trainX.shape, trainY.shape)
	return trainX, trainY

def getTestDataAll(path):
	testData = []
	tempX = []
	testX = []
	with open(path, 'rb') as fp:
		testData = fp.read().split(b'\r\n')
	#print(len(testData)/18) 260
	for i in range(int(len(testData)/18)):
		tempX.append([])
	for i in range(len(testData)):
		if i % 18 == 10:
			continue
		tempX[int(i/18)] += testData[i].split(b',')[2:]
	tempX = np.concatenate((tempX,np.ones((len(tempX),1))),axis=1)
	testX = np.asarray(tempX).astype(float)
	return testX

def problem2(trainDataPath, testDataPath):
	
	xtrainAll, ytrainAll = getTrainDataAll(trainDataPath)
	xtrainPm2dot5, ytrainPm2dot5 = getTrainDataPm2dot5(trainDataPath)
	
	wAll = trainW(xtrainAll, ytrainAll)
	wPm2dot5 = trainW(xtrainPm2dot5, ytrainPm2dot5)
	xtestAll = getTestDataAll(testDataPath)
	xtestPm2dot5 = getTestDataPm2dot5(testDataPath)
	np.save('xtestAll.npy', xtestAll)
	np.save('xtestPm2dot5.npy', xtestPm2dot5)
	np.save('wAll.npy', wAll)
	np.save('wPm2dot5.npy', wPm2dot5)
	'''
	xtestAll = np.load('xtestAll.npy')
	xtestPm2dot5 = np.load('xtestPm2dot5.npy')
	wAll = np.load('wAll.npy')
	wPm2dot5 = np.load('wPm2dot5.npy')
	#print(xtrainAll.shape, ytrainAll.shape, xtestAll.shape, wAll.shape)					#(5751, 154) (5751, 1) (260, 154) (154, 1)
	#print(xtrainPm2dot5.shape, ytrainPm2dot5.shape, xtestPm2dot5.shape, wPm2dot5.shape)	#(5505, 10) (5505, 1) (260, 10) (10, 1)
	#print(xtestAll)
	'''
	with open('hw1_Problem2_AllFeature.csv','w') as output:
		#xtestAll = map(float,xtestAll)
		#wAll = map(float,wAll)
		y = np.dot(xtestAll, wAll)
		print('id,value',file=output)
		for i in range(len(y)):
			print('id_%d,%f'%(i,round(float(y[i]), 0)), file=output)
	with open('hw1_Problem2_Pm2dot5Feature.csv','w') as output:
		#xtestPm2dot5 = map(float,xtestPm2dot5)
		#wPm2dot5 = map(float,wPm2dot5)
		y = np.dot(xtestPm2dot5, wPm2dot5)
		print('id,value',file=output)
		for i in range(len(y)):
			print('id_%d,%f'%(i,round(float(y[i]), 0)), file=output)
	
	print(countErr(wAll, xtrainAll, ytrainAll))				#22.6044837272
	print(countErr(wPm2dot5, xtrainPm2dot5, ytrainPm2dot5))	#7.03614825745

def problem3(path):
	trainX, trainY = getTrainDataPm2dot5(path)
	testX = np.load('xtestPm2dot5.npy')
	wLen = len(trainX[0])
	lam_list = [0, 3, 6, 9]
	for lam in lam_list:
		lr = 1
		w = np.zeros((wLen, 1))
		w[-1] = 1
		w_lr = np.zeros((wLen, 1))
		iteration = 10000
		for i in range(iteration):
			w_grad = np.zeros((wLen, 1))
			w_grad = w_grad - 2 * np.dot(trainX.T, (trainY - np.dot(trainX, w))) + (10**lam) * w
			w_lr = w_lr + np.power(w_grad, 2)
			w = w - lr/np.sqrt(w_lr) * w_grad
		print('hw1_Problem3_lamba: 10e%d, trainloss: %f'%(lam, countErr(w, trainX, trainY)))
		with open('lamba_10e%d.csv'%(lam),'w') as output:
			y = np.dot(testX, w)
			print('id,value',file=output)
			for i in range(len(y)):
				print('id_%d,%f'%(i,round(float(y[i]), 0)), file=output)
	'''
	hw1_Problem3_lamba: 10e0, trainloss: 5.533821
	hw1_Problem3_lamba: 10e3, trainloss: 5.541666
	hw1_Problem3_lamba: 10e6, trainloss: 8.376644
	hw1_Problem3_lamba: 10e9, trainloss: 32.849194
	'''

if __name__ == '__main__':
	trainDataPath = '../../ML_Report/hw1/data/train.csv'
	testDataPath = '../../ML_Report/hw1/data/test.csv'
	#trainX, trainY = getTrainDataPm2dot5(trainDataPath)
	
	#problem1(trainX, trainY)
	problem2(trainDataPath, testDataPath)
	problem3(trainDataPath)