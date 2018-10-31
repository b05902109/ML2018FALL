from hw2_train import *
import sys

bound = 0.5
ExtendAllPay = [0,14,81,82,83,84,85,86,87,88,89,90,91,92,92]

def sigmoid(X):
	ans = 1.0/(1.0 + np.exp(-X))
	return ans

def accuracy(yhat, y):
	#return np.sum(yhat == y)/len(y)
	return np.sum(yhat == y)/len(y)

def printNum(y, typeName):
	y[y == -1] = 0
	temp = np.sum(y)
	print('%s:\t1: %4d, 0: %4d'%(typeName, temp, len(y)-temp))

def generative(trainX, trainY, testX):
	c1X = trainX[np.where(trainY == 1)[0]]
	c2X = trainX[np.where(trainY == -1)[0]]
	mean1 = np.mean(c1X, axis=0)
	mean2 = np.mean(c2X, axis=0)
	#print(c1X.shape, mean1.shape)	#(15555, 34) (34,)
	p1 = len(c1X)/len(trainY)
	p2 = 1.0 - p1
	cov = np.cov(c1X, rowvar=False) * p1 + np.cov(c2X, rowvar=False) * p2
	covInv = np.linalg.pinv(cov)
	yhat = (trainX.dot(covInv).dot(mean1-mean2) - 0.5*(mean1.dot(covInv).dot(mean1)) + 0.5*mean2.dot(covInv).dot(mean2) + np.log(len(c1X)/len(c2X)))
	ypred = (testX.dot(covInv).dot(mean1-mean2) - 0.5*(mean1.dot(covInv).dot(mean1)) + 0.5*mean2.dot(covInv).dot(mean2) + np.log(len(c1X)/len(c2X)))
	yhat = sigmoid(yhat).reshape((len(trainY), 1))
	ypred = sigmoid(ypred).reshape((len(testX), 1))
	yhat[yhat >= 0.5] = 1
	yhat[yhat < 0.5] = -1
	ypred[ypred >= 0.5] = 1
	ypred[ypred < 0.5] = 0
	print('Accuracy: %f'%(accuracy(yhat, trainY)))
	printNum(yhat, 'train')
	printNum(ypred, 'test')
	return ypred

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Need Name.')
		exit()
	#trainX = np.load('trainX_hasW0_ExtendPay.npy')[:,:-1]
	#trainY = np.load('trainY.npy')
	#testX = np.load('testX_hasW0_ExtendPay.npy')[:,:-1]
	trainX = getTrainX(sys.argv[1])[:,:-1]
	trainY = getTrainY(sys.argv[2])
	testX = getTestX(sys.argv[3])[:,:-1]

	#Normalize
	for i in range(len(trainX[0])):
		if i in ExtendAllPay:
			xMean = trainX[:,i].mean(axis=0)
			xStd = trainX[:,i].std(axis=0)
			#print(i, xMean, xStd)
			trainX[:, i] = (trainX[:, i] - xMean) / xStd
			testX[:, i] = (testX[:, i] - xMean) / xStd
	
	y = generative(trainX, trainY, testX)
	saveAnswer(sys.argv[4], y)
