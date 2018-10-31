from hw2_train import *
import sys

bound = 0
ExtendAllPay = [0,14,81,82,83,84,85,86,87,88,89,90,91,92,92]

def sigmoid(X):
	ans = 1.0/(1.0 + np.exp(-X))
	return ans

def logistic(trainX, trainY, testX):
	#np.random.seed(7122)
	N = len(trainX)
	w = np.linalg.pinv(trainX).dot(trainY)
	#w = np.zeros((len(trainX[0]), 1))
	#w[-1] = 1
	#w = np.random.rand(len(trainX[0]), 1)
	b = 0
	w_grad_sum = np.zeros((len(trainX[0]), 1))
	b_grad_sum = 0
	#rate = 5e-6
	rate = 1
	iteration = 5000
	#lam = 5e-6
	for i in range(iteration):
		#print(i)
		#take = np.random.randint(0, N, 2000)
		#MLT logistic groups
		#take = np.asarray(range(2000*(i%10), 2000*(i%10 + 1)))
		#takeX = trainX[take, :]
		#takeY = trainY[take]
		takeX = trainX
		takeY = trainY
		L = sigmoid(-takeY * takeX.dot(w))		#(20000, 1)
		grad = -np.dot((takeY * takeX).T, L) / 2000 	#(29, 1)
		
		for i in range(len(grad)):
			if grad[i] == 0:
				grad[i] = 0.00001
			#print('%d %f'%(i, grad[i]))
		#exit()
		
		w_grad_sum += (grad)**2						#(29, 1)
		w -= rate / np.sqrt(w_grad_sum) * grad
		'''
		#MLT logistic
		L = sigmoid(-trainY * trainX.dot(w))		#(20000, 1)
		grad = -np.dot((trainY * trainX).T, L) / N 	#(29, 1)
		w_grad_sum += (grad)**2						#(29, 1)
		w -= rate / np.sqrt(w_grad_sum) * grad
		#w -= rate * grad
		#print('iter: %4d, Loss: %f'%(i, loss(trainX.dot(w), trainY)))
		'''
		'''
		#ML logistic
		yhat = trainX.dot(w) + b
		temp = trainY - sigmoid(yhat)
		
		w_grad = -np.sum(trainX.T.dot(temp))
		b_grad = -np.sum(temp)

		w_grad_sum += w_grad**2
		b_grad_sum += b_grad**2
		
		w -= rate * w_grad / np.sqrt(w_grad_sum)
		b -= rate * b_grad / np.sqrt(b_grad_sum)
		'''
		#print('iter: %4d, Accuracy: %f'%(i, loss(trainX.dot(w) + b, trainY)))
	print('Accuracy: %f'%(accuracy(trainX.dot(w), trainY)))
	#print('Accuracy: %f'%(accuracy(testX.dot(w), testY)))
	'''
	np.save(sys.argv[1][:-3] + 'npy', w)
	printNum(np.dot(trainX, w), 'train')
	printNum(np.dot(testX, w), 'test')
	y = np.dot(testX, w)
	y[y >= bound] = 1
	y[y < bound] = 0
	saveAnswer(sys.argv[1], y)
	'''
	return w
'''
def BAG(trainX, trainY, testX):
	valid = 0
	modelM = 1
	W = []
	while valid < modelM:
		break_flag = 0
		N = len(trainX)
		w = np.linalg.pinv(trainX).dot(trainY)
		w_grad_sum = np.zeros((len(trainX[0]), 1))
		rate = 1
		iteration = 5000
		for i in range(iteration):
			take = np.random.randint(0, N, 2000)
			#MLT logistic groups
			#take = np.asarray(range(2000*(i%10), 2000*(i%10 + 1)))
			takeX = trainX[take, :]
			takeY = trainY[take]
			L = sigmoid(-takeY * takeX.dot(w))		#(20000, 1)
			grad = -np.dot((takeY * takeX).T, L) / 2000 	#(29, 1)
			w_grad_sum += (grad)**2						#(29, 1)
			if i == 0:
				for j in range(len(w_grad_sum)):
					if w_grad_sum[j] == 0:
						break_flag = 1
						break
			w -= rate / np.sqrt(w_grad_sum) * grad
		if break_flag:
			continue
		else:
			temp = []
			for i in range(len(trainX[0])):
				temp.append(w[i])
			W.append(temp)
			valid += 1
	W = np.asarray(W)
	yhat = np.sum(np.dot(trainX, W.T), axis=0) / modelM
	ypred = np.sum(np.dot(testX, W.T), axis=0) / modelM
	yhat[yhat >= 0.5] = 1
	yhat[yhat < 0.5] = -1
	ypred[ypred >= 0.5] = 1
	ypred[ypred < 0.5] = -1
	print('Accuracy: %f'%(accuracy(ypred, trainY)))
	np.save(sys.argv[1][:-3] + 'npy', w)
	printNum(yhat, 'train')
	printNum(ypred, 'test')
	ypred[ypred == -1] = 0
	saveAnswer(sys.argv[1], ypred)
	return
'''
def printNum(y, typeName):
	y[y >= bound] = 1
	y[y < bound] = 0
	temp = np.sum(y)
	print('%s:\t1: %4d, 0: %4d'%(typeName, temp, len(y)-temp))

def accuracy(yhat, y):
	yhat[yhat >= bound] = 1
	yhat[yhat < bound] = -1
	#return np.sum(yhat == y)/len(y)
	return np.sum(np.sign(yhat) == y)/len(y)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Need Name.')
		exit()
	#trainX = np.load('trainX_hasW0_ExtendAllPay.npy')
	#trainY = np.load('trainY.npy')
	#testX = np.load('testX_hasW0_ExtendAllPay.npy')
	trainX = getTrainX(sys.argv[1])
	trainY = getTrainY(sys.argv[2])
	testX = getTestX(sys.argv[3])
	#Normalize
	for i in range(len(trainX[0])):
		if i in ExtendAllPay:
			xMean = trainX[:,i].mean(axis=0)
			xStd = trainX[:,i].std(axis=0)
			#print(i, xMean, xStd)
			trainX[:, i] = (trainX[:, i] - xMean) / xStd
			testX[:, i] = (testX[:, i] - xMean) / xStd
	#[18,19,25,26,30,32]
	'''
	for i in range(15, 21):
		print(np.unique(trainX[:, i]))
		print(np.unique(testX[:, i]))
		print('------')
	exit()
	'''
	
	#np.random.seed(7122)
	#w = logistic(trainX, trainY, testX)
	#np.save('train_bestW.npy', w)
	w = np.load('train_bestW.npy')
	print('Accuracy: %f'%(accuracy(np.dot(trainX, w), trainY)))
	printNum(np.dot(trainX, w), 'train')
	printNum(np.dot(testX, w), 'test')
	y = np.dot(testX, w)
	y[y >= bound] = 1
	y[y < bound] = 0
	saveAnswer(sys.argv[4], y)

	'''
	for out in ExtendAllPay:
		print('out %d'%(out))
		w = logistic(np.delete(trainX, out, 1), trainY, np.delete(testX, out, 1))
	exit()
	'''
	'''
	w = np.load('adaLogistic_08_r7122_3000_10vote.npy')[:, 8].T
	print('Accuracy: %f'%(accuracy(np.dot(trainX, w), trainY)))
	printNum(np.dot(trainX, w), 'train')
	printNum(np.dot(testX, w), 'test')
	y = np.dot(testX, w)
	y[y >= bound] = 1
	y[y < bound] = 0
	saveAnswer(sys.argv[1], y)
	exit()	
	
	w = np.array([])
	np.random.seed(7122)
	modelN = 10
	for i in range(modelN):
		print('----- model %d -----'%(i))
		modelTake = np.random.randint(0, len(trainX), 15000)
		if i == 0:
			w = logistic(trainX[modelTake], trainY[modelTake], testX)
		else:
			w = np.c_[w, (logistic(trainX[modelTake], trainY[modelTake], testX))]
	print('----- sum model -----')
	np.save(sys.argv[1][:-3] + 'npy', w)
	print('Accuracy: %f'%(accuracy(np.sum(np.sign(np.dot(trainX, w)), axis=1).reshape((len(trainY), 1)), trainY)))
	printNum(np.sum(np.sign(np.dot(trainX, w)), axis=1).reshape((len(trainY), 1)), 'train')
	printNum(np.sum(np.sign(np.dot(testX, w)), axis=1).reshape((len(testX), 1)), 'test')
	y = np.sum(np.sign(np.dot(testX, w)), axis=1).reshape((len(testX), 1))
	y[y >= bound] = 1
	y[y < bound] = 0
	saveAnswer(sys.argv[1], y)
	#logistic(np.delete(trainX, 16, 1), trainY, np.delete(testX, 16, 1))
	#BAG(trainX, trainY, testX)
	'''
	
'''
>python hw2_logistic.py adaLogistic_01_ExtendPay.csv
Accuracy: 0.811550
train:  1: 1536, 0: 18464
test:   1:  706, 0: 9294
>python hw2_logistic.py adaLogistic_02_ExtendPay_group.csv
Accuracy: 0.811750
train:  1: 1552, 0: 18448
test:   1:  721, 0: 9279
>python hw2_logistic.py adaLogistic_03_ExtendPay_randomgroup.csv	#7122
Accuracy: 0.813600
train:  1: 1759, 0: 18241
test:   1:  828, 0: 9172
>python hw2_logistic.py adaLogistic_04_otherPay_randomgroup.csv
Accuracy: 0.813550
train:  1: 1756, 0: 18244
test:   1:  830, 0: 9170

----- delete 0 feature -----
Accuracy: 0.811300
train:  1: 1581, 0: 18419
test:   1:  732, 0: 9268
----- delete 1 feature -----
Accuracy: 0.811900
train:  1: 1545, 0: 18455
test:   1:  715, 0: 9285
----- delete 2 feature -----
Accuracy: 0.811550
train:  1: 1540, 0: 18460
test:   1:  716, 0: 9284
----- delete 3 feature -----
Accuracy: 0.811800
train:  1: 1556, 0: 18444
test:   1:  721, 0: 9279
----- delete 4 feature -----
Accuracy: 0.811350
train:  1: 1542, 0: 18458
test:   1:  714, 0: 9286
----- delete 5 feature -----
Accuracy: 0.811850
train:  1: 1547, 0: 18453
test:   1:  713, 0: 9287
----- delete 6 feature -----
Accuracy: 0.811750
train:  1: 1554, 0: 18446
test:   1:  723, 0: 9277
----- delete 7 feature -----
Accuracy: 0.811700
train:  1: 1553, 0: 18447
test:   1:  722, 0: 9278
----- delete 8 feature -----
Accuracy: 0.811700
train:  1: 1551, 0: 18449
test:   1:  721, 0: 9279
----- delete 9 feature -----
Accuracy: 0.811850
train:  1: 1554, 0: 18446
test:   1:  721, 0: 9279
----- delete 10 feature -----
Accuracy: 0.811700
train:  1: 1553, 0: 18447
test:   1:  721, 0: 9279
----- delete 11 feature -----
Accuracy: 0.811800
train:  1: 1564, 0: 18436
test:   1:  725, 0: 9275
----- delete 12 feature -----
Accuracy: 0.811400
train:  1: 1533, 0: 18467
test:   1:  709, 0: 9291
----- delete 13 feature -----
Accuracy: 0.811750
train:  1: 1552, 0: 18448
test:   1:  721, 0: 9279
----- delete 14 feature -----
Accuracy: 0.811400
train:  1: 1549, 0: 18451
test:   1:  726, 0: 9274
----- delete 15 feature -----
Accuracy: 0.796250
train:  1: 1056, 0: 18944
test:   1:  500, 0: 9500
----- delete 16 feature -----
Accuracy: 0.812250
train:  1: 1524, 0: 18476
test:   1:  713, 0: 9287
----- delete 17 feature -----
Accuracy: 0.811900
train:  1: 1555, 0: 18445
test:   1:  725, 0: 9275
----- delete 18 feature -----
Accuracy: 0.812100
train:  1: 1551, 0: 18449
test:   1:  719, 0: 9281
----- delete 19 feature -----
Accuracy: 0.812050
train:  1: 1574, 0: 18426
test:   1:  724, 0: 9276
----- delete 20 feature -----
Accuracy: 0.811700
train:  1: 1557, 0: 18443
test:   1:  724, 0: 9276
----- delete 21 feature -----
Accuracy: 0.811700
train:  1: 1551, 0: 18449
test:   1:  715, 0: 9285
----- delete 22 feature -----
Accuracy: 0.811650
train:  1: 1548, 0: 18452
test:   1:  720, 0: 9280
----- delete 23 feature -----
Accuracy: 0.811650
train:  1: 1548, 0: 18452
test:   1:  720, 0: 9280
----- delete 24 feature -----
Accuracy: 0.811700
train:  1: 1549, 0: 18451
test:   1:  722, 0: 9278
----- delete 25 feature -----
Accuracy: 0.812000
train:  1: 1559, 0: 18441
test:   1:  720, 0: 9280
----- delete 26 feature -----
Accuracy: 0.812100
train:  1: 1561, 0: 18439
test:   1:  722, 0: 9278
----- delete 27 feature -----
Accuracy: 0.811950
train:  1: 1572, 0: 18428
test:   1:  719, 0: 9281
----- delete 28 feature -----
Accuracy: 0.811550
train:  1: 1550, 0: 18450
test:   1:  727, 0: 9273
----- delete 29 feature -----
Accuracy: 0.811800
train:  1: 1559, 0: 18441
test:   1:  724, 0: 9276
----- delete 30 feature -----
Accuracy: 0.812200
train:  1: 1555, 0: 18445
test:   1:  724, 0: 9276
----- delete 31 feature -----
Accuracy: 0.811550
train:  1: 1556, 0: 18444
test:   1:  716, 0: 9284
----- delete 32 feature -----
Accuracy: 0.812150
train:  1: 1560, 0: 18440
test:   1:  724, 0: 9276
----- delete 33 feature -----
Accuracy: 0.811400
train:  1: 1535, 0: 18465
test:   1:  710, 0: 9290
'''
'''
out 0
Accuracy: 0.821950
out 14
Accuracy: 0.821550
out 81
Accuracy: 0.821650
out 82
Accuracy: 0.821600
out 83
Accuracy: 0.821650
out 84
Accuracy: 0.821650
out 85
Accuracy: 0.821600
out 86
Accuracy: 0.821700
out 87
Accuracy: 0.821500
out 88
Accuracy: 0.821400
out 89
Accuracy: 0.821700
out 90
Accuracy: 0.821650
out 91
Accuracy: 0.821650
out 92
Accuracy: 0.821450
out 92
Accuracy: 0.821450
'''
