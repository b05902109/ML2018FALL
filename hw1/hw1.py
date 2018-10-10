from hw1_train import *
import numpy as np
import sys

wLen = 9
lower = 2
upper = 120
#bound = [lower, 9, 13, 16, 20, 23, 26, 30, 35, 42, upper]
#bound = [lower,5,25,50,53,55,57,60,63,65,67,70, upper]
#boundclass = len(bound) - 1 #10

def check(x):
	return ((x < lower) or (x > upper))

#w = np.load('ada_pm2dot5and10_5.npy')
#print(w)

w = np.load('best.npy')
#print(w)

#exit()
#testX = getTestDataPm2dot5and10(sys.argv[1])
#countBound(testX)
testX = getTestDataPm2dot5(sys.argv[1])
#print(testX.shape)
testXLen = len(testX)
for i in range(testXLen):
	for j in range(wLen):	#range(9):0.1....8
		if check(testX[i][8 - j]):
			if j == 0:
				testX[i][8 - j] = testX[i][8 - j - 4]
				'''
				if not check(testX[i][8 - j - 4]):
					#print(4)
					testX[i][8 - j] = testX[i][8 - j - 4]
				elif not check(testX[i][8 - j - 5]):
					print("=> 5")
					testX[i][8 - j] = testX[i][8 - j - 5]
				else:
					print("==> 3")
					testX[i][8 - j] = testX[i][8 - j - 3]
				'''
			else:
				testX[i][8 - j] = testX[i][8 - j + 1]
		'''
		if testX[i][j] < 2:
			testX[i][j] = 2
		if testX[i][j] > 110:
			testX[i][j] = 110
		'''
y = np.dot(testX, w)

'''
y = []
for i in range(testXLen):
	mean = np.sum(testX[i][0:wLen])/wLen
	#print(mean)
	for j in range(boundclass):
		if mean <= bound[j+1]:
			y.append(np.dot(testX[i], w[j]))
			break
#print(y)
#print(len(testX), len(y))
'''
output = open(sys.argv[2],'w')
print('id,value',file=output)
for i in range(len(testX)):
	print('id_%d,%f'%(i,round(float(y[i]), 0)), file=output)
