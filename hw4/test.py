from util import *
from myModel import *
import sys

batch_counter = 0
batch_size = 256

def generator(testX, dictionary):
	global batch_counter
	while True:
		if batch_counter + batch_size >= len(testX):
			X = makeWord2Vec(testX[batch_counter:], dictionary)
			batch_counter = 0
			yield (X)
		else:
			X = makeWord2Vec(testX[batch_counter:batch_counter+batch_size], dictionary)
			batch_counter += batch_size
			yield (X)

def predict(modelName, predictName, testX, dictionary):
	global batch_counter 
	batch_counter = 0
	model = load_model(modelName)
	y_pred = model.predict_generator(generator(testX, dictionary), steps=len(testX)//batch_size+1)
	y_pred = np.array(y_pred).reshape(-1, 1)
	y_pred[y_pred >= 0.5] = 1
	y_pred[y_pred < 0.5] = 0
	label = np.arange(y_pred.shape[0]).reshape(-1, 1)
	ans = np.concatenate([label, y_pred], axis=1).astype(int)
	np.savetxt(predictName, ans, fmt="%s", header='id,label', comments='', delimiter=",")

if __name__ == '__main__':
	testX = getTestX(sys.argv[1], sys.argv[2])
	dictionary = getDictionary()
	predict('./modelDir/model8_32-00008-0.74817.h5', sys.argv[3], testX, dictionary)
	