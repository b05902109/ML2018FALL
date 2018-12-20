from util import *
from myModel import *
import sys
from keras.callbacks import ModelCheckpoint, History

batch_counter = 0
batch_counter_valid = 0
epoch = 10
batch_size = 32

def generator(trainX, trainY, dictionary):
	global batch_counter
	while True:
		if batch_counter + batch_size >= len(trainX):
			X = makeWord2Vec(trainX[batch_counter:], dictionary)
			Y = np.array(trainY[batch_counter:])
			batch_counter = 0
			yield (X, Y)
		else:
			X = makeWord2Vec(trainX[batch_counter:batch_counter+batch_size], dictionary)
			Y = np.array(trainY[batch_counter:batch_counter+batch_size])
			batch_counter += batch_size
			yield (X, Y)

def generator_valid(trainX_valid, trainY_valid, dictionary):
	global batch_counter_valid
	while True:
		if batch_counter_valid + batch_size >= len(trainX_valid):
			X = makeWord2Vec(trainX_valid[batch_counter_valid:], dictionary)
			Y = np.array(trainY_valid[batch_counter_valid:])
			batch_counter_valid = 0
			yield (X, Y)
		else:
			X = makeWord2Vec(trainX_valid[batch_counter_valid:batch_counter_valid+batch_size], dictionary)
			Y = np.array(trainY_valid[batch_counter_valid:batch_counter_valid+batch_size])
			batch_counter_valid += batch_size
			yield (X, Y)


if __name__ == '__main__':
	trainX = getTrainX(sys.argv[1])
	testX = getTestX(sys.argv[3])
	dictionary = trainWord2Vec(trainX, testX, sys.argv[4])
	trainY = getTrainY(sys.argv[2])
	trainXLen = len(trainX)
	valid_n = trainXLen // 10
	train_n = trainXLen - valid_n
	#valid_n = 0
	trainX_valid = trainX[:valid_n]
	trainY_valid = trainY[:valid_n]
	
	model = model8(lineLengthMAX, Word2Vec_dim)
	hist = 	History()
	check_save = ModelCheckpoint("modelDir/model3_32-{epoch:05d}-{val_acc:.5f}.h5", monitor='val_acc', save_best_only=True)
	model.fit_generator(generator(trainX[valid_n:], trainY[valid_n:], dictionary), 
		steps_per_epoch=train_n//batch_size + 1, 
		epochs=epoch, 
		validation_data=generator_valid(trainX_valid, trainY_valid, dictionary), 
		validation_steps=valid_n//batch_size + 1, 
		callbacks=[check_save, hist]) 
	model.save('./modelDir/model8_32-00008-0.74817.h5')