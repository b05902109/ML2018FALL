import numpy as np
import jieba
#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
#import re
import pickle

trainXPath = './data/trainX.csv'
trainYPath = './data/trainY.csv'
testXPath = './data/testX.csv'
Word2Vec_dim = 100
lineLengthMAX = 200

not_save = False

def makeWord2Vec(data_i, dictionary):
	data_o = []
	for line in data_i:
		line_new = []
		for word in line:
			try:
				line_new.append(dictionary[word])
			except:
				pass
		line_new = line_new[:lineLengthMAX]
		line_new = [np.zeros([Word2Vec_dim,]) for _ in range(lineLengthMAX-len(line_new))] + line_new
		data_o.append(line_new)
	data_o = np.array(data_o)
	return data_o

def handleXData(data, dictName):
	print('---- handleXData ----')
	data_fixed = []
	#jeiba set dictionary
	jieba.set_dictionary(dictName)
	stopword_set = set()
	with open('data/stopwords.txt','r', encoding='utf-8') as stopwords:
		for stopword in stopwords:
			stopword_set.add(stopword.strip('\n'))
	data = data.split('\n')
	del data[-1]
	del data[0]
	for num, line in enumerate(data):
		tmp_line = line.split(',', 1)[1].replace("? ", "?").replace("？ ", "?").replace("! ", "!").replace("！ ", "!").replace(" ", "").replace("  ", "").replace("ㄞ", "").replace("ㄚ", "")
		words = jieba.cut(tmp_line, cut_all=False)
		words_fix = []
		for word in words:
			if word[0] != 'b' or word[0] != 'B':
				words_fix.append(word)
		data_fixed.append(words_fix)
		if (num%10000) == 0:
			print('finish %d line'%(num))
	return data_fixed

def trainWord2Vec(data1, data2):
	print('---- train Word2Vec ----')
	dictionary = {}
	if not_save:
		with open('dictionary_%d.pkl'%(Word2Vec_dim), 'rb') as f:
			dictionary = pickle.load(f)
		return dictionary
	data = data1 + data2
	gensim_model = gensim.models.Word2Vec(data, size=Word2Vec_dim, workers = 16)
	gensim_model.save("gensim_model_%d.pkl"%(Word2Vec_dim))
	#gensim_model = gensim.models.Word2Vec.load("gensim_model_%d"%(Word2Vec_dim))
	
	dictionary = {}
	for x in gensim_model.wv.vocab:
		dictionary.update({x:gensim_model.wv[x]})
        
	with open('dictionary_%d.pkl'%(Word2Vec_dim), 'wb') as f:
		pickle.dump(dictionary, f)
	return dictionary

def getDictionary():
	print('---- get dictionary ----')
	with open('dictionary_%d.pkl'%(Word2Vec_dim), 'rb') as f:
		dictionary = pickle.load(f)
	return dictionary

def getTrainY(path):
	print('---- get trainY data ----')
	if True:
		data_fixed = np.load('trainY.npy')
		return data_fixed
	with open(path, 'r') as fp:
		data = fp.read()
	data = data.split('\n')
	del data[-1]
	del data[0]
	data_fixed = []
	for line in data:
		data_fixed.append(line.split(',', 1)[1])
	np.save('trainY.npy', data_fixed)
	return data_fixed

def getTrainX(path, dictName):
	print('---- get trainX data ----')
	if not_save:
		trainX = np.load('trainX_raw.npy')
		return trainX
	trainX = []
	data = []
	with open(path, 'r', encoding='utf-8') as fp:
		data = fp.read()
	trainX = handleXData(data, dictName)
	np.save('trainX_raw.npy', trainX)
	return trainX

def getTestX(path, dictName):
	print('---- get testX data ----')
	if not_save:
		testX = np.load('testX_raw.npy')
		return testX
	data = []
	with open(path, 'r', encoding='utf-8') as fp:
		data = fp.read()
	testX = handleXData(data, dictName)
	np.save('testX_raw.npy', testX)
	return testX

if __name__ == '__main__':
	print("util")