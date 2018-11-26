from util import *
import sys
from keras.models import load_model

if __name__ == '__main__':
    testX = readTestData(sys.argv[1])
    testX = normalize(testX).reshape(-1, 48, 48, 1)
    print('---- read data finish ----')
    
    modelEns = load_model('model1234567.h5?dl=1%0D')
    print('---- load model finish ----')
    
    
    testY = modelEns.predict(testX)
    testY = np.argmax(testY, axis=1).reshape(-1, 1)
    saveAnswer(sys.argv[2], testY)
    print('---- save model1234567.csv finish ----')
    