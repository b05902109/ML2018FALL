from util import *
from myModels import *
import sys
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense
from keras import layers

def ensembleModels(modelList):
        model_input = Input(shape=modelList[0].input_shape[1:])
        yModels=[model(model_input) for model in modelList] 
        yAvg=layers.average(yModels)
        modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
        return modelEns

def trainModel(index, model, datagen, trainX, trainY):
    datagen.fit(trainX)
    datagen_flow = datagen.flow(x=trainX, y=trainY, batch_size=128)
    #print('---- Start training ----')
    for i in range(6):
        model.fit_generator(datagen_flow,steps_per_epoch=trainX.shape[0] / 128, initial_epoch=i * 50,epochs=(i + 1) * 50)
        model.save('modelDir/model%d_%d.h5'%(index, i))

    model.save('modelDir/model%d.h5'%(index))
    return model

if __name__ == '__main__':

    data = readTrainData(sys.argv[1])
    trainX = data['trainX']
    trainY = data['trainY']
    print('---- read data finish ----')
    trainX = normalize(trainX)
    print('---- normalize finish ----')   

    model, datagen = model1()
    model1 = trainModel(1, model, datagen, trainX, trainY)
    model, datagen = model2()
    model2 = trainModel(2, model, datagen, trainX, trainY)
    model, datagen = model3()
    model3 = trainModel(3, model, datagen, trainX, trainY)
    model, datagen = model4()
    model4 = trainModel(4, model, datagen, trainX, trainY)
    model, datagen = model5()
    model5 = trainModel(5, model, datagen, trainX, trainY)
    model, datagen = model6()
    model6 = trainModel(6, model, datagen, trainX, trainY)
    model, datagen = model7()
    model7 = trainModel(7, model, datagen, trainX, trainY)
    print('---- training ending ----')
    
    '''
    model1 = load_model('modelDir/model1.h5')
    model2 = load_model('modelDir/model2.h5')
    model3 = load_model('modelDir/model3.h5')
    model4 = load_model('modelDir/model4.h5')
    model5 = load_model('modelDir/model5.h5')
    model6 = load_model('modelDir/model6.h5')
    model7 = load_model('modelDir/model7.h5')
    '''
    modelList = [model1, model2, model3, model4, model5, model6, model7]
    modelEns = ensembleModels(modelList)
    modelEns.save('model1234567.h5?dl=1%0D')
    print('---- model save finish ----')