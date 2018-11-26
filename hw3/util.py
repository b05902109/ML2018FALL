import numpy as np
import pandas as pd

def readTrainData(training):
    train = pd.read_csv(training)
    ret = {
        'trainX': np.array(train['feature'].str.split(" ").values.tolist()).reshape(-1, 48, 48, 1).astype(np.float32),
        'trainY': pd.get_dummies(train['label']).values.astype(int),
    }
    return ret

def readTestData(testing):
    test = pd.read_csv(testing)
    return np.array(test['feature'].str.split(" ").values.tolist()).reshape(-1, 48, 48, 1).astype(np.float32)


def normalize(x):
    #mean = np.mean(x, axis=0).astype(np.float32)
    #dev = np.std(x, axis=0).astype(np.float32)
    #np.save("./mean.npy", mean)
    #np.save("./dev.npy", dev)
    mean = np.load('./mean.npy')
    dev = np.load('./dev.npy')
    x = (x-mean) / (dev+1e-10)
    return x

def saveAnswer(name, ans):
    label = np.arange(ans.shape[0]).reshape(-1, 1)
    outputFormat = np.concatenate([label, ans], axis=1).astype(int)
    np.savetxt(name, outputFormat, fmt="%s", header='id,label', comments='', delimiter=",")
    return

if __name__ == '__main__':
    print('util.py')