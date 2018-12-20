from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Activation
from keras.models import load_model
from keras.optimizers import Adam
from keras import layers

def model1(input_len, input_dim):
	model = Sequential()
	model.add(Bidirectional(LSTM(100, kernel_initializer='Orthogonal', return_sequences=False, dropout=0.5),input_shape=(input_len, input_dim)))
	model.add(BatchNormalization(axis=-1, momentum=0.5))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def model2(input_len, input_dim):
	model = Sequential()
	model.add(LSTM(256,activation="tanh",dropout=0.3,return_sequences = False, kernel_initializer='Orthogonal', input_shape=(input_len, input_dim)))
	model.add(BatchNormalization(axis=-1, momentum=0.5))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def model3(input_len, input_dim):
	model = Sequential()
	model.add(Bidirectional(LSTM(100, kernel_initializer='Orthogonal', return_sequences=False, dropout=0.3, recurrent_dropout=0.3),input_shape=(input_len, input_dim)))
	model.add(BatchNormalization(axis=-1, momentum=0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def model4(input_len, input_dim):
	model = Sequential()
	model.add(LSTM(256,activation="tanh",dropout=0.3,return_sequences = False, kernel_initializer='Orthogonal', input_shape=(input_len, input_dim)))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def model5(input_len, input_dim):
	model = Sequential()
	model.add(LSTM(256,activation="tanh",dropout=0.3,return_sequences = True,kernel_initializer='Orthogonal', input_shape=(input_len, input_dim)))
	model.add(LSTM(256,activation="tanh",dropout=0.3,return_sequences = False,kernel_initializer='Orthogonal'))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
	return model

def model6(input_len, input_dim):
	model = Sequential()
	model.add(LSTM(256,dropout=0.3,return_sequences = False, recurrent_dropout=0.3, kernel_initializer='Orthogonal', input_shape=(input_len, input_dim)))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['acc'])
	return model

def model7(input_len, input_dim):
	model = Sequential()
	model.add(LSTM(256,dropout=0.3,return_sequences = True, recurrent_dropout=0.3, kernel_initializer='Orthogonal', input_shape=(input_len, input_dim)))
	model.add(LSTM(256,dropout=0.3,return_sequences = False, recurrent_dropout=0.3, kernel_initializer='Orthogonal'))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['acc'])
	return model

def model8(input_len, input_dim):
	model = Sequential()
	model.add(LSTM(256,dropout=0.3,return_sequences = True, recurrent_dropout=0.3, input_shape=(input_len, input_dim)))
	model.add(LSTM(256,dropout=0.3,return_sequences = False, recurrent_dropout=0.3))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
	model.compile(loss='binary_crossentropy',optimizer=adam, metrics=['acc'])
	return model