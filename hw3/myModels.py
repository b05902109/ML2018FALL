import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add

def model1():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3, decay=5e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
                        rotation_range=30.0,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='constant',
                        vertical_flip=False)
    
    return model, datagen

def model2():
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=(48, 48, 1), padding='same'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, 3, padding='same'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    
    model.add(Conv2D(512, 3, padding='same'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    
    model.add(Conv2D(512, 3, padding='same'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3, decay=5e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
                        rotation_range=30.0,
                        width_shift_range=0.15,
                        height_shift_range=0.15,
                        shear_range=0.2,
                        zoom_range=0.3,
                        horizontal_flip=True,
                        fill_mode='constant',
                        vertical_flip=False)
    
    return model, datagen

def model3():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3, decay=5e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
                        rotation_range=30.0,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='constant',
                        vertical_flip=False)
    
    return model, datagen

def model4():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3, decay=5e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
                        rotation_range=30.0,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='constant',
                        vertical_flip=False)
    
    return model, datagen

def model5():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3, decay=5e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
                        rotation_range=30.0,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='constant',
                        vertical_flip=False)
    
    return model, datagen

def model6():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3, decay=5e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
                        rotation_range=30.0,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='constant',
                        vertical_flip=False)
    
    return model, datagen

def model7():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.5))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU())
    model.add(Dense(7))
    model.add(Activation('softmax'))
    adam = Adam(lr=1e-3, decay=5e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
                        rotation_range=30.0,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='constant',
                        vertical_flip=False)
    
    return model, datagen
