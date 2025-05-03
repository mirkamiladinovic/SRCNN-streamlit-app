import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam

def build_model():
    model = Sequential()
    model.add(Conv2D(128, (9, 9), activation='relu', padding='valid', input_shape=(None, None, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (5, 5), activation='linear', padding='valid'))
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error')
    return model

def load_model(weights_path='weights/3051crop_weight_200.h5'):
    model = build_model()
    model.load_weights(weights_path)
    return model
