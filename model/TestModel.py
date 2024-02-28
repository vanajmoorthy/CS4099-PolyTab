from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, Activation
import numpy as np
import librosa
from scipy.io import wavfile
from keras import backend as K
from Metrics import *
import tensorflow as tf

# Define paths
MODEL_WEIGHTS_PATH = 'model/saved/c 2024-02-27 151223/0/weights.h5'
AUDIO_FILE_PATH = 'model/00_BN1-147-Gb_solo_mic.wav'

SR_DOWN = 22050  # Sample rate to use
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between successive frames
N_MELS = 128  # Number of Mel bands to generate
CQT_N_BINS = 192
CQT_BINS_PER_OCTAVE = 24

SR_ORIGINAL, data = wavfile.read(AUDIO_FILE_PATH)

# Define your model architecture here


class TestModel:
    def __init__(self,
                 batch_size=128,
                 epochs=8,
                 con_win_size=9):
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size

        self.input_shape = (192, self.con_win_size, 1)

        # these probably won't ever change
        self.num_classes = 21
        self.num_strings = 6

    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:, i, :]), axis=1))
        return K.concatenate(string_sm, axis=1)

    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(
                target[:, i, :], output[:, i, :])
        return loss

    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        model.compile(loss=self.catcross_by_string,
                      optimizer=tf.keras.optimizers.Adadelta(),
                      metrics=[self.avg_acc])

        return model


def preprocess_audio(audio_file_path, sr_down=SR_DOWN):
    sr, data = wavfile.read(audio_file_path)
    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(data, orig_sr=sr, target_sr=sr_down)
    data = np.abs(librosa.cqt(data, hop_length=HOP_LENGTH, sr=sr_down,
                  n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE))
    return data


if __name__ == '__main__':
    model = TestModel()
    model = model.build_model()

    # Load the weights
    model.load_weights(MODEL_WEIGHTS_PATH)

    # Preprocess the audio file
    preprocessed_audio = preprocess_audio(AUDIO_FILE_PATH)

    # Predict using the preprocessed audio
    predictions = model.predict(
        np.array([preprocessed_audio]))  # Add batch dimension

    print(predictions)
