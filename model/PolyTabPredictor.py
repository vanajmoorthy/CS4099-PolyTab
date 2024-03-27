import argparse
import os
import numpy as np
import librosa
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Conv2D, MaxPooling2D
from keras import backend as K


class PolyTabPredictor:
    def __init__(self, model_weights_path, con_win_size=9, spec_repr="c"):
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.cqt_n_bins = 192
        self.cqt_bins_per_octave = 24
        self.n_fft = 2048
        self.hop_length = 512
        self.sr_downs = 22050
        self.num_classes = 21
        self.num_strings = 6
        self.input_shape = (192, self.con_win_size, 1)
        self.load_model(model_weights_path)

    def load_model(self, model_weights_path):
        self.model = self.build_model()
        self.model.load_weights(model_weights_path)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu', input_shape=self.input_shape))
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
        return model

    def softmax_by_string(self, t):
        sh = K.shape(t)
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:, i, :]), axis=1))
        return K.concatenate(string_sm, axis=1)

    def preprocess_audio(self, audio_file):
        data, sr = librosa.load(audio_file, sr=None)
        data = librosa.util.normalize(data)
        data = librosa.resample(data, orig_sr=sr, target_sr=self.sr_downs)
        data = np.abs(librosa.cqt(data, hop_length=self.hop_length, sr=self.sr_downs,
                                  n_bins=self.cqt_n_bins, bins_per_octave=self.cqt_bins_per_octave))
        return np.swapaxes(data, 0, 1)

    def predict(self, audio_file, output_dir=None):
        repr_ = self.preprocess_audio(audio_file)
        full_x = np.pad(
            repr_, [(self.con_win_size // 2, self.con_win_size // 2), (0, 0)], mode='constant')
        predictions = []
        for frame_idx in range(len(repr_)):
            sample_x = full_x[frame_idx:frame_idx + self.con_win_size]
            sample_x = np.expand_dims(np.expand_dims(
                np.swapaxes(sample_x, 0, 1), 0), -1)
            prediction = self.model.predict(sample_x)
            predictions.append(prediction[0])

        # Convert predictions to guitar tabs format
        tabs = self.predictions_to_tabs(predictions)

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            audio_filename = os.path.basename(audio_file)
            raw_predictions_file = os.path.join(output_dir, f"{os.path.splitext(audio_filename)[0]}_raw_predictions.txt")
            tabs_file = os.path.join(output_dir, f"{os.path.splitext(audio_filename)[0]}_tabs.txt")

            # Save raw predictions
            with open(raw_predictions_file, "w") as rp_file:
                for frame_idx, raw_prediction in enumerate(predictions):
                    formatted_prediction = np.array2string(raw_prediction, separator=', ', suppress_small=True)
                    rp_file.write(f"Frame {frame_idx}: {formatted_prediction}\n")

            # Save mapped predictions (tabs)
            with open(tabs_file, "w") as tf_file:
                for frame_idx, tab_frame in enumerate(tabs):
                    tf_file.write(f"Frame {frame_idx}: {' '.join(map(str, tab_frame))}\n")

        return tabs


    def predictions_to_tabs(self, predictions):
        tabs = []
        for frame in predictions:
            tab_frame = ['-'] * self.num_strings  # Initialize with '-' for no play
            for string_index, string_predictions in enumerate(frame):
                # Get the fret with the highest probability
                fret = np.argmax(string_predictions)
                # Check if the fret is the open string played
                if string_predictions[fret] > some_threshold:  # You need to define some_threshold
                    tab_frame[string_index] = '0' if fret == 0 else str(fret - 1)
            tabs.append(tab_frame)
        return tabs



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict guitar tabs from audio file.')
    parser.add_argument('--weights', required=True, help='Path to the model weights file.')
    parser.add_argument('--audio', required=True, help='Path to the audio file.')
    args = parser.parse_args()

    # Set numpy print options to suppress scientific notation and increase precision
    np.set_printoptions(suppress=True, precision=8)

    # Use the provided arguments
    model_weights_path = args.weights
    audio_file = args.audio
    output_dir = "predictions"

    # Initialize and use your predictor
    predictor = PolyTabPredictor(model_weights_path)
    predictions = predictor.predict(audio_file, output_dir=output_dir)
