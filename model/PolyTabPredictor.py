import argparse
import os
import numpy as np
import librosa
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

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
        return self.predictions_to_tabs(predictions)

    def predictions_to_tabs(self, predictions, threshold=0.1):
        """Convert model predictions to guitar tab format.

        Args:
            predictions: Raw predictions from the model.
            threshold: Probability threshold to consider a prediction as valid.

        Returns:
            A list of guitar tab frames.
        """
        tabs = []
        for frame in predictions:
            tab_frame = ['-'] * self.num_strings  # Default to no play
            for string_index, string_predictions in enumerate(frame):
                max_prob = np.max(string_predictions)
                if max_prob > threshold:
                    fret = np.argmax(string_predictions)
                    tab_frame[string_index] = str(fret - 1) if fret > 0 else 'x'  # '0' for open string, otherwise fret number
            tabs.append(tab_frame)
        return tabs
    
    def aggregate_predictions(self, tabs, aggregation_window=43):
        """
        Aggregate predictions over a specified window to reduce granularity.
        This method assumes tabs to be a list of lists where each sublist represents
        a frame's prediction across all strings.
        
        Args:
            tabs (list): The list of per-frame predictions.
            aggregation_window (int): Number of frames to aggregate over.
            
        Returns:
            list: Aggregated tab predictions.
        """
        aggregated_tabs = []
        for i in range(0, len(tabs), aggregation_window):
            window = tabs[i:i+aggregation_window]
            # Aggregate predictions by taking the most common prediction in the window for each string
            aggregated_frame = []
            for string_idx in range(6):  # Assuming 6 strings
                # Extract predictions for the current string across the window
                string_predictions = [frame[string_idx] for frame in window if frame[string_idx] != 'x']
                if string_predictions:
                    # Find the most common prediction, default to 'x' if no prediction is present
                    most_common = max(set(string_predictions), key=string_predictions.count)
                else:
                    most_common = 'x'
                aggregated_frame.append(most_common)
            aggregated_tabs.append(aggregated_frame)
        return aggregated_tabs
    
    def save_aggregated_tabs_to_file(self, aggregated_tabs, filename):
        """
        Saves the aggregated tab predictions to a text file.
        
        Args:
            aggregated_tabs (list): Aggregated tab predictions.
            filename (str): Path to the output text file.
        """
        with open(filename, 'w') as file:
            for tab in aggregated_tabs:
                tab_line = ' '.join(tab)
                file.write(f"{tab_line}\n")
        print(f"Aggregated tabs saved to {filename}")
    
    def predict_save_aggregated(self, audio_file, output_dir=None, aggregation_window=43):
        """
        Predicts and aggregates tabs for an audio file and saves them to a text file.
        """
        tabs = self.predict(audio_file, output_dir)
        aggregated_tabs = self.aggregate_predictions(tabs, aggregation_window)

        # Save aggregated tabs to a text file
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            base_filename = os.path.splitext(os.path.basename(audio_file))[0]
            aggregated_tabs_file = os.path.join(output_dir, f"{base_filename}_aggregated_tabs.txt")
            self.save_aggregated_tabs_to_file(aggregated_tabs, aggregated_tabs_file)

            # Generate and save images of aggregated tabs.
            self.create_guitar_tab_image(aggregated_tabs, audio_file[:-4])
        else:
            print("Output directory is not specified.")
        
        # For visualizing tabs as images, you can still use your existing method
        # self.create_guitar_tab_image(aggregated_tabs, output_dir=tab_output_dir, lines_per_image=5)
    
    def create_guitar_tab_image(self, tabs, output_dir, lines_per_image=5):
        """Generate images of guitar tabs, splitting into multiple images if necessary."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("saving image")
        total_frames = len(tabs)
        frames_per_line = 6  # Adjust based on how many frames you want per line of tab
        total_lines = total_frames // frames_per_line + (1 if total_frames % frames_per_line > 0 else 0)
        image_count = total_lines // lines_per_image + (1 if total_lines % lines_per_image > 0 else 0)
        
        for image_index in range(image_count):
            start_line = image_index * lines_per_image
            end_line = min((image_index + 1) * lines_per_image, total_lines)
            
            # Set the figure size based on the number of lines in this image
            fig_height = end_line - start_line
            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, end_line - start_line)
            
            ax.invert_yaxis()  # Invert the y-axis to have the first frame at the top
            
            for line_index in range(start_line, end_line):
                for frame_offset in range(frames_per_line):
                    frame_index = line_index * frames_per_line + frame_offset
                    if frame_index >= total_frames:
                        break
                    tab_frame = tabs[frame_index]
                    
                    # Draw strings for this frame
                    for string_index in range(6):
                        y_position = line_index - start_line + (string_index * 0.1)  # Adjust spacing as needed
                        ax.axhline(y_position, color='black', linewidth=2)
                        
                        fret = tab_frame[string_index]
                        if fret not in ['x', '-']:  # Check if fret is a number or 'x'
                            ax.text(5 + frame_offset, y_position, fret, ha='center', va='center', fontsize=8, family='monospace')  # Adjust text size and position as needed
            
            ax.axis('off')
            plt.box(False)
            
            output_path = os.path.join(output_dir, f"tab_{image_index + 1:03d}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict and save guitar tabs from an audio file.')
    parser.add_argument('--weights', required=True, help='Path to the model weights file.')
    parser.add_argument('--audio', required=True, help='Path to the audio file.')
    args = parser.parse_args()

    model_weights_path = args.weights
    audio_file = args.audio
    output_dir = "predictions"

    predictor = PolyTabPredictor(model_weights_path)
    predictor.predict_save_aggregated(audio_file, output_dir=output_dir, aggregation_window=43)
