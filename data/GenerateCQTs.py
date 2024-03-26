import os
import numpy as np
import jams
from scipy.io import wavfile
import librosa
from tensorflow.keras.utils import to_categorical


class GenerateCQTs:
    def __init__(self, data_path="./data/GuitarSet/"):
        self.audio_path = os.path.join(data_path, "audio/audio_mic/")
        self.anno_path = os.path.join(data_path, "annotation/")
        self.string_midi_pitches = [40, 45, 50, 55, 59, 64]
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2
        self.downsample = True
        self.normalize = True
        self.sr_downs = 22050
        self.cqt_n_bins = 192
        self.cqt_bins_per_octave = 24
        self.n_fft = 2048
        self.hop_length = 512
        self.save_path = "./data/cqt/"

    def load_repr_and_labels(self, filename):
        audio_file = os.path.join(self.audio_path, f"{filename}_mic.wav")
        anno_file = os.path.join(self.anno_path, f"{filename}.jams")
        jam = jams.load(anno_file)
        sr_original, data = wavfile.read(audio_file)
        sr_curr = sr_original

        repr_ = self.preprocess_audio(data, sr_original, sr_curr)
        labels = self.get_labels(jam, repr_, sr_curr)

        return repr_, labels

    def get_labels(self, jam, repr_, sr_curr):
        frame_indices = range(len(repr_))
        times = librosa.frames_to_time(
            frame_indices, sr=sr_curr, hop_length=self.hop_length)
        labels = []

        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            string_labels = [-1 if not label else int(round(label[0]) - self.string_midi_pitches[string_num])
                             for label in string_label_samples]
            labels.append(string_labels)

        labels = np.array(labels)
        labels = np.swapaxes(labels, 0, 1)
        labels = self.clean_labels(labels)

        return labels

    def correct_numbering(self, n):
        return max(min(n + 1, self.highest_fret), 0)

    def categorical(self, label):
        return to_categorical(label, self.num_classes)

    def clean_label(self, label):
        label = [self.correct_numbering(n) for n in label]
        return self.categorical(label)

    def clean_labels(self, labels):
        return np.array([self.clean_label(label) for label in labels])

    def preprocess_audio(self, data, sr_original, sr_curr):
        data = data.astype(float)
        if self.normalize:
            data = librosa.util.normalize(data)
        if self.downsample:
            data = librosa.resample(
                data, orig_sr=sr_original, target_sr=self.sr_downs)
            sr_curr = self.sr_downs

        data = np.abs(librosa.cqt(data,
                                  hop_length=self.hop_length,
                                  sr=sr_curr,
                                  n_bins=self.cqt_n_bins,
                                  bins_per_octave=self.cqt_bins_per_octave))

        return np.swapaxes(data, 0, 1)

    def save_data(self, filename, repr_, labels):
        np.savez(os.path.join(self.save_path,
                 f"{filename}.npz"), repr=repr_, labels=labels)

    def get_filenames(self):
        filenames = [f[:-5]
                     for f in os.listdir(self.anno_path) if f.endswith(".jams")]
        return sorted(filenames)

    def process_file(self, n):
        filename = self.get_filenames()[n]
        repr_, labels = self.load_repr_and_labels(filename)
        self.save_data(filename, repr_, labels)
        print(f"Processed: {filename}, {len(labels)} frames")


def main(n):
    gen = GenerateCQTs()
    gen.process_file(n)


if __name__ == "__main__":
    n = int(sys.argv[1])
    main(n)
