import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, data_paths, batch_size=128, shuffle=True, label_dim=(6, 21), spec_repr="c", con_win_size=9):
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_dim = label_dim
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(float(len(self.data_paths)) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.data_paths[index *
                                      self.batch_size:(index + 1) * self.batch_size]
        X, y = self._data_generation(batch_paths)
        return X, y

    def on_epoch_end(self):
        self.data_paths = np.random.permutation(
            self.data_paths) if self.shuffle else self.data_paths

    def _data_generation(self, batch_paths):
        X = np.empty((self.batch_size, 192, self.con_win_size, 1))
        y = np.empty((self.batch_size, self.label_dim[0], self.label_dim[1]))

        for i, path in enumerate(batch_paths):
            filename, frame_idx = path.split("_")[-2:]
            frame_idx = int(frame_idx)
            data_dir = f"../data/spec_repr/{self.spec_repr}/"
            loaded = np.load(f"{data_dir}{filename}.npz")
            full_x = np.pad(loaded["repr"], [
                            (self.halfwin, self.halfwin), (0, 0)], mode='constant')
            sample_x = full_x[frame_idx:frame_idx + self.con_win_size]
            X[i] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)
            y[i] = loaded["labels"][frame_idx]

        return X, y
