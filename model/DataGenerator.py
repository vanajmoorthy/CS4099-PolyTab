import numpy as np
import keras
from keras.utils import Sequence


class DataGenerator(Sequence):

    def __init__(self, list_IDs, data_path="../data/cqt/", batch_size=128, shuffle=True, label_dim=(6, 21), spec_repr="c", con_win_size=9):
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_dim = label_dim
        self.spec_repr = spec_repr
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2
        self.X_dim = (self.batch_size, 192, self.con_win_size, 1)
        self.y_dim = (self.batch_size, self.label_dim[0], self.label_dim[1])
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty(self.X_dim)
        y = np.empty(self.y_dim)

        for i, ID in enumerate(list_IDs_temp):

            data_dir = self.data_path + "/"
            filename = "_".join(ID.split("_")[:-1]) + ".npz"
            frame_idx = int(ID.split("_")[-1])

            loaded = np.load(data_dir + filename)
            full_x = np.pad(loaded["repr"], [
                            (self.halfwin, self.halfwin), (0, 0)], mode='constant')
            sample_x = full_x[frame_idx: frame_idx + self.con_win_size]
            X[i,] = np.expand_dims(np.swapaxes(sample_x, 0, 1), -1)

            y[i,] = loaded["labels"][frame_idx]

        return X, y
