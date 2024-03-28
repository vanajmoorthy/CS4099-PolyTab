''' A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
'''


import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, Activation
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pandas as pd
from DataGenerator import DataGenerator
from LearnableWeightedLoss import LearnableWeightedLoss
from Metrics import pitch_precision, pitch_recall, pitch_f_measure, tab_precision, tab_recall, tab_f_measure, tab_disamb

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} Physical GPUs, {len(tf.config.experimental.list_logical_devices('GPU'))} Logical GPUs")
    except RuntimeError as e:
        print(e)

class PolyTab:
    def __init__(self, batch_size=128, epochs=8, con_win_size=9, spec_repr="c", data_path="./data/cqt/", id_file="id.csv", save_path="./model/saved/"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr
        self.data_path = data_path
        self.id_file = id_file
        self.save_path = save_path
        self.input_shape = (192, con_win_size, 1)
        self.num_classes = 21
        self.num_strings = 6
        self.load_IDs()
        self.save_folder = os.path.join(self.save_path, f"{self.spec_repr} {datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')}/")
        os.makedirs(self.save_folder, exist_ok=True)
        self.log_file = os.path.join(self.save_folder, "log.txt")
        self.metrics = {"pp": [], "pr": [], "pf": [], "tp": [], "tr": [], "tf": [], "tdr": [], "data": ["g0", "g1", "g2", "g3", "g4", "g5", "mean", "std dev"]}

    def load_IDs(self):
        self.list_IDs = list(pd.read_csv(os.path.join(self.data_path, self.id_file), header=None)[0])

    def partition_data(self, data_split):
        self.partition = {"training": [], "validation": []}
        for ID in self.list_IDs:
            guitarist = int(ID.split("_")[0])
            self.partition["validation" if guitarist == data_split else "training"].append(ID)

        self.training_generator = DataGenerator(self.partition['training'], data_path=self.data_path, batch_size=self.batch_size, shuffle=True, spec_repr=self.spec_repr, con_win_size=self.con_win_size)
        self.validation_generator = DataGenerator(self.partition['validation'], data_path=self.data_path, batch_size=len(self.partition['validation']), shuffle=False, spec_repr=self.spec_repr, con_win_size=self.con_win_size)

        self.split_folder = os.path.join(self.save_folder, str(data_split))
        os.makedirs(self.split_folder, exist_ok=True)

    def softmax_by_string(self, t):
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(tf.expand_dims(tf.nn.softmax(t[:, i, :]), axis=1))
        return tf.concat(string_sm, axis=1)
    

    # def catcross_by_string(self, target, output):
        #     # Compute standard categorical crossentropy
        #     cce = K.categorical_crossentropy(target, output)

        #     # Compute the absolute difference between the true and predicted classes
        #     true_classes = K.argmax(target, axis=-1)
        #     pred_classes = K.argmax(output, axis=-1)
        #     class_diff = K.abs(true_classes - pred_classes)

        #     # Ensure the operations are compatible with TensorFlow's dtype by casting to float
        #     weights = K.switch(K.less_equal(class_diff, 1),
        #                        K.cast(K.ones_like(class_diff), 'float32') * 0.5,
        #                        K.cast(K.ones_like(class_diff), 'float32') * 1.2)

        #     # Apply the weights to the crossentropy loss
        #     weighted_cce = cce * weights

        #     return K.mean(weighted_cce)

    # def catcross_by_string(self, target, output):
        # # Compute standard categorical crossentropy
        # cce = K.categorical_crossentropy(target, output)

        # # Compute the absolute difference between the true and predicted classes
        # true_classes = K.argmax(target, axis=-1)
        # pred_classes = K.argmax(output, axis=-1)
        # class_diff = K.abs(true_classes - pred_classes)

        # # Define a function for the weight, e.g., linear increase with class_diff
        # # You can adjust the slope (0.1 in this example) as necessary
        # weights = 1 + (0.01 * K.cast(class_diff, 'float32'))

        # # Apply the weights to the crossentropy loss
        # weighted_cce = cce * weights

        # return K.mean(weighted_cce)

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes * self.num_strings),
            Reshape((self.num_strings, self.num_classes)),
            Activation(self.softmax_by_string)
        ])
        loss_layer = LearnableWeightedLoss()

        # model.compile(optimizer=tf.keras.optimizers.Adadelta(),
        #               metrics=[self.avg_acc],
        #               loss=loss_layer.call)

        # model.compile(loss=self.catcross_by_string,
        #               optimizer=tf.keras.optimizers.Adadelta(),
        #               metrics=[self.avg_acc])

        model.compile(optimizer=AdamW(learning_rate=0.01, weight_decay=1e-4), metrics=['accuracy'], loss=loss_layer.call)
        self.model = model

    def train(self):
        tensorboard_dir = os.path.join(self.save_folder, 'tensorboard_logs')
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=tensorboard_dir)

        checkpoint_filepath = os.path.join(self.save_folder, 'checkpoint.h5')
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='loss', mode='min', save_best_only=True)

        self.model.fit(self.training_generator, epochs=self.epochs, verbose=1, callbacks=[model_checkpoint_callback, tensorboard_callback], use_multiprocessing=True, workers=9)

    def save_weights(self):
        self.model.save_weights(os.path.join(self.split_folder, "weights.h5"))

    def test(self):
        self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test)

    def save_predictions(self):
        np.savez(os.path.join(self.split_folder, "predictions.npz"), y_pred=self.y_pred, y_gt=self.y_gt)

    def evaluate(self):
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))

    def save_results_csv(self):
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] = self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(os.path.join(self.save_folder, "results.csv"))

if __name__ == '__main__':
    polytab = PolyTab()
    polytab.build_model()
    with open(polytab.log_file, 'w') as f:
        polytab.model.summary(print_fn=lambda x: f.write(x + '\n'))

    for fold in range(6):
        print(f"\nfold {fold}")
        polytab.partition_data(fold)
        polytab.build_model()
        polytab.train()
        polytab.save_weights()
        polytab.test()
        polytab.save_predictions()
        polytab.evaluate()

    polytab.save_results_csv()