

from keras.layers import Layer
from keras.initializers import Constant
from keras import backend as K


class LearnableWeightedLoss(Layer):
    def __init__(self, **kwargs):
        super(LearnableWeightedLoss, self).__init__(**kwargs)
        # Initialize the weights to 1, indicating no initial penalty
        self.weight = self.add_weight(name='loss_weight',
                                      shape=(1,),
                                      initializer=Constant(value=1.),
                                      trainable=True)

    def call(self, y_true, y_pred):
        # Standard categorical crossentropy
        cce = K.categorical_crossentropy(y_true, y_pred)

        # Compute the absolute difference between true and predicted classes
        true_classes = K.argmax(y_true, axis=-1)
        pred_classes = K.argmax(y_pred, axis=-1)
        class_diff = K.abs(true_classes - pred_classes)

        # Apply the learned weights as a function of class difference
        weighted_cce = cce * (1 + self.weight * K.cast(class_diff, 'float32'))

        # Return the mean loss
        return K.mean(weighted_cce)
