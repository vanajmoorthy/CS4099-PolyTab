import numpy as np
import librosa
from keras.models import load_model
from scipy.io import wavfile


# Define the preprocess_audio function (from TabDataReprGen.py)

# Replace with the path to your saved model
MODEL_PATH = 'saved/c 2024-02-27 151223/0/weights.h5'
AUDIO_FILE_PATH = '00_BN1-147-Gb_solo_mic.wav'
SR_DOWN = 22050  # Sample rate to use
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between successive frames
N_MELS = 128  # Number of Mel bands to generate
CQT_N_BINS = 192
CQT_BINS_PER_OCTAVE = 24

SR_ORIGINAL, data = wavfile.read(AUDIO_FILE_PATH)


def preprocess_audio(data):
    data = data.astype(float)
    data = librosa.util.normalize(data)
    data = librosa.resample(
        data, orig_sr=SR_ORIGINAL, target_sr=SR_DOWN)
    data = np.abs(librosa.cqt(data,
                              hop_length=HOP_LENGTH,
                              sr=SR_ORIGINAL,
                              n_bins=CQT_N_BINS,
                              bins_per_octave=CQT_BINS_PER_OCTAVE))

    return data


# Load your saved model
model = load_model(MODEL_PATH)


preprocessed_audio = preprocess_audio(AUDIO_FILE_PATH)

# Run the model
predictions = model.predict(preprocessed_audio)

# TODO: Post-process the predictions as needed
# This could involve converting the output tensor to a readable format,
# mapping class indices back to labels, etc.

print(predictions)
