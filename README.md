# PolyTab

A machine-learning model to automatically transcribe audio recordings of solo acoustic guitars to readable guitar tablature built using TensorFlow. Undertaken for my Senior-Honours dissertation at the University of St Andrews, this model builds on academic work in the nascent field of using machine learning for Music Information Retrieval (MIR), particularly the [TabCNN](https://archives.ismir.net/ismir2019/paper/000033.pdf) model and utilises a convolutional neural network to learn features from spectograms of guitar audio recordings to automatically produce usable guitar tablature

Supervised by [Dr Oggie Arandelovic](https://www.st-andrews.ac.uk/computer-science/people/oa7/) who awarded me a grade of a first and called this project "very good work with some genuine contributions to the state of the art".

To run the project and train the model yourself a few step must be taken.
* The dataset can be downloaded [here](https://zenodo.org/record/1422265/files/GuitarSet_audio_and_annotation.zip?download=1). Make sure to unzip this folder and place it in the root directory of the project.
* After that, you must activate a virtual environment to install the dependencies. This can be done with `python3 -m venv <env>`
* Then you can activate the virtual environment by running `source <env>/bin/activate`
* Next, please install the dependencies with `pip install -r requirements.txt`
* After this, you have to generate the CQT representations for the audio files using `python3 ParallelGenerateCQTs.py`
* And finally you can train the model with `python3 PolyTab.py`
* Once the model has trained, you can run `python3 PolyTabPredictor.py --weights "path/to/weights.h5" --audio "path/to/audio/file.wav"`
with the trained weights and the audio you want to predict for. The saved predictions can be found in the /predictions folder.
* You can then run `python3 PolyTabPredictor.py --weights "saved/c 2024-03-21 171741/5/weights.h5" --audio "path/to/audio/file.wav"` to predict using the model which was trained with the learnable weighted loss and AdamW optimiser.

The accompanying paper for this project is in the file Automatic Polyphonic Guitar Transcription.pdf
