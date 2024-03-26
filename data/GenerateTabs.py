import os
import numpy as np
import librosa
import jams

data_path = "GuitarSet/annotation"
save_path = "ground_truth_tabs"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def convert_note_to_tab(note, string_midi):
    # Conversion logic to find the fret number
    fret = note - string_midi
    if fret < 0 or fret > highest_fret:
        return None
    return fret

def get_string_notes(annotations, string_midi):
    string_notes = []
    for note in annotations:
        if note.value == string_midi:
            string_notes.append(note)
    return string_notes

def save_tab(file_name, annotations):
    with open(os.path.join(save_path, f"{file_name}.txt"), "w") as f:
        for string_annotations in annotations:
            # Assuming `string_annotations` is a list of lists with fret numbers per frame
            for frame in string_annotations:
                f.write('\t'.join(str(fret) if fret is not None else '-' for fret in frame))
                f.write('\n')  # Newline to separate frames

# List of MIDI notes for standard guitar strings
string_midi_pitches = [40, 45, 50, 55, 59, 64]
highest_fret = 19

for file_name in os.listdir(data_path):
    if file_name.endswith(".jams"):
        # Load JAMS annotation
        jam = jams.load(os.path.join(data_path, file_name))
        annotations = []
        
        for string_midi in string_midi_pitches:
            # Get the annotations for each string
            print(string_midi)
            print(jam.search(namespace='note_midi'))
            string_annotations = get_string_notes(jam.annotations['note_midi'][0], string_midi)
        
        # Save the tab for the song
        save_tab(file_name.split('.')[0], annotations)