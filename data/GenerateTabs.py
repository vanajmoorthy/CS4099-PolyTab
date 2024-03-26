import os
import numpy as np
import librosa
import jams

data_path = "GuitarSet/annotation"
save_path = "ground_truth_tabs"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Conversion logic to find the fret number
def convert_note_to_fret(midi_note, string_midi_pitch):
    fret_number = int(round(midi_note)) - string_midi_pitch
    if 0 <= fret_number <= highest_fret:
        return str(fret_number)
    return 'X'  # 'X' indicates a note that is not playable on this string

# Load JAMS annotation and convert to guitar tabs
def process_annotations(file_name):
    jam = jams.load(os.path.join(data_path, file_name))
    tab_lines = [[] for _ in range(6)]  # one sublist for each string

    for ann in jam.annotations['note_midi']:
        for note in ann:
            string_num = None
            # Determine which string this note belongs to
            for i, string_midi in enumerate(string_midi_pitches):
                if abs(note.value - string_midi) < 3:  # threshold for string matching
                    string_num = i
                    break
            if string_num is not None:
                fret = convert_note_to_fret(note.value, string_midi_pitches[string_num])
                tab_lines[string_num].append(fret)

    # Write the tab to a text file, with each string on its own line
    with open(os.path.join(save_path, f"{os.path.splitext(file_name)[0]}_tab.txt"), 'w') as fp:
        for string_frets in tab_lines:
            fp.write(' '.join(string_frets) + '\n')

# List of MIDI notes for standard guitar strings
string_midi_pitches = [40, 45, 50, 55, 59, 64]
highest_fret = 19

# Process all .jams files in the directory
for file_name in os.listdir(data_path):
    if file_name.endswith(".jams"):
        process_annotations(file_name)
