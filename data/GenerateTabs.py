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


def format_and_save_tab(file_name, annotations):
    # Assuming `annotations` is a list of lists, where each sublist is a string's frets over time
    num_frames = len(annotations[0])  # Assuming all strings have the same number of frames

    with open(os.path.join(save_path, f"{file_name}_formatted_tab.txt"), 'w') as f:
        for frame_idx in range(num_frames):
            f.write(f"Frame {frame_idx}: ")
            frame_frets = [str(annotations[string_idx][frame_idx]) if annotations[string_idx][frame_idx] != 'X' else '0' for string_idx in range(6)]
            f.write(' '.join(frame_frets) + '\n')

# Load JAMS annotation and convert to guitar tabs
def process_annotations(file_name):
    jam = jams.load(os.path.join(data_path, file_name))
    # Determine the max number of notes across all strings to ensure alignment
    max_notes = 0
    for string_midi in string_midi_pitches:
        notes = [note for ann in jam.annotations['note_midi'] for note in ann.data if abs(note.value - string_midi) < 6]
        if len(notes) > max_notes:
            max_notes = len(notes)
    
    # Initialize tab_lines with placeholders for no note played
    tab_lines = [['X' for _ in range(max_notes)] for _ in range(6)]
    
    for ann in jam.annotations['note_midi']:
        for note in ann.data:
            string_num = None
            for i, string_midi in enumerate(string_midi_pitches):
                if abs(note.value - string_midi) < 6:  # threshold for string matching
                    string_num = i
                    break
            if string_num is not None:
                fret = convert_note_to_fret(note.value, string_midi_pitches[string_num])
                # Find the next available slot (X) in the corresponding string line
                next_slot = tab_lines[string_num].index('X')
                tab_lines[string_num][next_slot] = fret if fret != 'X' else '0'  # Replace 'X' with '0' if fret is not playable
    
    print(tab_lines)
    # Now that tab_lines are filled, format and save the tab
    format_and_save_tab(file_name.replace('.jams', ''), tab_lines)


# List of MIDI notes for standard guitar strings
string_midi_pitches = [40, 45, 50, 55, 59, 64]
highest_fret = 19

# Process all .jams files in the directory
for file_name in os.listdir(data_path):
    if file_name.endswith(".jams"):
        process_annotations(file_name)
