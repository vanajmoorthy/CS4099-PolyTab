import os
import numpy as np
import librosa
import jams

data_path = "GuitarSet/annotation"
save_path = "ground_truth_tabs"

# Ensure the save_path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define the highest fret and MIDI pitches for standard guitar strings
highest_fret = 19
string_midi_pitches = [40, 45, 50, 55, 59, 64]

# Function to convert MIDI note to fret number
def convert_note_to_fret(midi_note, string_midi_pitch):
    fret_number = int(round(midi_note)) - string_midi_pitch
    return fret_number if 0 <= fret_number <= highest_fret else None

# Function to format and save tabs
def format_and_save_tab(file_name, annotations):
    # Assuming `annotations` is a list of lists, where each sublist is a string's frets over time
    with open(os.path.join(save_path, f"{file_name}_formatted_tab.txt"), 'w') as f:
        for frame_idx in range(len(annotations[0])):
            f.write(f"Frame {frame_idx}: ")
            frame_frets = [str(annotations[string_idx][frame_idx]) if annotations[string_idx][frame_idx] is not None else '-' for string_idx in range(6)]
            f.write(' '.join(frame_frets) + '\n')

# Function to process annotations and convert to guitar tabs
def process_annotations(file_name):
    jam = jams.load(os.path.join(data_path, file_name))
    duration = jam.file_metadata.duration
    # Time resolution is the hop_length in seconds
    time_resolution = librosa.samples_to_time(1, sr=22050) * 512  # Assuming a hop_length of 512 and sr of 22050

    # Calculate the number of frames for the entire duration
    num_frames = int(np.ceil(duration / time_resolution))
    
    # Initialize empty tab lines with the correct number of frames
    tab_lines = [['-' for _ in range(num_frames)] for _ in range(6)]
    
    # Process each note in the JAMS file
    for string_num, string_midi in enumerate(string_midi_pitches):
        for note in jam.search(namespace='note_midi')[string_num]:
            fret = convert_note_to_fret(note.value, string_midi)
            if fret is not None:
                # Find the corresponding frame based on note timing
                frame_idx = int(note.time / time_resolution)
                tab_lines[string_num][frame_idx] = str(fret)
    
    # Now that tab_lines are filled, format and save the tab
    format_and_save_tab(file_name.replace('.jams', ''), tab_lines)

# Process all .jams files in the directory
for file_name in os.listdir(data_path):
    if file_name.endswith(".jams"):
        process_annotations(file_name)
