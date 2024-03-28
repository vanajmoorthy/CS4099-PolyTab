import os
import numpy as np
import jams
from scipy.stats import mode

class GenerateTabs:
    def __init__(self, data_path="./GuitarSet/", output_dir="./ground_truth_tabs/"):
        self.anno_path = os.path.join(data_path, "annotation/")
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.string_midi_pitches = [40, 45, 50, 55, 59, 64]
        self.highest_fret = 19

    def convert_labels_to_tabs(self, labels):
        """
        Converts numeric labels into guitar tab strings.
        """
        tab_strings = []
        for label in labels:
            tab_string = ""
            for string_label in label:
                if string_label == -1:  # No play
                    tab_string += "x "
                else:
                    tab_string += f"{string_label} "
            tab_strings.append(tab_string.strip())
        return tab_strings

    def load_and_process_labels(self, filename):
        """
        Loads labels from a .jams file and processes them into a human-readable tab format.
        Adjusted to handle overlapping notes by considering only the first note encountered per string.
        """
        anno_file = os.path.join(self.anno_path, filename)
        jam = jams.load(anno_file)

        # Initialize a list for each string's labels with the max possible length based on the jam's duration and frame rate
        duration = jam.file_metadata.duration
        frame_rate = 43  # Assuming a frame rate of 43 frames per second as mentioned
        max_len = int(np.ceil(duration * frame_rate))

        labels = np.full((6, max_len), -1)  # Initialize with -1 for no play

        for string_num in range(6):
            anno = jam.search(namespace='note_midi')[string_num]
            for note in anno.data:
                start_time = note.time
                end_time = note.time + note.duration
                start_frame = int(np.floor(start_time * frame_rate))
                end_frame = int(np.ceil(end_time * frame_rate))
                pitch = note.value
                fret_number = int(round(pitch)) - self.string_midi_pitches[string_num]
                fret_number = max(min(fret_number, self.highest_fret), 0)  # Ensure fret number is within valid range

                # Fill the frames corresponding to the note duration with the fret number, considering only the first note
                for frame in range(start_frame, min(end_frame, max_len)):
                    if labels[string_num, frame] == -1:  # Fill only if no note has been registered yet
                        labels[string_num, frame] = fret_number

        # Convert the labels matrix into the expected list format for tab conversion
        labels_list = labels.T.tolist()  # Transpose and convert to list for compatibility
        return labels_list


    def temporal_smoothing(self, labels, window_size=43):
        """
        Applies temporal smoothing to labels over a specified window size.
        """
        smoothed_labels = []
        for frame_idx in range(len(labels)):
            window_start = max(0, frame_idx - window_size // 2)
            window_end = min(len(labels), frame_idx + window_size // 2 + 1)
            window_labels = labels[window_start:window_end]
            
            # Find the mode (most common element) in the window for each string
            mode_labels = mode(window_labels, axis=0).mode[0]
            smoothed_labels.append(mode_labels)
        
        return np.array(smoothed_labels)

    def generate_tabs_from_labels(self, filename):
        """
        Generates text files for guitar tabs from labels with temporal smoothing applied.
        """
        labels = self.load_and_process_labels(filename)
        smoothed_labels = self.temporal_smoothing(labels)  # Apply temporal smoothing
        tab_strings = self.convert_labels_to_tabs(smoothed_labels)

        output_file_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_tabs.txt")
        with open(output_file_path, 'w') as f:
            for tab_string in tab_strings:
                f.write(tab_string + "\n")

        print(f"Guitar tabs for {filename} saved to {output_file_path}")

    def generate_tabs_for_all_files(self):
        """
        Generates guitar tabs for all .jams files in the annotation directory.
        """
        jams_files = [f for f in os.listdir(self.anno_path) if f.endswith('.jams')]
        for filename in jams_files:
            self.generate_tabs_from_labels(filename)
            print(f"Generated tabs for {filename}")

if __name__ == "__main__":
    tab_generator = GenerateTabs()
    tab_generator.generate_tabs_for_all_files()
