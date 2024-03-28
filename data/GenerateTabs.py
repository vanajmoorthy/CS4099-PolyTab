import os
import numpy as np
import jams

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
        Correctly handles the .jams file structure to access MIDI pitch values.
        """
        anno_file = os.path.join(self.anno_path, filename)
        jam = jams.load(anno_file)
        
        # Initialize a list for each string's labels
        labels = [[] for _ in range(6)]

        # Iterate through all annotations, filter for note_midi type
        for anno in jam.annotations:
            if anno.namespace != 'note_midi':
                continue  # Skip annotations that are not MIDI notes

            # Determine which string this annotation belongs to
            string_num = int(anno.annotation_metadata.instrument[0].split()[1]) - 1  # Assuming the instrument naming follows "String X"
            for note in anno.data:
                pitch = note.value
                # Convert MIDI pitch to fret number, considering the open string's MIDI pitch
                fret_number = int(round(pitch - self.string_midi_pitches[string_num]))
                # Clamp the fret number to the range [0, highest_fret]
                labels[string_num].append(max(min(fret_number, self.highest_fret), 0))

        # Normalize label lengths by padding with -1
        max_len = max(len(l) for l in labels)
        for l in labels:
            l.extend([-1] * (max_len - len(l)))  # Pad with -1 for no play

        labels = np.array(labels).T  # Correct the orientation of labels
        return labels



    def generate_tabs_from_labels(self, filename):
        """
        Generates text files for guitar tabs from labels.
        """
        labels = self.load_and_process_labels(filename)
        tab_strings = self.convert_labels_to_tabs(labels)

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
