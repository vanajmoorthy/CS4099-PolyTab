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
        Ensures that all lists have the same length before converting to a NumPy array.
        """
        anno_file = os.path.join(self.anno_path, f"{filename}")
        jam = jams.load(anno_file)
        labels = []

        # Determine the maximum length among all strings' annotations
        max_length = 0
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            max_length = max(max_length, len(anno.data))

        # Process annotations, ensuring all lists are the same length
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_labels = []
            for note in anno.data:
                pitch = note.value if note.value else -1  # Directly use the note value
                fret = int(round(pitch)) - self.string_midi_pitches[string_num] if pitch != -1 else -1
                fret = max(min(fret, self.highest_fret), -1)  # Ensure fret number is within valid range
                string_labels.append(fret)

            # Pad the list to ensure it's the same length as the longest list
            string_labels += [-1] * (max_length - len(string_labels))
            labels.append(string_labels)

        labels = np.array(labels).T  # Now safe to transpose
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
