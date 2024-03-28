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
        """
        anno_file = os.path.join(self.anno_path, f"{filename}")
        jam = jams.load(anno_file)
        labels = []

        # Iterate over each string's annotations
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_labels = []

            # Process each note in the annotations for the current string
            for note in anno.data:
                if note.value:  # Check if there is a note value
                    pitch = note.value  # Directly use the note value (which is a float)
                    fret = int(round(pitch)) - self.string_midi_pitches[string_num]
                    string_labels.append(max(min(fret, self.highest_fret), -1))  # Ensure fret number is within valid range
                else:
                    string_labels.append(-1)  # Indicate no play with -1

            labels.append(string_labels)
        
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
