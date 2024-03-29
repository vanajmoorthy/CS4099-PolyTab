import os
import numpy as np
import jams

class GenerateTabs:
    def __init__(self, data_path="./GuitarSet/", output_dir="./ground_truth_tabs/"):
        self.anno_path = os.path.join(data_path, "annotation/")
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.string_midi_pitches = [40, 45, 50, 55, 59, 64]  # MIDI numbers for open strings (EADGBE).

    def midi_to_fret(self, midi_number, string_number):
        """Converts a MIDI number to a fret number based on the open string's MIDI number."""
        return midi_number - self.string_midi_pitches[string_number]

    def generate_tab_for_jams(self, jams_file):
        """Generates guitar tabs directly from jams file annotations."""
        jam = jams.load(jams_file)
        tab_strings = []  # Store the final tab strings for each time slice.

        for string_num in range(6):  # For each guitar string.
            anno = jam.search(namespace='note_midi')[string_num]
            for note in anno.data:
                start_time = note.time
                end_time = note.time + note.duration
                midi_number = note.value
                fret_number = self.midi_to_fret(midi_number, string_num)
                # Here, you can insert logic to handle overlapping notes if necessary.
                tab_strings.append(f"{string_num+1}: Fret {fret_number}, Start: {start_time}s, End: {end_time}s")

        return tab_strings

    def generate_tabs_from_annotations(self):
        """Processes all .jams files in the directory and generates tabs."""
        for filename in os.listdir(self.anno_path):
            if filename.endswith('.jams'):
                jams_file = os.path.join(self.anno_path, filename)
                tab_strings = self.generate_tab_for_jams(jams_file)
                output_filename = filename.replace('.jams', '_tabs.txt')
                output_file = os.path.join(self.output_dir, output_filename)
                with open(output_file, 'w') as f:
                    for tab_string in tab_strings:
                        f.write(tab_string + "\n")
                print(f"Tabs for {filename} have been saved to {output_file}")

if __name__ == "__main__":
    tab_generator = GenerateTabs(data_path="./GuitarSet/", output_dir="./ground_truth_tabs/")
    tab_generator.generate_tabs_from_annotations()
