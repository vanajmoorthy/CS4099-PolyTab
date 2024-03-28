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
        self.frame_rate = 43

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
        max_len = int(np.ceil(duration * self.frame_rate))

        labels = np.full((6, max_len), -1)  # Initialize with -1 for no play

        for string_num in range(6):
            anno = jam.search(namespace='note_midi')[string_num]
            for note in anno.data:
                start_time = note.time
                end_time = note.time + note.duration
                start_frame = int(np.floor(start_time * self.frame_rate))
                end_frame = int(np.ceil(end_time * self.frame_rate))
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

    
    def aggregate_annotations(self, annotations):
        """
        Aggregates annotations over a specified window to perform temporal smoothing.
        """
        # Calculate the total frames based on the maximum duration across all strings
        max_len = max(len(string_ann) for string_ann in annotations)

        # Assuming annotations have been correctly aligned and padded before this method
        # Initialize aggregated annotations list
        aggregated_annotations = []

        # Perform aggregation over each time window for all strings
        for frame in range(0, max_len, self.frame_rate):
            window = [string_ann[frame:frame + self.frame_rate] for string_ann in annotations]
            aggregated_frame = []
            
            for string_ann in window:
                # Exclude '-1' indicating no play, then find the most common fret if any
                frets = [fret for fret in string_ann if fret != -1]
                if frets:
                    most_common_fret = max(set(frets), key=frets.count)
                else:
                    most_common_fret = -1  # No play for the entire window
                aggregated_frame.append(most_common_fret)
                
            aggregated_annotations.append(aggregated_frame)

        return aggregated_annotations


    def generate_tabs_from_labels(self, filename):
        """
        Generates text files for guitar tabs from labels, including temporal smoothing.
        """
        annotations = self.load_and_process_labels(filename)
        smoothed_annotations = self.aggregate_annotations(annotations)
        tab_strings = self.convert_labels_to_tabs(smoothed_annotations)

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
