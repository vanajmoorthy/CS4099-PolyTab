from TabDataReprGen import main
from multiprocessing import Pool

# number of files to process overall
num_filenames = 360
modes = ["c", "m", "cm", "s"]

# Create a list of indices for filenames and multiply it by the number of modes
filename_indices = list(range(num_filenames)) * len(modes)
# Create a list that repeats each mode num_filenames times and concatenates them
mode_list = [mode for mode in modes for _ in range(num_filenames)]

if __name__ == "__main__":
    # number of processes that will run simultaneously
    with Pool(11) as pool:
        # zip creates pairs of filename_indices and mode_list, which are passed to main
        results = pool.starmap(main, [[index, mode] for index, mode in zip(filename_indices, mode_list)])
