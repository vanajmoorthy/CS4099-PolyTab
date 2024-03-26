from GenerateCQTs import main
from multiprocessing import Pool
import sys

# number of files to process overall
num_filenames = 360

filename_indices = list(range(num_filenames)) * 4

if __name__ == "__main__":
    # number of processes will run simultaneously
    pool = Pool(11)
    inputval = list(zip(filename_indices))
    print(inputval)
    results = pool.starmap(main, inputval)