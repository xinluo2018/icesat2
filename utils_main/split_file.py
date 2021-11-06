# author: Fernando Paolo, 
# modify: xin luo, 2021.8.10.   
# des: split file into multiple files.


'''
Splits large 1d HDF5 file(s) into smaller ones.
example:
    python split_file.py file.h5 -k 16
To see available options:
    python split_file.py -h
Also see complementary program: 'merge.py'
'''

import os
import h5py
import argparse


# Default number of smaller files to split to
KFILES = 2

# Default njobs for sequential/parallel run
NJOBS = 1


# Pass command-line arguments
parser = argparse.ArgumentParser(
        description='Splits large HDF5 file(s) into smaller ones.')

parser.add_argument(
        'files', metavar='files', type=str, nargs='+',
        help='HDF5 file(s) to split')

parser.add_argument(
        '-k', metavar='nfiles', dest='nfiles', type=int, nargs=1,
        default=[KFILES],
        help=('number of smaller files to split to (-n 2)'))

parser.add_argument(
        '-n', metavar='njobs', dest='njobs', type=int, nargs=1,
        default=[NJOBS],
        help=('number of jobs for parallel processing (-n 1)'))


def partition(length, parts):
    """des: Partitions 'length' into (approximately) equal 'parts'.
        arg:
            length: int, length of the Dataset  
            parts: int, parts of the Dataset should be divided into.
        retrun:
            sublengths: list, contains length (int) of each part divided Dataset.
    """
    sublengths = [length//parts] * parts
    for i in range(length % parts):  # treatment of remainder
        sublengths[i] += 1
    return sublengths


def split_file(ifile):

    print(('input -> ', ifile))

    with h5py.File(ifile, 'r') as f:

        # Determine the total legth of input file
        total_length = list(f.values())[0].shape[0]

        # Determine the length of output files
        lengths = partition(total_length, nfiles)
        print(lengths)

        # Determine the names of output files
        fname = os.path.splitext(ifile)[0] + '_%03d.h5'
        outfiles = [(fname % k) for k in range(len(lengths))]

        i1, i2 = 0, 0
        for outfile, length in zip(outfiles, lengths):
            i2 += length
            # Save chunks of each variable from f -> f2 
            with h5py.File(outfile, 'w') as f2:
                for key in list(f.keys()):
                    f2[key] = f[key][i1:i2]
            i1 = i2
            print(('output ->', outfile))

if __name__ == '__main__':
    # global variables
    args = parser.parse_args()
    files = args.files
    nfiles = args.nfiles[0]
    njobs = args.njobs[0]

    if njobs == 1:
        # Sequential code
        print('Running sequential code ...')
        [split_file(f) for f in files]

    else:
        # Parallel code
        print(('Running parallel code (%d jobs) ...' % njobs))
        from joblib import Parallel, delayed
        Parallel(n_jobs=njobs, verbose=5)(delayed(split_file)(f) for f in files)