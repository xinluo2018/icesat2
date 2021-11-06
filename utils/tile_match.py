# author: Fernando Paolo, 
# modify: xin luo, 2021.8.15
# des: get pair-wise tiles.

import os
import glob


def tile_num(fname):
    """ Extract tile number from file name. """
    l = os.path.splitext(fname)[0].split('_')   # fname -> list
    i = l.index('tile')   # i is the index in the list
    return int(l[i+1])

def tile_match(str1, str2):
    """ 
    des: matches tile indices 
    args:
        str1, str2:  Unix style pathname, e.g., 'root/dir/*'
    return:
        pair-wise files (have the same tile number).
    """
    # Get file names
    files1 = glob.glob(str1)
    files2 = glob.glob(str2)
    # Create output list
    f1out = []
    f2out = []
    # Loop trough files-1
    for file1 in files1:
        f1 = tile_num(file1)  # Get tile index
        # Loop trough files-2
        for file2 in files2:
            f2 = tile_num(file2)   # Get tile index
            # Check if tiles have same index
            if f1 == f2:
                # Save if true
                f1out.append(file1)
                f2out.append(file2)
                break    # break loop: 'for file2 in files2'
    return f1out, f2out
