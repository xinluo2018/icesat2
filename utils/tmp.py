import argparse

# def get_args():
description = "Read ICESat-2 ATL06 data files."
parser = argparse.ArgumentParser(description=description)

parser.add_argument(    # positional parameter
    "ifilesss",
    metavar="ifilessss",
    type=str,
    nargs="+",
    help="input files to read (.h5).",
)

parser.add_argument(   # optional parameter, begin with - or --
        '-o',
        metavar=('outdir'),
        dest='outdir1',
        type=str,
        nargs=2,
        help='path to output folder',
        default=[""]
)
# return parser.parse_args()

# args = get_args()
args = parser.parse_args()
ifile = args.ifilesss
opath = args.outdir1

def main(ifile):
    print(type(ifile))
    print(opath)

if __name__ == "__main__":
    main(ifile)


