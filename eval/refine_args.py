import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--indir", required=True, type=str)
parser.add_argument("--outdir", required=True, type=str)
parser.add_argument("--cmin", required=True, type=int)
parser.add_argument("--cmax", required=True, type=int)
parser.add_argument("--amin", default=-3, type=float)
parser.add_argument("--amax", default=3, type=float)
parser.add_argument("--anum", default=7, type=int)
