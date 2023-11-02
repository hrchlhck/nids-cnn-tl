from pathlib import Path
from tqdm import tqdm

p = Path('/data/img_nids/mlp/NIGEL_2014_11.csv')

with open(p, 'r') as fp:
    print(fp.readline(100))