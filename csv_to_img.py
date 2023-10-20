import sys
import cv2
import pickle as pk
import numpy as np

from pathlib import Path
from pyDeepInsight.image_transformer import ImageTransformer
from multiprocessing import Process, cpu_count
from IPython import embed
from tqdm import tqdm

MODEL_PATH = Path('data/models')

def save_model(model: object, name: str) -> None:
    with open(MODEL_PATH / name, 'wb') as fp:
        fp.write(pk.dumps(model))

def load_model(name: str) -> None:
    with open(MODEL_PATH / name, 'rb') as fp:
        return pk.loads(fp.read())

def train_image_transformer(df: np.ndarray) -> ImageTransformer:
    it = ImageTransformer(
        feature_extractor='tsne', 
        pixels=224,
    ) 

    X = df[:, :-1]

    it.fit(X)

    return it

def to_image(img_preffix: str, output: Path, df: np.ndarray, it: ImageTransformer, total: int, offset: int):
    y = df[:, -1]

    n = len(str(total))
    n_samples = y.shape[0]

    for i in tqdm(range(n_samples), total=n_samples):
        ii = i*offset
        feat = it.transform(df[i, :-1], img_format='scalar').squeeze()
        fname = f'{img_preffix}_normal_{str(ii).zfill(n)}.jpg'
        if y[i] == 1:
            fname = f'{img_preffix}_attack_{str(ii).zfill(n)}.jpg'
        
        img = cv2.normalize(feat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # type: ignore
        img = img.astype(np.uint8)
        
        cv2.imwrite(str(output / fname), img)
    
    print("Done process: ", offset)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: ./{sys.argv[0]} <csv_path> <output_path>")
        exit(1)
    
    CSV_PATH, OUTPUT_PATH = sys.argv[1:]
    CSV_PATH = Path(CSV_PATH)
    OUTPUT_PATH = Path(OUTPUT_PATH)

    df = np.loadtxt(CSV_PATH, delimiter=",", dtype=np.float32, skiprows=1)
    print("Loaded dataset", CSV_PATH)
    
    it = train_image_transformer(df)
    print("Trained transformer")

    n_cores = cpu_count()

    print("Total cores:", n_cores)
    processes = [
        Process(target=to_image, args=(CSV_PATH.stem, OUTPUT_PATH, _slice, it, df.shape[0], i+1)) 
        for i, _slice in enumerate(np.array_split(df, n_cores))
    ]
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Done")
    