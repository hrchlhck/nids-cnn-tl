import pandas as pd

from collections import namedtuple
from pathlib import Path
from io import StringIO
from IPython import embed
from tqdm import tqdm
from typing import List
from multiprocessing import Process, cpu_count
FILEINFO = namedtuple('FILEINFO', ('view', 'year', 'month'))
ORIGIN = Path('/home/pedro/Downloads/')

# https://github.com/haloboy777/arfftocsv/blob/master/arffToCsv.py
def toCsv(text):
    data = False
    header = ""
    new_content = []
    for line in text:
        if not data:
            if "@ATTRIBUTE" in line or "@attribute" in line:
                attributes = line.split()
                if("@attribute" in line):
                    attri_case = "@attribute"
                else:
                    attri_case = "@ATTRIBUTE"
                column_name = attributes[attributes.index(attri_case) + 1]
                header = header + column_name + ","
            elif "@DATA" in line or "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                new_content.append(header)
        else:
            new_content.append(line)
    return new_content

def get_fileinfo(f: Path) -> FILEINFO:
    f_str = f.stem
    year = f_str[4:8]
    month = f_str[8:10]
    view = f_str.split("_")[-1]
    return FILEINFO(view, year, month)

def convert(year: int, p: Path, view: str):
    for month in tqdm(range(1, 13), total=12):
        month_str = str(month).zfill(2) 

        files = [f for f in p.iterdir() if get_fileinfo(f).month == month_str]

        df = pd.DataFrame()
        for file in files:
            with open(file, 'r') as fd:
                data = fd.read()
            
            data_csv = toCsv(map(lambda x: x.strip(), data.split("\n")))
            strio = StringIO("\n".join(data_csv))
            df = pd.concat([df, pd.read_csv(strio, on_bad_lines='skip')], ignore_index=True)
        
        df.to_csv(f"data/csv/{view[:-2]}/{view[:-2]}_{year}_{month_str}.csv", index=False)
        print("Done", view, year, month_str)

for view in ["MOORE_B", "VIEGAS_B", "ORUNADA_B", "NIGEL_B"]:
    processes: List[Process] = []
    for year in range(2014, 2020):
        p = ORIGIN / f"{year}_B/" / view
        processes.append(Process(target=convert, args=(year, p, view)))
    
    for process in processes:
        process.start()

    for process in processes:
        process.join()

