import shutil
from tqdm import tqdm
import pymeshlab
import os
from pathlib import Path

dataset_path = "/Users/tix/itmo/diploma/dataset/ModelNet40"

def find_off_files(dataset_path):
    off_files = []
    for dir in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, dir)):
            for dir2 in os.listdir(os.path.join(dataset_path, dir)):
                if os.path.isdir(os.path.join(dataset_path, dir, dir2)):
                    for filename in os.listdir(os.path.join(dataset_path, dir, dir2)):
                        if filename.endswith(".off"):
                            off_files.append(os.path.join(dataset_path, dir, dir2, filename))
    return off_files

def off_to_obj(off_files):
    with tqdm(total=len(off_files), desc="Processing files", unit="file") as pbar:
        for off_file in off_files:
            file_path = Path(off_file).stem
            obj_file = os.path.join(os.path.dirname(off_file), file_path + ".obj")
            ms = pymeshlab.MeshSet()
            try:
                ms.load_new_mesh(off_file)
            except RuntimeError as e:
                pbar.update(1)
                continue

            ms.save_current_mesh(obj_file)
            os.remove(off_file)
            pbar.update(1)


def fix_off_files(off_files):
    for target_path in off_files:
        tmp_file_path = target_path + ".tmp"
        with open(target_path, "r") as file, open(tmp_file_path, "w") as tmp_file:
            line = file.readline().strip()
            if line[0] != "OFF":
                new_line = line[3:]
                tmp_file.write("OFF\n")
                tmp_file.write(new_line + "\n")
                shutil.copyfileobj(file, tmp_file)
                os.replace(tmp_file_path, target_path)


def conversion_dataset(dataset_path):
    off_to_obj(find_off_files(dataset_path))
    fix_off_files(find_off_files(dataset_path))
    off_to_obj(find_off_files(dataset_path))


conversion_dataset(dataset_path)
