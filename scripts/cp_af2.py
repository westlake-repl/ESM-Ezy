import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm

def copy_pdb_file(args):
    stage, name, af2_uniref50_path, overlap_path = args
    if not os.path.exists(os.path.join(af2_uniref50_path, stage, name + ".pdb")):
        raise FileNotFoundError("pdb file not found for {}".format(name))
    shutil.copyfile(os.path.join(af2_uniref50_path, stage, name + ".pdb"), os.path.join(overlap_path, name + ".pdb"))

def multi_process_copy_pdb_files(af2_uniref50_path, overlap_names, overlap_path):
    args_list = []
    for name in tqdm(overlap_names):
        if os.path.exists(os.path.join(overlap_path, name + ".pdb")):
            continue
        for stage in ["train", "valid"]:
            if os.path.exists(os.path.join(af2_uniref50_path, stage, name + ".pdb")):
                args_list.append((stage, name, af2_uniref50_path, overlap_path))
                break
        else:
            raise FileNotFoundError("pdb file not found for {}".format(name))

    with Pool(1024) as pool:
        pool.map(copy_pdb_file, args_list)

# Example usage
data_path = "/zhouxibin/workspaces/ESM-Ezy/data"
af2_uniref50_path = "/yuanfajie/AF2_Uniref50/"
overlap_path = os.path.join(data_path, "af_uniref50", "overlap")
overlap_names = []
with open(os.path.join(data_path, "af_uniref50", "overlap_names.txt"), "r") as f:
    for line in f:
        overlap_names.append(line.strip())
# reverse the order of overlap_names
overlap_names = overlap_names[::-1]
multi_process_copy_pdb_files(af2_uniref50_path, overlap_names, overlap_path)
