import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--act_class', type=str)
    parser.add_argument('--num_threads', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=30)
    args = parser.parse_args()

    return args

timeout = 60 * 60 * 8
bvh_dir = "data/bvh/hdm05_aligned_split/"

args = parse_args()
num_workers = args.num_workers
num_thread = args.num_threads
keyword = args.act_class

base_cmd = ["python3", "mpi_run.py",
            "--arg_file", f"args/ik_fanshape_{keyword}.txt",
            "--num_workers", str(num_workers)]

timeout_opt = ["--timeout", str(timeout)]

opts = []
for file in os.listdir(bvh_dir):
    if ".bvh" in file and keyword in file:
        bvhpath = os.path.join(bvh_dir, file)
        outpath = os.path.join("models", "ikaug_" + keyword, file[:-4])
        opt = ["--bvh", bvhpath, "--output_path", outpath]
        
        model_path = os.path.join(outpath, "agent0_model.ckpt")
        if os.path.exists(model_path + ".meta"):
            opt += ["--model_files", model_path]
        opts.append(opt)

num_paralellized_cmd = int(num_thread / num_workers)
for ind in range(0, len(opts), num_paralellized_cmd):
    opts_paralelled = opts[ind:ind + num_paralellized_cmd]
    for opt in opts_paralelled:
        print(" ".join(base_cmd + opt + timeout_opt))
    procs = [subprocess.Popen(base_cmd + opt + timeout_opt) for opt in opts_paralelled]
    [p.wait() for p in procs]
