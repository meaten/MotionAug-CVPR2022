import os
import itertools
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--act_class', type=str)
    parser.add_argument('--num_threads', type=int)
    return parser.parse_args()


timeout = 60 * 60 * 24 * 3

args = parse_args()
num_thread = args.num_threads
cla = args.act_class

subject = ["bd", "bk", "dg", "tr", "mm"]
subject.sort()

num_workers = int(num_thread / len(subject))

resforcetype = "rootPD_weight_1"
base_cmd = ["python3", "mpi_run.py",
            "--arg_file", f"args/VAE_sampling_{cla}.txt",
            "--class", cla,
            "--resforcetype", resforcetype,
            "--num_workers", str(num_workers)]

timeout_opt = ["--timeout", str(timeout)]




opts = []
for s in itertools.combinations(subject, 4):
    outpath = os.path.join("models", "3dv", f"vaeaug_{cla}_cluster", resforcetype + "_" + "_".join(s))
    opt = ["--subject"] + list(s) + ["--output_path", outpath]
    
    model_path = os.path.join(outpath, "agent0_model.ckpt")
    if os.path.exists(model_path + ".meta"):
        opt += ["--model_files", model_path]
    opts.append(opt)

num_paralellized_cmd = int(num_thread / num_workers)
for ind in range(0, len(opts), num_paralellized_cmd):
    opts_paralelled = opts[ind:ind + num_paralellized_cmd]
    procs = [subprocess.Popen(base_cmd + opt + timeout_opt) for opt in opts_paralelled]
    [print(" ".join(base_cmd + opt + timeout_opt)) for opt in opts_paralelled]
    [p.wait() for p in procs]
