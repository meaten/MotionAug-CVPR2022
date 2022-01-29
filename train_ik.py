import os
import subprocess

timeout = 60 * 60 * 3
#timeout = 30
num_workers = 5
num_thread = 30

bvh_dir = "data/bvh/hdm05_aligned_split/"
keyword = "grab"

base_cmd = ["python3", "mpi_run.py",
            "--arg_file", f"args/ik_fanshape_{keyword}.txt",
            "--num_workers", str(num_workers)]

timeout_opt = ["--timeout", str(timeout)]

opts = []
for file in os.listdir(bvh_dir):
    if ".bvh" in file and keyword in file:
        bvhpath = os.path.join(bvh_dir, file)
        outpath = os.path.join("models", "cvpr2022", "ikaug_" + keyword, file[:-4])
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
