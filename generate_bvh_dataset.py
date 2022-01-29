import os
import itertools
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--aug", help="augmentation: Fixed_phys, Fixed_kin, IK_kin, IK_phys, VAE_kin, VAE_phys")
    parser.add_argument("--act_class", help="action class")
    parser.add_argument("--parallel", type=int, help="num of parallel process", default=5)  # not viable for IK_kin
    parser.add_argument("--outdir", help="directory path to output .npz file", default="evaluate/dataset")
    parser.add_argument("--rep", type=str, default="expmap")
    parser.add_argument("--debug", action="store_true")
    
    return parser.parse_args()


def cmd_bvh_kin(args):
    cmds = []
    output_path = f"models/3dv/{args.aug}_{args.act_class}_cluster"
    
    base_cmd = ["python3", "mpi_run.py",
                "--default_params_file", "args/default_record_kin.txt",
                "--num_workers", str(1),
                "--output_path", output_path]

    aug = args.aug
    if "Fixed" in aug:
        base_cmd += ["--arg_file", f"args/ik_fanshape_{args.act_class}.txt",
                     "--aug_num", str(1),
                     "--goal_shape", "Fixed"]
        
        dirpath = "data/bvh/hdm05_aligned_split/"
        filelist = os.listdir(dirpath)
        
        for file in filelist:
            if ".bvh" in file and args.act_class in file:
                bvh_arg = ["--bvh", os.path.join(dirpath, file)]
                cmd = base_cmd + bvh_arg
                cmds.append(cmd)
    elif "IK" in aug:
        base_cmd += ["--arg_file", f"args/ik_fanshape_{args.act_class}.txt",
                     "--aug_num", str(10)]
        
        dirpath = "data/bvh/hdm05_aligned_split/"
        filelist = os.listdir(dirpath)
        
        for file in filelist:
            if ".bvh" in file and args.act_class in file:
                bvh_arg = ["--bvh", os.path.join(dirpath, file)]
                cmd = base_cmd + bvh_arg
                cmds.append(cmd)

    elif "VAE_comp" in aug:
        base_cmd.remove(output_path)
        base_cmd.remove("--output_path")
        base_cmd += ["--arg_file", f"args/VAE_sampling_{args.act_class}.txt",
                     "--class", args.act_class,
                     "--aug_num", str(1000),
                     "--train_agents", "false"]
        
        subject = ["bd", "bk", "dg", "mm", "tr"]
        subject.sort()

        cfgs = ['./motionAE/args/cvpr/GAN.txt',
                 './motionAE/args/cvpr/lstmVAE.txt',
                 './motionAE/args/cvpr/lstmVAE_adversarial_frame.txt',
                 './motionAE/args/cvpr/lstmVAE_adversarial_sequence.txt',
                 './motionAE/args/cvpr/lstmVAE_sns.txt']
                 
        cfgs = ['./motionAE/args/cvpr/lstmVAE_sns.txt', './motionAE/args/cvpr/lstmVAE_adversarial.txt', './motionAE/args/cvpr/lstmVAE_adversarial_sns.txt']
        
        output_path = dict()
        for s in itertools.combinations(subject, 4):
            for c in cfgs:
                subj_arg = ["--subject"] + list(s)
                conf_arg = ["--sampler_arg_file"] + [c]
                path = f"./models/3dv/{args.aug}/{os.path.basename(c)}"
                out_arg = ["--output_path", path]
                cmd = base_cmd + subj_arg + conf_arg + out_arg
                cmds.append(cmd)

                output_path[c] = path
                
    elif "VAE" in aug:
        base_cmd += ["--arg_file", f"args/VAE_sampling_{args.act_class}.txt",
                     "--class", args.act_class,
                     "--aug_num", str(1000),
                     "--train_agents", "false"]
        
        subject = ["bd", "bk", "dg", "mm", "tr"]
        subject.sort()
        
        for s in itertools.combinations(subject, 4):
            subj_arg = ["--subject"] + list(s)
            cmd = base_cmd + subj_arg
            cmds.append(cmd)
        
    else:
        raise ValueError
    
    return cmds, output_path


def cmd_bvh_phys(args):
    cmds = []
    output_path = f"models/3dv/{args.aug}_{args.act_class}"
    
    base_cmd = ["python3", "mpi_run.py",
                "--output_path", output_path,
                "--num_workers", str(1),
                "--write_bvh", "true"]
    
    aug = args.aug
    if "Fixed" in aug:
        base_cmd += ["--aug_num", str(1), "--goal_shape", "Fixed"]

        dirpath = f"models/cvpr2022/ikaug_{args.act_class}/"
        if args.act_class in ["walk"]:
            dirpath = f"models/re_ikaug_{args.act_class}/"
        argfile = "args.txt"
        filelist = os.listdir(dirpath)

        for subdir in filelist:
            argpath = os.path.join(dirpath, subdir, argfile)
            if os.path.exists(argpath):
                arg = ["--arg_file", argpath]
                cmd = base_cmd + arg
                cmds.append(cmd)
                
    elif "IK" in aug:
        base_cmd += ["--aug_num", str(10)]
        
        dirpath = f"models/cvpr2022/ikaug_{args.act_class}/"
        if args.act_class in ["walk"]:
            dirpath = f"models/re_ikaug_{args.act_class}/"
        argfile = "args.txt"
        filelist = os.listdir(dirpath)
        
        for subdir in filelist:
            argpath = os.path.join(dirpath, subdir, argfile)
            if os.path.exists(argpath):
                arg = ["--arg_file", argpath]
                cmd = base_cmd + arg
                cmds.append(cmd)
                
    elif "VAE" in aug:
        base_cmd += ["--aug_num", str(1000)]
        
        dirpath = f"models/3dv/vaeaug_{args.act_class}_cluster/"
        argfile = "args.txt"
        filelist = os.listdir(dirpath)
        cmds = []
        for subdir in filelist:
            argpath = os.path.join(dirpath, subdir, argfile)
            if os.path.exists(argpath):
                arg = ["--arg_file", argpath]
                cmds.append(base_cmd + arg)
                
    else:
        raise ValueError
    
    return cmds, output_path


def gen_bvh(args):
    cmds = []
    output_path = ""
    
    aug = args.aug
    if "kin" in aug:
        cmds, output_path = cmd_bvh_kin(args)
        
    elif "phys" in aug:
        cmds, output_path = cmd_bvh_phys(args)

    else:
        raise ValueError
    
    para_cmd_exec(cmds, args)
    
    return output_path


def gen_npz(bvh_dir, args):
    cmd = ["python", "data/bvh/bvhToNp.py",
           "--dirpath", os.path.join(bvh_dir, "bvh"),
           "--outpath", os.path.join(args.outdir, f"dataset_{args.aug}_{args.act_class}.npz"),
           "--rep", args.rep,
           '--parallel', str(args.parallel)]
    
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = "./"
    if not args.debug:
        proc = subprocess.Popen(cmd, env=my_env)
        proc.wait()

def gen_npz_dict(bvh_dir, args):
    cmds=[]
    for k in bvh_dir:
        cmds.append(["python", "data/bvh/bvhToNp.py",
            "--dirpath", os.path.join(bvh_dir[k], "bvh"),
            "--outpath", os.path.join(args.outdir, "VAE_comp", f"dataset_{os.path.basename(k)}.npz"),
            "--rep", args.rep,
            '--parallel', str(args.parallel)])
        
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = "./"
    for cmd in cmds:
        proc = subprocess.Popen(cmd, env=my_env)
        proc.wait()

def para_cmd_exec(cmds, args):
    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = "./"
    
    #parallel = args.parallel if not args.aug == "IK_kin" else 1
    parallel = args.parallel
    
    for i in range(0, len(cmds), parallel):
        cmds_sub = cmds[i:i + parallel]
        [print(" ".join(cmd)) for cmd in cmds_sub]
        if not args.debug:
            procs = [subprocess.Popen(cmd, env=my_env) for cmd in cmds_sub]
            [p.wait() for p in procs]


def main():
    args = parse_args()
    bvh_dir = gen_bvh(args)
    if type(bvh_dir) != dict:
        gen_npz(bvh_dir, args)
    else:
        gen_npz_dict(bvh_dir, args)

    
if __name__ == "__main__":
    main()
