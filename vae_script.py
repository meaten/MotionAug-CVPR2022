import os
import subprocess
import itertools
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--act_class", help="action class")
    parser.add_argument("--training", choices=["true", "false"], default="true")
    parser.add_argument("--gpu", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    
    base_cmd = ["python3", "motionAE/src/train.py",
                "--training", args.training,
                "--gpu", str(args.gpu)]
    
    arg_files = ['./motionAE/args/lstmVAE_adversarial_cluster.txt']

    subject = sorted(["bd", "bk", "dg", "tr", "mm"])

    residual = ["false"]

    cla = [args.act_class]

    my_env = os.environ.copy()
    my_env["PYTHONPATH"] = "./"

    cmds = []
    for a, r, c in itertools.product(arg_files, residual, cla):
        for s in itertools.combinations(subject, 4):
            arc_list = ['--arg_file', a]
            sub_list = ["--subject"] + list(s)
            res_list = ["--residual", r]
            cla_list = ["--class", c]

            cmd = base_cmd + arc_list + sub_list + res_list + cla_list

            cmds.append(cmd)

    parallel = 5

    for i in range(0, len(cmds), parallel):
        cmds_sub = cmds[i:i + parallel]
        [print(" ".join(cmd)) for cmd in cmds_sub]
        procs = [subprocess.Popen(cmd, env=my_env) for cmd in cmds_sub]
        [p.wait() for p in procs]


if __name__ == "__main__":
    main()
