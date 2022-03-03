import os
import subprocess
import itertools

actions = ["sneak"]
#actions = ["kick"]
aug_mode = ["IK", "VAE"]
#aug_mode = ["VAE"]
#kin_phys = ["phys", "kin"]
kin_phys = ["phys"]

base_cmd = ["python", "debias.py", "--debiaser_type", "NN"]

parallel = 1

cmds = []
for aug, kp, a in itertools.product(aug_mode, kin_phys, actions):
    base_dir = "../dataset"
    cmd = base_cmd + ["--phys_data_npz",
                      os.path.join(base_dir, f"dataset_Fixed_{kp}_{a}.npz"),
                      "--aug_data_npz",
                      os.path.join(base_dir, f"dataset_{aug}_{kp}_{a}.npz"),
                      #os.path.join(base_dir, f"dataset_Fixed_{kp}_{a}.npz"),
                      "--act_class",
                      a]
    cmds.append(cmd)

for idx in range(0, len(cmds), parallel):
    cmds_para = cmds[idx:idx + parallel]
    [print(" ".join(cmd)) for cmd in cmds_para]
    procs = [subprocess.Popen(cmd) for cmd in cmds_para]
    [p.wait() for p in procs]
