import itertools
import subprocess


base_cmds = [["python3", "main.py"]]
#gpu = [3,4,5,6,7,8,9]
gpu = [3, 4, 5, 6, 7, 8]
per_gpu = 5

aug_mode = ["VAE_IK_PHYSICAL_cluster_OFFSET_NN"]

actions = ["kick"]

model_types = ["GCN"]

subject = ["bd"]

batch_size = str(64)

cmds = []
for g, base, m, aug, s, a, in itertools.product(gpu, base_cmds, model_types, aug_mode, subject, actions):
    exp = aug + "_" + s
    cmd = base + ['--ckpt',
                  f"checkpoint_tune",
                  "--gpu",
                  str(g),
                  "--exp",
                  exp,
                  "--test_subj",
                  s,
                  "--actions",
                  a,
                  "--aug_mode",
                  aug,
                  "--train_batch",
                  batch_size,
                  "--test_batch",
                  batch_size,
                  "--reward_thres",
                  str(0.7),
                  "--model_type",
                  m,
                  "--tune",
                  str(True)]
    cmds.append(cmd)

all_cmds = []
for cmd in cmds:
    [print(" ".join(cmd))]
    all_cmds.extend([cmd for i in range(per_gpu)])

procs = [subprocess.Popen(cmd) for cmd in all_cmds]
[p.wait() for p in procs]
