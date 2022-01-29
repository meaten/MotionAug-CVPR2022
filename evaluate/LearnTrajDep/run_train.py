import itertools
import subprocess


base_cmds = [["python3", "main.py"], ["python3", "main_eval.py"]]
#base_cmds = [["python3", "main_eval.py"]]
gpu = str(0)

"""
aug_mode = ["NOAUG", "NOISE", 
            "VAE", "VAE_PHYSICAL", "VAE_OFFSET_NN", "VAE_PHYSICAL_OFFSET_NN",
            "IK", "IK_PHYSICAL", "IK_OFFSET_NN", "IK_PHYSICAL_OFFSET_NN",
            "VAE_IK", "VAE_IK_PHYSICAL", "VAE_IK_OFFSET_NN", "VAE_IK_PHYSICAL_OFFSET_NN"]
"""
aug_mode = ["NOAUG", "NOISE",
            "VAE_IK_cluster", "VAE_IK_PHYSICAL", "VAE_IK_PHYSICAL_OFFSET_NN"]
aug_mode = ["VAE_IK_cluster"]
#aug_mode = ["NOAUG", "NOISE", "VAE_cluster", "IK", "VAE_IK_cluster"]
#aug_mode = ["VAE_IK", "VAE_PHYSICAL", "IK_PHYSICAL", "VAE_IK_PHYSICAL"]

#aug_mode = ["VAE", "VAE_IK"]
#aug_mode = ["VAE_PHYSICAL","VAE_IK_PHYSICAL"]
#aug_mode = ["NOAUG", "IK", "IK_PHYSICAL", "IK_PHYSICAL_OFFSET"]
#aug_mode = [
#    "VAE_PHYSICAL", "VAE_PHYSICAL_OFFSET_NN",
#    "IK_PHYSICAL", "IK_PHYSICAL_OFFSET_NN",
#    "VAE_IK_PHYSICAL", "VAE_IK_PHYSICAL_OFFSET_NN"]

#aug_mode = ["VAE_IK_PHYSICAL_cluster_OFFSET_NN"]
"""
aug_mode  = ["VAE_cluster", "VAE_IK_cluster", 
             "VAE_PHYSICAL_cluster", "VAE_IK_PHYSICAL_cluster",
             "VAE_PHYSICAL_cluster_OFFSET_NN", "VAE_IK_PHYSICAL_cluster_OFFSET_NN"]
aug_mode  = ["VAE_IK_cluster", 
             "VAE_PHYSICAL_cluster", "VAE_IK_PHYSICAL_cluster",
             "VAE_PHYSICAL_cluster_OFFSET_NN", "VAE_IK_PHYSICAL_cluster_OFFSET_NN"]
"""

#actions = ["kick"]
#actions = ["punch"]
#actions = ["walk"]
#actions = ["deposit"]
actions = ["grab", "deposit", "jog"]
#actions = ["sneak", "throw"]

#model_types = ["seq2seq"]
#model_types = ["GCN", "seq2seq"]
model_types = ["transformer"]

subject = ["bd", "bk", "dg", "mm", "tr"]
subject.sort()

parallel = 5

batch_size = str(64)

cmds = []
for base, m, a, aug, s in itertools.product(base_cmds, model_types, actions, aug_mode, subject):
    exp = aug + "_" + s
    cmd = base + ['--ckpt',
                  f"checkpoint_{a}",
                  # f"checkpoint_punch",
                  "--gpu",
                  str(subject.index(s)),
                  #gpu,
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
                  "--load_tuned"]
    cmds.append(cmd)

for idx in range(0, len(cmds), parallel):
    cmds_para = cmds[idx:idx + parallel]
    [print(" ".join(cmd)) for cmd in cmds_para]
    procs = [subprocess.Popen(cmd) for cmd in cmds_para]
    [p.wait() for p in procs]
