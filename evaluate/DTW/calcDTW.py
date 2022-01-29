from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
import itertools
from joblib import Parallel, delayed


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--act_class", default="kick", type=str)
    return parser.parse_args()
    
    
def load_train(mode, subjects, act_class):
    actions = [act_class]
    
    path_list = get_path(mode, act_class)
    train = load_motion(path_list, subjects, actions)
    
    return train


def load_test(subjects, act_class):
    actions = [act_class]
    
    path_list = orig_path()
    test = load_motion(path_list, subjects, actions)
    
    return test
    

def load_motion(path_list, subjects, actions):
    subjects = list(subjects)
    subjects.append("".join(subjects))
    
    all_motions = []
    all_names = []
    all_subjects = []
    all_classes = []
    for path in path_list:
        with np.load(path, allow_pickle=True) as dataset:
            all_motions.append(dataset['motions'])
            all_names.append(dataset['names'])
            all_subjects.append(dataset['subjects'])
            all_classes.append(dataset['classes'])
        
    all_motions = np.concatenate(all_motions, axis=0)
    all_names = np.concatenate(all_names, axis=0)
    all_subjects = np.concatenate(all_subjects, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)
    
    sample_rate = 4
    
    sampled_seq = []
    for motion, subj, cla, name in zip(all_motions, all_subjects, all_classes, all_names):
        if subj in subjects and cla in actions:
            n, d = motion.shape
            used_frames = range(0, n, sample_rate)
            the_sequence = np.array(motion[used_frames, :])
            sampled_seq.append(the_sequence)
            
    return sampled_seq
            
    
def get_path(mode, act_class):
    path_list = orig_path()
    if mode == "train":
        pass
    elif mode == "vae":
        path_list += [f"../dataset/dataset_VAE_kin_{act_class}.npz"]
    elif mode == "ik":
        path_list += [f"../dataset/dataset_IK_kin_{act_class}.npz"]
    elif mode == "vae_ik":
        path_list += [f"../dataset/dataset_VAE_kin_{act_class}.npz", f"../dataset/dataset_IK_kin_{act_class}.npz"]
    elif mode == "vae_phys":
        path_list += [f"../dataset/dataset_VAE_phys_{act_class}.npz"]
    elif mode == "ik_phys":
        path_list += [f"../dataset/dataset_IK_phys_{act_class}.npz"]
    elif mode == "vae_ik_phys":
        path_list += [f"../dataset/dataset_VAE_phys_{act_class}.npz", f"../dataset/dataset_IK_phys_{act_class}.npz"]
    elif mode == "fixed_phys":
        path_list = [f"dataset_Fixed_phys_{act_class}.npz"]
    elif mode == "fixed_phys_offset":
        path_list = [f"dataset_Fixed_phys_{act_class}_offset.npz"]
    elif mode == 'ik_offset':
        path_list += [f"../dataset/dataset_IK_kin_{act_class}_offset_NN.npz"]
    elif mode == 'vae_offset':
        path_list += [f"../dataset/dataset_VAE_kin_{act_class}_offset_NN.npz"]
    elif mode == "vae_ik_offset":
        path_list += [f"../dataset/dataset_IK_kin_{act_class}_offset_NN.npz", f"../dataset/dataset_VAE_kin_{act_class}_offset_NN.npz"]
    elif mode == 'ik_phys_offset':
        path_list += [f"../dataset/dataset_IK_phys_{act_class}_offset_NN.npz"]
    elif mode == 'vae_phys_offset':
        path_list += [f"../dataset/dataset_VAE_phys_{act_class}_offset_NN.npz"]
    elif mode == "vae_ik_phys_offset":
        path_list += [f"../dataset/dataset_IK_phys_{act_class}_offset_NN.npz", f"../dataset/dataset_VAE_phys_{act_class}_offset_NN.npz"]
    else:
        raise ValueError
    
    return path_list

        
def orig_path():
    return ["../dataset/dataset_split.npz"]


def addnoise(frame):
    return frame + np.random.normal(loc=0.0, scale=np.sqrt(10), size=len(frame))


def addnoise_to_motion(motion):
    length = len(motion)
    
    for i in range(int(length / 7 * 2)):
        motion[i] = addnoise(motion[i])
        
    return motion


def calcDTW(seq1, seq2, noise=False):
    seq1 = seq1[:, 6:]
    seq2 = seq2[:, 6:]
    if noise:
        seq1 = addnoise_to_motion(seq1)
        seq2 = addnoise_to_motion(seq2)
    dist = fastdtw(seq1, seq2, dist=2)[0] / max(len(seq1), len(seq2))
    dist = np.sqrt(dist) / 180 * np.pi
    return dist


def dist_to_all(seq, seq_arr):
    return [calcDTW(seq, seq_) for seq_ in seq_arr]


def main():
    args = parse_args()
    act_class = args.act_class
    
    subject = sorted(["bd", "bk", "dg", "tr", "mm"])
    #mode = ["train", "vae_naive", "vae_adv", "vae_interp", "vae", "ik", "vae_ik", "vae_phys", "ik_phys", "vae_ik_phys"]
    #mode = ["train", "fixed_phys", "fixed_phys_offset", "ik", "ik_phys", "ik_phys_offset"]
    mode = ["train", "ik", "vae", "vae_ik", "ik_phys", "vae_phys", "vae_ik_phys", "ik_offset", "vae_offset", "vae_ik_offset", "ik_phys_offset", "vae_phys_offset", "vae_ik_phys_offset"]
    #mode = ["ik_phys", "ik_phys_offset"]
    mode = ["ik", "vae", "vae_ik"]
    for m in mode:
        mean_min = []
        mean_mean = []
        for s in subject:
            train_subj = deepcopy(subject)
            train_subj.remove(s)
            test_subj = [s]
            train = load_train(mode=m, subjects=train_subj, act_class=act_class)
            test = load_test(subjects=test_subj, act_class=act_class)
            dists = np.array(Parallel(n_jobs=12)(delayed(dist_to_all)(seq, train) for seq in tqdm(test)))
            min_DTW = np.mean(np.min(dists, axis=1))
            mean_DTW = np.mean(dists)
            mean_min.append(min_DTW)
            mean_mean.append(mean_DTW)
            # print(s)
            # print(f"mode:\t{m}\tmean min DTW dist:\t{min_DTW}\tmean mean DTW dist:\t{mean_DTW}")
        print("mean")
        print(f"mode:\t{m}\tmean min DTW dist:\t{np.mean(mean_min)}\tmean mean DTW dist:\t{np.mean(mean_mean)}")
        

if __name__ == "__main__":
    main()
