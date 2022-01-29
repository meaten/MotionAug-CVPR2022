import os
from enum import Enum, auto, unique
from re import A
from numpy.lib.npyio import NpzFile
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy
from utils import data_utils


@unique
class AugMode(Enum):
    NOAUG = auto()
    NOISE = auto()
    VAE = auto()
    VAE_cluster = auto()
    IK = auto()
    VAE_IK = auto()
    VAE_IK_cluster = auto()
    VAE_PHYSICAL = auto()
    VAE_PHYSICAL_cluster = auto()
    IK_PHYSICAL = auto()
    VAE_IK_PHYSICAL = auto()
    VAE_IK_PHYSICAL_cluster = auto()
    IK_PHYSICAL_OFFSET = auto()
    IK_PHYSICAL_OFFSET_NN = auto()
    VAE_PHYSICAL_OFFSET_NN = auto()
    VAE_PHYSICAL_cluster_OFFSET_NN = auto()
    VAE_IK_PHYSICAL_OFFSET_NN = auto()
    VAE_IK_PHYSICAL_cluster_OFFSET_NN = auto()
    IK_OFFSET_NN = auto()
    VAE_OFFSET_NN = auto()
    VAE_IK_OFFSET_NN = auto()
    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return AugMode[s]
        except KeyError:
            raise ValueError()


class HDM05motion(Dataset):

    def __init__(self, path_to_data, actions, input_n=12, output_n=30, dct_n=20, split=0, sample_rate=2, data_mean=0,
                 data_std=0, mode: AugMode = AugMode.NOAUG, test_subj: str = 'tr', reward_thres: float = None, do_dct=True):
        """
        read HDM05 data to get the dct coefficients.
        :param path_to_data:
        :param actions: actions to read
        :param input_n: past frame length
        :param output_n: future frame length
        :param dct_n: number of dct coeff. used
        :param split: 0 train, 1 test, 2 validation
        :param sample_rate: 2
        :param data_mean: mean of expmap
        :param data_std: standard deviation of expmap
        """

        self.mode = mode
        self.dct = do_dct

        self.path_to_data = path_to_data
        self.split = split
        # subs = [['bd', 'bk', 'dg'], ['mm'], ['tr']]
        all_subs = ['bd', 'bk', 'dg', 'mm', 'tr']
        all_subs.sort()
        all_subs.remove(test_subj)

        subs = [all_subs, all_subs, [test_subj]]

        acts = data_utils.define_actions_hdm05(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        if split == 0:
            is_test = False
        else:
            is_test = True
            mode = AugMode.NOAUG

        noaug_path = [os.path.join(path_to_data, "dataset_split.npz")]
        vae_path = [os.path.join(path_to_data, f"dataset_VAE_kin_{act}.npz") for act in acts]
        vae_cluster_path = [os.path.join(path_to_data, f"dataset_VAE_kin_{act}_cluster.npz") for act in acts]
        ik_path = [os.path.join(path_to_data, f"dataset_IK_kin_{act}.npz") for act in acts]
        if reward_thres is None or reward_thres == 0.7:
            vae_phys_path = [os.path.join(path_to_data, f"dataset_VAE_phys_{act}.npz") for act in acts]
            vae_phys_cluster_path = [os.path.join(path_to_data, f"dataset_VAE_phys_{act}_cluster.npz") for act in acts]
            ik_phys_path = [os.path.join(path_to_data, f"dataset_IK_phys_{act}.npz") for act in acts]
            ik_phys_offset_path = [os.path.join(path_to_data, f"dataset_IK_phys_{act}_offset_Linear.npz") for act in acts]
            ik_phys_offset_NN_path = [os.path.join(path_to_data, f"dataset_IK_phys_{act}_offset_NN.npz") for act in acts]
            vae_phys_offset_NN_path = [os.path.join(path_to_data, f"dataset_VAE_phys_{act}_offset_NN.npz") for act in acts]
            vae_phys_cluster_offset_NN_path = [os.path.join(path_to_data, f"dataset_VAE_phys_{act}_cluster_offset_NN.npz") for act in acts]
            ik_kin_offset_NN_path = [os.path.join(path_to_data, f"dataset_IK_kin_{act}_offset_NN.npz") for act in acts]
            vae_kin_offset_NN_path = [os.path.join(path_to_data, f"dataset_VAE_kin_{act}_offset_NN.npz") for act in acts]
        """
        elif reward_thres == 0.9:
            vae_phys_path = os.path.join(path_to_data, "dataset_vae_phys_th09.npz")
            ik_phys_path = os.path.join(path_to_data, "dataset_ik_phys_th09.npz")
        elif reward_thres == 0.5:
            vae_phys_path = os.path.join(path_to_data, "dataset_vae_phys_th05.npz")
            ik_phys_path = os.path.join(path_to_data, "dataset_ik_phys_th05.npz")
        else:
            raise ValueError
        """

        npz_path = noaug_path
        if mode is AugMode.NOAUG:
            pass
        elif mode is AugMode.NOISE:
            pass
        elif mode is AugMode.VAE:
            npz_path += vae_path
        elif mode is AugMode.VAE_cluster:
            npz_path += vae_cluster_path
        elif mode is AugMode.IK:
            npz_path += ik_path
        elif mode is AugMode.VAE_IK:
            npz_path += vae_path + ik_path
        elif mode is AugMode.VAE_IK_cluster:
            npz_path += vae_cluster_path + ik_path
        elif mode is AugMode.VAE_PHYSICAL:
            npz_path += vae_phys_path
        elif mode is AugMode.VAE_PHYSICAL_cluster:
            npz_path += vae_phys_cluster_path
        elif mode is AugMode.IK_PHYSICAL:
            npz_path += ik_phys_path
        elif mode is AugMode.VAE_IK_PHYSICAL:
            npz_path += vae_phys_path + ik_phys_path
        elif mode is AugMode.VAE_IK_PHYSICAL_cluster:
            npz_path += vae_phys_cluster_path + ik_phys_path
        elif mode is AugMode.IK_PHYSICAL_OFFSET:
            npz_path += ik_phys_offset_path
        elif mode is AugMode.IK_PHYSICAL_OFFSET_NN:
            npz_path += ik_phys_offset_NN_path
        elif mode is AugMode.VAE_PHYSICAL_OFFSET_NN:
            npz_path += vae_phys_offset_NN_path
        elif mode is AugMode.VAE_PHYSICAL_cluster_OFFSET_NN:
            npz_path += vae_phys_cluster_offset_NN_path
        elif mode is AugMode.VAE_IK_PHYSICAL_OFFSET_NN:
            npz_path += vae_phys_offset_NN_path + ik_phys_offset_NN_path
        elif mode is AugMode.VAE_IK_PHYSICAL_cluster_OFFSET_NN:
            npz_path += vae_phys_cluster_offset_NN_path + ik_phys_offset_NN_path
        elif mode is AugMode.IK_OFFSET_NN:
            npz_path += ik_kin_offset_NN_path
        elif mode is AugMode.VAE_OFFSET_NN:
            npz_path += vae_kin_offset_NN_path
        elif mode is AugMode.VAE_IK_OFFSET_NN:
            npz_path += ik_kin_offset_NN_path + vae_kin_offset_NN_path
        else:
            raise ValueError

        all_seqs, idxs, dim_ignore, dim_used, data_mean, data_std = data_utils.load_data_hdm05(npz_path,
                                                                                               subjs, acts,
                                                                                               sample_rate,
                                                                                               input_n, output_n,
                                                                                               data_mean=data_mean,
                                                                                               data_std=data_std,
                                                                                               is_test=is_test)

        self.data_mean = data_mean
        self.data_std = data_std
        
        self.idxs = idxs
        
        self.input_n = input_n
        self.output_n = output_n
        self.seq_len = input_n + output_n
        
        # first 6 elements are global translation and global rotation
        # dim_used = np.where(np.asarray(dim_use) > 5)[0]
        self.dim_used = dim_used
        
        self.dct_n = dct_n
        self.dct_m, _ = data_utils.get_dct_matrix(self.seq_len)
        self.dct_m = self.dct_m[:dct_n]

        self.all_seqs = all_seqs
    
        if self.mode == AugMode.NOISE:
            self.std = None
            all_inputs = [self.__getitem__(i)[0] for i in range(self.__len__())]
            self.std = np.std(all_inputs, axis=0) * 0.01

    def __len__(self):
        return self.idxs[-1]

    def __getitem__(self, item):
        idx_seq, window = self.map_idx(item)
        seq = self.all_seqs[idx_seq][window]
        seq_return = deepcopy(seq)
        seq = seq[:, self.dim_used]  # frame, dim

        # padding the observed sequence so that it has the same length as observed + future sequence
        pad_idx = np.repeat([self.input_n - 1], self.output_n)
        i_idx = np.append(np.arange(0, self.input_n), pad_idx)
        if not self.dct:
            return seq[:self.input_n, :], seq[self.input_n-1:-1, :], seq_return[self.input_n:, :]
        input_dct_seq = np.matmul(self.dct_m, seq[i_idx, :])
        input_dct_seq = input_dct_seq.transpose()

        output_dct_seq = np.matmul(self.dct_m, seq)
        output_dct_seq = output_dct_seq.transpose()

        if self.mode == AugMode.NOISE and self.std is not None:
            return self.add_noise(input_dct_seq), output_dct_seq, seq_return
        else:
            return input_dct_seq, output_dct_seq, seq_return

    def add_noise(self, array):
        return array + np.random.normal(loc=0, scale=self.std)

    def map_idx(self, idx):
        idx_seq = len(self.idxs[idx >= self.idxs]) - 1
        idx_frame = idx - self.idxs[idx_seq]
        window = np.arange(idx_frame, idx_frame + self.seq_len)
        # assert (window >= 0).all()
        return idx_seq, window
