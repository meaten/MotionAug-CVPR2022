import os
import sys
from motionAE.src.Trainer_builder import build_Trainer
from motionAE.src.train import build_arg_parser

from util.bvh import bvhToMotion
from bvh import Bvh


class Sampler(object):
    def __init__(self, args):
        arg_parser = build_arg_parser(args)
        arg_parser._table['training'][0] = ['false']
        self.trainer = build_Trainer(arg_parser, gpu=False)
        self.trainer.load_model(gpu=False)
        
    def sample_MimicMotion(self, motion_path):
        bvhpath = motion_path[:-7]
        motions = self.trainer.sample(batch_size=1, gpu=False)
        names = [os.path.basename(bvhpath[:-4])]
        frametime = 1 / self.trainer.fps
        dirpath = os.path.dirname(bvhpath)
                
        data_path = self.trainer.data_path
        settings = self.trainer.settings
        args = {'supress': True, 'return_string': True}
        bvh_strings = self.trainer.write_bvhs(motions, names, frametime, dirpath, **args)
        assert(len(bvh_strings) == 1)
        bvh = Bvh(bvh_strings[0])
        
        return bvhToMotion(bvh, bvhpath, data_path, "", settings, return_string=True)
    
    
if __name__ == '__main__':
    args = sys.argv[1:]
    sampler = Sampler(args)
    print(sampler.sample_MimicMotion(""))
