from util.arg_parser import ArgParser

from motionAE.src.motionGANTrainer import motionGANTrainer
from motionAE.src.motionCVAE2Trainer import motionCVAE2Trainer
from motionAE.src.motionCVAETrainer import motionCVAETrainer
from motionAE.src.motionVAEAdvInitFeedTrainer import motionVAEAdvInitFeedTrainer
from motionAE.src.motionVAEInitFeedTrainer import motionVAEInitFeedTrainer
from motionAE.src.motionVAEAdvTrainer import motionVAEAdvTrainer
from motionAE.src.motionVAETrainer import motionVAETrainer
from motionAE.src.motionAETrainer import motionAETrainer


def build_Trainer(arg_parser: ArgParser, **kwargs):
    arc = arg_parser.parse_string("architecture")
    
    print(f"architecture: {arc}")
    
    if 'GAN' in arc:
        return motionGANTrainer(arg_parser, **kwargs)
    elif 'CVAE2' in arc:
        return motionCVAE2Trainer(arg_parser, **kwargs)
    elif 'CVAE' in arc:
        return motionCVAETrainer(arg_parser, **kwargs)
    elif 'VAE_adv_initfeed' in arc:
        return motionVAEAdvInitFeedTrainer(arg_parser, **kwargs)
    elif 'VAE_adversarial' in arc:
        return motionVAEAdvTrainer(arg_parser, **kwargs)
    elif 'VAE_initfeed' in arc:
        return motionVAEInitFeedTrainer(arg_parser, **kwargs)
    elif 'VAE' in arc:
        return motionVAETrainer(arg_parser, **kwargs)
    elif 'AE' in arc:
        return motionAETrainer(arg_parser, **kwargs)
    else:
        raise ValueError(f'invalid architecture type {arc} is set')
