import sys
from motionAE.src.Trainer_builder import build_Trainer
from util.arg_parser import ArgParser


def build_arg_parser(args: list) -> ArgParser:
    arg_parser = ArgParser()
    arg_parser.load_args(args)
    
    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ
        
    return arg_parser


def main():
    args = sys.argv[1:]
    
    arg_parser = build_arg_parser(args)
    trainer = build_Trainer(arg_parser)
    
    if trainer.training:
        trainer.train()
    else:
        trainer.load_model()
        
    trainer.test()
    

if __name__ == '__main__':
    main()
