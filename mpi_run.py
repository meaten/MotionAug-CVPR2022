import os
import sys
import subprocess
from util.logger import Logger
from DeepMimic import build_arg_parser


def main():
    # Command line arguments
    args = sys.argv[1:]
    
    # build motion
    arg_parser = build_arg_parser(args)
    output_path = arg_parser.parse_string('output_path')
    if arg_parser.parse_bool("build_from_bvh"):
        arg_parser._table["build_from_bvh"] = ["false"]
        args.extend(['--build_from_bvh', 'false'])
        args.extend(['--character_files', arg_parser._table["character_files"][0]])
        args.extend(['--char_ctrl_files', arg_parser._table["char_ctrl_files"][0]])
        args.extend(["--motion_file", arg_parser._table["motion_file"][0]])
        
    # dump args for test execution
    arg_parser._table['model_files'] = [os.path.join(output_path, 'agent0_model.ckpt')]
    arg_parser._table['train_agents'] = ['false']
    arg_parser._table['hold_end_frame'] = ['0.0']
    arg_parser.dump_file(os.path.join(output_path, 'args.txt'))
    
    num_workers = arg_parser.parse_int('num_workers', 1)
    timeout = arg_parser.parse_int("timeout", default=-1)
    assert(num_workers > 0)
    
    # discard argparser change and build arg_parser again in DeepMimic_Optimizer.py
    Logger.print('Running with {:d} workers'.format(num_workers))
    cmd = 'mpiexec -n {:d} --timeout {:d} python DeepMimic_Optimizer.py '.format(num_workers, timeout)
    cmd += ' '.join(args)
    subprocess.run(cmd, shell=True)
    return


if __name__ == '__main__':
    main()
