import os
import tensorflow as tf
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

import os
import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from env.deepmimic_env import DeepMimicEnv
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.util as Util
from util.bvh import bvhToMimic

# Dimensions of the window we are drawing into.
win_width = 800
win_height = int(win_width * 9.0 / 16.0)
reshaping = False

# anim
fps = 60
update_timestep = 1.0 / fps
display_anim_time = int(1000 * update_timestep)
animating = True

playback_speed = 1
playback_delta = 0.05

# FPS counter
prev_time = 0
updates_per_sec = 0

args = []
world = None

initial_flag = True
write_count = 0
trial_count = 0


def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)
    
    arg_file = arg_parser.parse_string('arg_file')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ, Logger.print('Failed to load args from: ' + arg_file)
    
    default_file_key = 'default_params_file'
    if default_file_key in arg_parser._table:
        default_params_file = arg_parser.parse_string(default_file_key)
        succ = arg_parser.load_file(default_params_file)
        assert succ, Logger.print('Failed to load args from: ' + arg_file)
    
    rand_seed_key = 'rand_seed'
    if (rand_seed_key in arg_parser._table):
        rand_seed = arg_parser.parse_int(rand_seed_key)
        rand_seed += 1000 * MPIUtil.get_proc_rank()
        Util.set_global_seeds(rand_seed)

    if 'bvh' in arg_parser._table:
        bvh = arg_parser.parse_string('bvh')
        build_from_bvh = arg_parser.parse_bool('build_from_bvh')
        if build_from_bvh:
            import shutil
            output_path = arg_parser.parse_string("output_path", default="data/bvh/test/")
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            resforcetype = arg_parser.parse_string("resforcetype", default="rootPD_weight_1")
            motion_name = bvhToMimic(bvh, output_path, resforcetype=resforcetype)
            arg_parser._table['character_files'] = [os.path.join(output_path, "character.txt")]
            arg_parser._table['char_ctrl_files'] = [os.path.join(output_path, "ctrl.txt")]
            arg_parser._table['motion_file'] = [os.path.join(output_path, motion_name)]
            basename = os.path.basename(bvh)
            bvhpath_copied = os.path.join(output_path, basename)
            try:
                shutil.copyfile(bvh, bvhpath_copied)
            except shutil.SameFileError:
                pass
            arg_parser._table['bvh'] = [bvhpath_copied]
    return arg_parser


def update_intermediate_buffer():
    if not (reshaping):
        if (win_width != world.env.get_win_width() or win_height != world.env.get_win_height()):
            world.env.reshape(win_width, win_height)

    return


def update_world(world, time_elapsed):
    num_substeps = world.env.get_num_update_substeps()
    timestep = time_elapsed / num_substeps
    num_substeps = 1 if (time_elapsed == 0) else num_substeps

    for i in range(num_substeps):
        world.update(timestep)

        valid_episode = world.env.check_valid_episode() or (not world.arg_parser.parse_bool('early_termination'))
        if valid_episode:
            end_episode = world.env.is_episode_end()
            if (end_episode):
                world.end_episode()
                if world.arg_parser.parse_bool('write_bvh'):
                    reward_threshold = world.arg_parser.parse_float('reward_threshold')
                    record_kin = world.arg_parser.parse_bool('record_kin')
                    write_bvh(world, reward_threshold=reward_threshold, record_kin=record_kin)
                world.reset()
                break
        else:
            if world.arg_parser.parse_bool('write_bvh'):
                reward_threshold = 1.1  # just fails to count up
                write_bvh(world, reward_threshold=reward_threshold)
            world.reset()
            break
    return


def draw():
    global reshaping

    update_intermediate_buffer()
    world.env.draw()
    
    glutSwapBuffers()
    reshaping = False

    return


def reshape(w, h):
    global reshaping
    global win_width
    global win_height

    reshaping = True
    win_width = w
    win_height = h

    return


def step_anim(timestep):
    global animating
    global world

    update_world(world, timestep)
    animating = False
    glutPostRedisplay()
    return


def reload():
    global world
    global args

    world = build_world(args, enable_draw=True)
    return


def reset():
    world.reset()
    return


def get_num_timesteps():
    global playback_speed

    num_steps = int(playback_speed)
    if (num_steps == 0):
        num_steps = 1

    num_steps = np.abs(num_steps)
    return num_steps


def calc_display_anim_time(num_timestes):
    global display_anim_time
    global playback_speed

    anim_time = int(display_anim_time * num_timestes / playback_speed)
    anim_time = np.abs(anim_time)
    return anim_time


def shutdown():
    global world

    Logger.print('Shutting down...')
    world.shutdown()
    sys.exit(0)
    return


def get_curr_time():
    curr_time = glutGet(GLUT_ELAPSED_TIME)
    return curr_time


def init_time():
    global prev_time
    global updates_per_sec
    prev_time = get_curr_time()
    updates_per_sec = 0
    return


def animate(callback_val):
    global prev_time
    global updates_per_sec
    global world

    counter_decay = 0

    if (animating):
        num_steps = get_num_timesteps()
        curr_time = get_curr_time()
        time_elapsed = curr_time - prev_time
        prev_time = curr_time

        timestep = -update_timestep if (playback_speed < 0) else update_timestep
        for i in range(num_steps):
            update_world(world, timestep)
        
        # FPS counting
        update_count = num_steps / (0.001 * time_elapsed)
        if (np.isfinite(update_count)):
            updates_per_sec = counter_decay * updates_per_sec + (1 - counter_decay) * update_count
            world.env.set_updates_per_sec(updates_per_sec)
            
        timer_step = calc_display_anim_time(num_steps)
        update_dur = get_curr_time() - curr_time
        timer_step -= update_dur
        timer_step = np.maximum(timer_step, 0)
        
        glutTimerFunc(int(timer_step), animate, 0)
        glutPostRedisplay()

    if (world.env.is_done()):
        shutdown()

    return


def toggle_animate():
    global animating

    animating = not animating
    if (animating):
        glutTimerFunc(display_anim_time, animate, 0)

    return


def change_playback_speed(delta):
    global playback_speed

    prev_playback = playback_speed
    playback_speed += delta
    world.env.set_playback_speed(playback_speed)

    if (np.abs(prev_playback) < 0.0001 and np.abs(playback_speed) > 0.0001):
        glutTimerFunc(display_anim_time, animate, 0)

    return


def toggle_training():
    global world

    world.enable_training = not world.enable_training
    if (world.enable_training):
        Logger.print('Training enabled')
    else:
        Logger.print('Training disabled')
    return


def keyboard(key, x, y):
    key_val = int.from_bytes(key, byteorder='big')
    world.env.keyboard(key_val, x, y)

    if (key == b'\x1b'):  # escape
        shutdown()
    elif (key == b' '):
        toggle_animate()
    elif (key == b'>'):
        step_anim(update_timestep)
    elif (key == b'<'):
        step_anim(-update_timestep)
    elif (key == b','):
        change_playback_speed(-playback_delta)
    elif (key == b'.'):
        change_playback_speed(playback_delta)
    elif (key == b'/'):
        change_playback_speed(-playback_speed + 1)
    elif (key == b'l'):
        reload()
    elif (key == b'r'):
        reset()
    elif (key == b't'):
        toggle_training()

    glutPostRedisplay()
    return


def mouse_click(button, state, x, y):
    world.env.mouse_click(button, state, x, y)
    glutPostRedisplay()


def mouse_move(x, y):
    world.env.mouse_move(x, y)
    glutPostRedisplay()
    
    return


def init_draw():
    glutInit()
    
    glutInitContextVersion(3, 2)
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
    glutInitContextProfile(GLUT_CORE_PROFILE)

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(win_width, win_height)
    glutCreateWindow(b'DeepMimic')
    return
    
    
def setup_draw():
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutTimerFunc(display_anim_time, animate, 0)

    reshape(win_width, win_height)
    world.env.reshape(win_width, win_height)
    
    return


def build_world(args, enable_draw, playback_speed=1):
    arg_parser = build_arg_parser(args)
    env = DeepMimicEnv(args, enable_draw)
    world = RLWorld(env, arg_parser)
    world.env.set_playback_speed(playback_speed)
    return world


def draw_main_loop():
    init_time()
    glutMainLoop()
    return


def write_bvh(world, record_kin=False, **kwargs):
    global initial_flag
    global write_count
    global trial_count
    
    filepath = ''
    mode = world.arg_parser.parse_string('mode')
    dir = os.path.join(world.arg_parser.parse_string('output_path'), "bvh")
    os.makedirs(dir, exist_ok=True)
    bvhpath = world.arg_parser.parse_string('bvh')
    gen_or_kin = "_kin" if record_kin else "_gen"
    if mode == 'IK':
        basename = os.path.basename(bvhpath)
        base, ext = os.path.splitext(basename)
        filepath = os.path.join(dir, base + gen_or_kin + str(write_count) + ext)
    elif mode == 'VAE':
        cla = world.arg_parser.parse_string('class')
        subject = world.arg_parser.parse_strings('subject')
        sampler_arg_file = os.path.basename(world.arg_parser.parse_string('sampler_arg_file'))
        filepath = os.path.join(dir, "HDM_" + "".join(subject) + "_" + cla + "_" + sampler_arg_file + gen_or_kin + str(write_count) + ".bvh")
        
    write_limit = world.arg_parser.parse_int('aug_num')
    if write_count >= write_limit or trial_count >= 10 * write_limit:
        sys.exit(0)
    
    if not initial_flag:
        write_count += world.agents[0].path.write_bvh(bvhpath, filepath, record_kin=record_kin, **kwargs)
        trial_count += 1
    else:
        initial_flag = False


def main():
    global args

    # Command line arguments
    args = sys.argv[1:]

    init_draw()
    reload()
    setup_draw()
    draw_main_loop()

    return


if __name__ == '__main__':
    main()
