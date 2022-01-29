#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""overall code framework is adapped from https://github.com/weigq/3d_pose_baseline_pytorch"""
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar
import pandas as pd
from copy import deepcopy

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.hdm05_motion import HDM05motion, AugMode
import utils.model as nnmodel
from utils.seq2seq_model import Seq2SeqModel
from utils.PoseTransformer import PoseTransformer
import utils.PoseGCN as PoseGCN
import utils.data_utils as data_utils


def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    is_cuda = torch.cuda.is_available()

    # define log csv file
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + \
        "_in{:d}_out{:d}_dctn{:d}".format(opt.input_n, opt.output_n, opt.dct_n)

    # create model
    print(f">>> creating model {opt.model_type}")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    # 48 nodes for angle prediction
    if opt.model_type == "GCN":
        if opt.load_tuned:
            opt = load_tune(opt)
        model = nnmodel.GCN(
            input_feature=dct_n,
            hidden_feature=opt.linear_size,
            p_dropout=opt.dropout,
            num_stage=opt.num_stage,
            node_n=51)
    elif opt.model_type == "seq2seq":
        model = Seq2SeqModel(architecture="tied",
                             source_seq_len=input_n,
                             target_seq_len=output_n,
                             rnn_size=64,  # hidden layer size
                             num_layers=1,
                             max_gradient_norm=5,
                             batch_size=opt.train_batch,
                             learning_rate=opt.lr,
                             learning_rate_decay_factor=opt.lr_decay,
                             loss_to_use="sampling_based",
                             number_of_actions=1,
                             one_hot=False,
                             residual_velocities=False,
                             dtype=torch.float32)

    elif opt.model_type == "transformer":
        encoder = PoseGCN.SimpleEncoder(
            n_nodes=17,
            input_features=3,
            model_dim=128,
            p_dropout=0.3
            )

        decoder = nn.Linear(128, 51)
        utils.weight_init(decoder, init_fn_=utils.xavier_init_)
        
        
        model = PoseTransformer(
            pose_dim=51,
            input_dim=51,
            model_dim=128,
            num_encoder_layers=4,
            num_decoder_layers=4,
            num_heads=4,
            dim_ffn=2048,
            dropout=0.3,
            target_seq_length=output_n,
            source_seq_length=input_n,
            init_fn=utils.xavier_init_,
            non_autoregressive=True,
            use_query_embedding=False,
            pre_normalization=True,
            predict_activity=False,
            num_activities=1,
            use_memory=False,
            pose_embedding=encoder,
            pose_decoder=decoder,
            query_selection=False,
            pos_encoding_params=(500, 10)
        )
    
    
    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel()
          for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # continue from checkpoint
    if opt.is_load:
        model_path_len = 'checkpoint/test/ckpt_main_gcn_muti_att_best.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(
            ">>> ckpt len loaded (epoch: {} | err: {})".format(
                start_epoch, err_best))

    # data loading
    print(">>> loading data")
    train_dataset = HDM05motion(
        path_to_data=opt.data_dir_hdm05,
        actions=opt.actions,
        input_n=input_n,
        output_n=output_n,
        split=0,
        sample_rate=sample_rate,
        dct_n=dct_n,
        mode=opt.aug_mode,
        test_subj=opt.test_subj,
        reward_thres=opt.reward_thres,
        do_dct= (opt.model_type == "GCN"))
    data_std = train_dataset.data_std
    data_mean = train_dataset.data_mean

    val_dataset = HDM05motion(
        path_to_data=opt.data_dir_hdm05,
        actions=opt.actions,
        input_n=input_n,
        output_n=output_n,
        split=1,
        sample_rate=sample_rate,
        data_mean=data_mean,
        data_std=data_std,
        dct_n=dct_n,
        mode=AugMode.NOAUG,
        test_subj=opt.test_subj,
        reward_thres=opt.reward_thres,
        do_dct= (opt.model_type == "GCN"))
    # load dadasets for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)

    acts = data_utils.define_actions_hdm05(opt.actions)
    test_data = dict()
    for act in acts:
        test_dataset = HDM05motion(
            path_to_data=opt.data_dir_hdm05,
            actions=act,
            input_n=input_n,
            output_n=output_n,
            split=2,
            sample_rate=sample_rate,
            data_mean=data_mean,
            data_std=data_std,
            dct_n=dct_n,
            mode=AugMode.NOAUG,
            test_subj=opt.test_subj,
            reward_thres=opt.reward_thres,
            do_dct= (opt.model_type == "GCN"))
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l, t_e, t_3d = train(opt.model_type, train_loader, model, optimizer, input_n=input_n,
                                       lr_now=lr_now, max_norm=opt.max_norm, is_cuda=is_cuda,
                                       dim_used=train_dataset.dim_used, dct_n=dct_n)
        ret_log = np.append(ret_log, [lr_now, t_l, t_e, t_3d])
        head = np.append(head, ['lr', 't_l', 't_e', 't_3d'])

        v_e = 0
        v_3d = 0
        v_e, v_3d = val(opt.model_type, val_loader, model, input_n=input_n,
                        is_cuda=is_cuda, dim_used=train_dataset.dim_used, dct_n=dct_n)

        ret_log = np.append(ret_log, [v_e, v_3d])
        head = np.append(head, ['v_e', 'v_3d'])

        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_e, test_3d = test(opt.model_type, test_data[act], model, input_n=input_n, output_n=output_n,
                                   is_cuda=is_cuda, dim_used=train_dataset.dim_used, dct_n=dct_n)
            ret_log = np.append(ret_log, test_e)
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(
                test_3d_head, [
                    act + '3d100', act + '3d200', act + '3d300', act + '3d400'])
            head = np.append(
                head, [
                    act + '100', act + '200', act + '300', act + '400'])
            if output_n > 12:
                head = np.append(head, [act + '700', act + '1000'])
                test_3d_head = np.append(test_3d_head,
                                         [act + '3d700', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        # update log file and save checkpoint
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(
                opt.ckpt +
                '/' +
                script_name +
                '.csv',
                header=head,
                index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if not np.isnan(v_e):
            is_best = v_e < err_best
            err_best = min(v_e, err_best)
        else:
            is_best = False
        file_name = [
            'ckpt_' +
            script_name +
            '_best.pth.tar',
            'ckpt_' +
            script_name +
            '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_e[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)
    
    return sum(test_e)


def train(
        model_type,
        train_loader,
        model,
        optimizer,
        input_n=20,
        dct_n=20,
        lr_now=None,
        max_norm=True,
        is_cuda=False,
        dim_used=[]):
    t_l = utils.AccumLoss()
    t_e = utils.AccumLoss()
    t_3d = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        # skip the last batch if only have one sample for batch_norm layers
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            targets = Variable(targets.cuda()).float()
            all_seq = Variable(all_seq.cuda()).float()
        if model_type == "GCN":
            outputs = model(inputs)
            n = outputs.shape[0]
            outputs = outputs.view(n, -1)
            # targets = targets.view(n, -1)
            n, seq_len, dim_full_len = all_seq.data.shape
            outputs = loss_funcs.dct2expmap(outputs, seq_len, dct_n, dim_used)
            loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n)
        elif model_type in ["seq2seq", "transformer"]:
            decoder_inputs = deepcopy(targets)
            targets = deepcopy(all_seq)
            if model_type == "seq2seq":
                """
                outputs = model(inputs, all_seq)
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)
                """
                outputs = model(inputs, decoder_inputs)
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)
            if model_type == "transformer":
                b, s, d = targets.size()
                targets = targets[:, 0, :].unsqueeze(1).expand(-1, s, -1).contiguous()
                outputs = model(inputs, decoder_inputs)
                outputs = outputs[0][0]
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)
            
        

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        n, _, _ = all_seq.data.shape

        # 3d error
        # m_err = loss_funcs.mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n)

        # angle space error
        e_err = loss_funcs.euler_error(
            outputs, all_seq, input_n, dim_used, dct_n)

        # update the training loss
        t_l.update(loss.cpu().data.numpy() * n, n)
        t_e.update(e_err.cpu().data.numpy() * n, n)
        # t_3d.update(m_err.cpu().data.numpy()[0] * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(
            i + 1, len(train_loader), time.time() - bt, time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg, t_e.avg, t_3d.avg


def test(
        model_type,
        train_loader,
        model,
        input_n=20,
        output_n=50,
        dct_n=20,
        is_cuda=False,
        dim_used=[]):
    N = 0
    t_l = 0
    if output_n == 12:
        eval_frame = [2, 5, 8, 11]
    else:
        raise(ValueError)

    t_e = np.zeros(len(eval_frame))
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            targets = Variable(targets.cuda()).float()
            all_seq = Variable(all_seq.cuda()).float()

        n, seq_len, dim_full_len = all_seq.data.shape
        if model_type == "GCN":
            outputs = model(inputs)
            n = outputs.shape[0]
            outputs = outputs.view(n, -1)
            # targets = targets.view(n, -1)
            outputs = loss_funcs.dct2expmap(outputs, seq_len, dct_n, dim_used)
            loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n)
        elif model_type in ["seq2seq", "transformer"]:
            decoder_inputs = deepcopy(targets)
            targets = deepcopy(all_seq)
            if model_type == "seq2seq":
                """
                outputs = model(inputs, all_seq)
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)
                """
                outputs = model(inputs, decoder_inputs)
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)
            if model_type == "transformer":
                b, s, d = targets.size()
                targets = targets[:, 0, :].unsqueeze(1).expand(-1, s, -1).contiguous()
                outputs = model(inputs, decoder_inputs)
                outputs = outputs[0][0]
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)

            
        outputs_exp = outputs

        pred_expmap = all_seq.clone()
        dim_used = np.array(dim_used)
        pred_expmap[:, :, dim_used] = outputs_exp
        pred_expmap = pred_expmap[:, input_n:,
                                  :].contiguous().view(-1, dim_full_len)
        targ_expmap = all_seq[:, input_n:, :].clone(
        ).contiguous().view(-1, dim_full_len)

        pred_expmap[:, 0:6] = 0
        targ_expmap[:, 0:6] = 0
        pred_expmap = pred_expmap.view(-1, 3)
        targ_expmap = targ_expmap.view(-1, 3)

        # get euler angles from expmap
        pred_eul = data_utils.rotmat2euler_torch(
            data_utils.expmap2rotmat_torch(pred_expmap))
        pred_eul = pred_eul.view(-1,
                                 dim_full_len).view(-1,
                                                    output_n,
                                                    dim_full_len)
        targ_eul = data_utils.rotmat2euler_torch(
            data_utils.expmap2rotmat_torch(targ_expmap))
        targ_eul = targ_eul.view(-1,
                                 dim_full_len).view(-1,
                                                    output_n,
                                                    dim_full_len)

        # get 3d coordinates
        targ_p3d = data_utils.expmap2xyz_torch_hdm05(
            targ_expmap.view(-1, dim_full_len)).view(n, output_n, -1, 3)
        pred_p3d = data_utils.expmap2xyz_torch_hdm05(
            pred_expmap.view(-1, dim_full_len)).view(n, output_n, -1, 3)

        # update loss and testing errors
        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_e[k] += torch.mean(torch.norm(pred_eul[:, j, :] -
                                 targ_eul[:, j, :], 2, 1)).cpu().data.numpy() * n

            t_3d[k] += torch.mean(torch.norm(targ_p3d[:,
                                                      j,
                                                      :,
                                                      :].contiguous().view(-1,
                                                                           3) - pred_p3d[:,
                                                                                         j,
                                                                                         :,
                                                                                         :].contiguous().view(-1,
                                                                                                              3),
                                             2,
                                             1)).cpu().data.numpy() * n

        t_l += loss.cpu().data.numpy() * n
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(
            i + 1, len(train_loader), time.time() - bt, time.time() - st)
        bar.next()
    bar.finish()
    return t_e / N, t_3d / N


def val(model_type, train_loader, model, input_n=20, dct_n=20, is_cuda=False, dim_used=[]):
    # t_l = utils.AccumLoss()
    t_e = utils.AccumLoss()
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            targets = Variable(targets.cuda()).float()
            all_seq = Variable(all_seq.cuda()).float()

        if model_type == "GCN":
            outputs = model(inputs)
            n = outputs.shape[0]
            outputs = outputs.view(n, -1)
            # targets = targets.view(n, -1)
            n, seq_len, dim_full_len = all_seq.data.shape
            outputs = loss_funcs.dct2expmap(outputs, seq_len, dct_n, dim_used)
            #loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n)
        elif model_type in ["seq2seq", "transformer"]:
            decoder_inputs = deepcopy(targets)
            targets = deepcopy(all_seq)
            if model_type == "seq2seq":
                """
                outputs = model(inputs, all_seq)
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)
                """
                outputs = model(inputs, decoder_inputs)
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)
            if model_type == "transformer":
                b, s, d = targets.size()
                targets = targets[:, 0, :].unsqueeze(1).expand(-1, s, -1).contiguous()
                outputs = model(inputs, decoder_inputs)
                outputs = outputs[0][0]
                loss = torch.mean(torch.sum(torch.abs(outputs - targets), dim=2).view(-1))
                outputs = torch.cat([inputs, outputs], axis=1)
                all_seq = torch.cat([inputs, targets], axis=1)

        n, _, _ = all_seq.data.shape
        # m_err = loss_funcs.mpjpe_error(outputs, all_seq, input_n, dim_used, dct_n)
        e_err = loss_funcs.euler_error(
            outputs, all_seq, input_n, dim_used, dct_n)

        # t_l.update(loss.cpu().data.numpy()[0] * n, n)
        t_e.update(e_err.cpu().data.numpy() * n, n)
        # t_3d.update(m_err.cpu().data.numpy()[0] * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(
            i + 1, len(train_loader), time.time() - bt, time.time() - st)
        bar.next()
    bar.finish()
    return t_e.avg, t_3d.avg


def tune(option):
    import optuna
    from copy import deepcopy

    def objective_with_arg(option):
        _option = deepcopy(option)

        def objective(trial):
            _option.linear_size = trial.suggest_int("linear_size", 8, 256)
            _option.num_stage = trial.suggest_int("num_stage", 1, 12)
            return main(_option)

        return objective

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(sampler=sampler, pruner=pruner,
                                direction="minimize",
                                storage=os.path.join(
                                    "sqlite:///", option.ckpt, "optuna.db"),
                                study_name="my_opt",
                                load_if_exists=True)
    import pdb;pdb.set_trace()
    study.optimize(objective_with_arg(option), n_trials=10, gc_after_trial=True)

    trial = study.best_trial

    print(trial.value, trial.params)

def load_tune(option):
    print("load params from optuna database")
    import optuna
    study = optuna.load_study(storage=os.path.join("sqlite:///", option.model_type,
                                                   "checkpoint_tune",
                                                   "VAE_IK_PHYSICAL_cluster_OFFSET_NN_bd",
                                                   "optuna.db"), study_name=None)

    trial_dict = study.best_trial.params
    
    for key in list(trial_dict.keys()):
        if type(trial_dict[key]) == str:
            exec(f"option.{key} = '{trial_dict[key]}'")
        else:
            exec(f"option.{key} = {trial_dict[key]}")
    
    return option

if __name__ == "__main__":
    option = Options().parse()
    if not option.tune:
        main(option)
    else:
        option.epochs = 5
        tune(option)
