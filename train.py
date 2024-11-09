import os

import argparse
import random
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
import pytorch_warmup as warmup

from configs import *
from inference import *

from ofold.np import residue_constants

from flowmatch import flowmatcher

from model import main_network, genzyme, folding_network, esm3_network

from flowmatch.data import utils as du
from flowmatch.data import all_atom
from evaluation.metrics import *
from evaluation.loss import *
from data.utils import *
from data.loader import *
from data.data import *


def train_epoch(args, model, flow_matcher, optimizer, lr_scheduler, warmup_scheduler, dataloader):
    model.train()
    optimizer.zero_grad()

    n_data = 0
    avg_sample_time = 0
    total_loss = 0
    aa_loss = 0
    msa_loss = 0
    ec_loss = 0
    violation_loss = 0
    fape_loss = 0
    plddt_loss = 0
    tm_loss = 0
    pae_loss = 0
    rot_loss = 0
    trans_loss = 0
    bb_atom_loss = 0
    dist_mat_loss = 0
    inversefold_loss = 0
    struct_token_loss = 0
    seq_token_loss = 0
    trained_step = 0

    for train_feats in tqdm(dataloader):
        train_feats = {
                k: v.to(args.device) if torch.is_tensor(v) else v for k, v in train_feats.items()
            }
        

        if (
            args.embed.embed_self_conditioning
            and trained_step % 2 == 1
            ):
            with torch.no_grad():
                train_feats = self_conditioning_fn(args, model, train_feats)
        
        pred_frames = model.forward_frame(train_feats)
        loss_gen, aux_data = loss_fn(args, train_feats, pred_frames, flow_matcher)
        
        inversefold_feats = model.frames_to_inversefold(pred_frames)
        _, pred_aa = model.forward_inversefold(inversefold_feats)
        loss_invfold = loss_inversefold(args, train_feats, pred_aa)
        
        quantized_feats = model.quantize_frames(train_feats, pred_frames)
        loss_inpaint, loss_inpaint_bd = model.forward_inpaint_loss(train_feats, quantized_feats)
                
        loss = loss_gen + loss_invfold + loss_inpaint
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        with warmup_scheduler.dampening():
            lr_scheduler.step()
    
        n_data           += aux_data['examples_per_step']
        avg_sample_time  += aux_data['batch_time'].sum().item()
        total_loss       += loss.item() * aux_data['examples_per_step']
        aa_loss          += aux_data['aa_loss'] * aux_data['examples_per_step']
        msa_loss         += aux_data['msa_loss'] * aux_data['examples_per_step']
        ec_loss          += aux_data['ec_loss'] * aux_data['examples_per_step']
        violation_loss   += aux_data['violation_loss'] * aux_data['examples_per_step']
        fape_loss        += aux_data['fape_loss'] * aux_data['examples_per_step']
        plddt_loss       += aux_data['plddt_loss'] * aux_data['examples_per_step']
        tm_loss          += aux_data['tm_loss'] * aux_data['examples_per_step']
        pae_loss         += aux_data['pae_loss'] * aux_data['examples_per_step']
        rot_loss         += aux_data['rot_loss'] * aux_data['examples_per_step']
        trans_loss       += aux_data['trans_loss'] * aux_data['examples_per_step']
        bb_atom_loss     += aux_data['bb_atom_loss'] * aux_data['examples_per_step']
        dist_mat_loss    += aux_data['dist_mat_loss'] * aux_data['examples_per_step']
        inversefold_loss += loss_invfold * aux_data['examples_per_step']
        struct_token_loss += loss_inpaint_bd['nelbo'] * aux_data['examples_per_step']
        seq_token_loss    += loss_inpaint_bd['seq_nll'] * aux_data['examples_per_step']
        trained_step      += 1
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    

    total_loss = total_loss / n_data
    avg_sample_time = avg_sample_time / n_data
    aa_loss = aa_loss / n_data
    msa_loss = msa_loss / n_data
    ec_loss = ec_loss / n_data
    violation_loss = violation_loss / n_data
    fape_loss = fape_loss / n_data
    plddt_loss = plddt_loss / n_data
    tm_loss = tm_loss / n_data
    pae_loss = pae_loss / n_data
    rot_loss = rot_loss / n_data
    trans_loss = trans_loss / n_data
    bb_atom_loss = bb_atom_loss / n_data
    dist_mat_loss = dist_mat_loss / n_data
    inversefold_loss = inversefold_loss / n_data
    struct_token_loss = struct_token_loss / n_data
    seq_token_loss = seq_token_loss / n_data

    return total_loss, avg_sample_time, aa_loss, msa_loss, ec_loss, violation_loss, fape_loss, plddt_loss, tm_loss, pae_loss, rot_loss, trans_loss, bb_atom_loss, dist_mat_loss, inversefold_loss, struct_token_loss, seq_token_loss


def main(args):
    flow_matcher = flowmatcher.SE3FlowMatcher(args)
    gen_model = main_network.ProteinLigandNetwork(args)
    inversefold_model = folding_network.ProDesign_Model(args.inverse_folding)
    
    esm_model = esm3_network.CustomizedESM3(args.inpainting)
    vqvae_model = genzyme.initialize_structure_encoder(args, pretrained_structure_encoder=esm_model.get_structure_encoder())
    inpainting_model = genzyme.initialize_inpainting_module(args, esm_model, vqvae_model=vqvae_model)
    if (args.gen_ckpt_path is not None) & (args.ckpt_path is None):
        print(f'loading generative model from {args.gen_ckpt_path}')
        checkpoint = torch.load(args.gen_ckpt_path, map_location='cpu', weights_only=True)
        model_state_dict = checkpoint["model_state_dict"]
        gen_model.load_state_dict(model_state_dict, strict=False)
        
        
    if (args.inversefold_ckpt_path is not None) & (args.ckpt_path is None):
        print(f'loading inverse folding model from {args.inversefold_ckpt_path}')
        checkpoint = torch.load(args.inversefold_ckpt_path, map_location='cpu', weights_only=False)
        inversefold_model.load_state_dict(checkpoint, strict=False)
        
        
    if (args.inpainting_ckpt_path is not None) & (args.ckpt_path is None):
        print(f'loading inpainting model from {args.inpainting_ckpt_path}')
        checkpoint = torch.load(args.inpainting_ckpt_path, map_location='cpu', weights_only=True)
        model_state_dict = checkpoint["model_state_dict"]
        inpainting_model.load_state_dict(model_state_dict, strict=False)
    
        
    model = genzyme.GENzyme(args, gen_model, inversefold_model, inpainting_model=inpainting_model)
    model = model.float()
    
    
    current_pointer = 0
    best_train_loss = float('inf')
    best_epoch = 0
    starting_epoch = 0
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Before loading pretrained, #model parameters {num_parameters}")
    
    if args.ckpt_path is not None:
        print(f'resume training for {args.ckpt_path}')
        checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=True)
        model_state_dict = checkpoint["model_state_dict"]
    
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)

        
    model = model.to(args.device)

    print('loading data...')
    trn_data = PdbDataset(
        args = args,
        gen_model = flow_matcher,
        is_training = True,
    )

    val_data = PdbDataset(
        args = args,
        gen_model = flow_matcher,
        is_training = False,
    )

    trn_loader = create_data_loader(
                    trn_data,
                    sampler=None,
                    length_batch=True,
                    batch_size=args.trn_batch_size,
                    shuffle=True,
                    num_workers=args.num_worker,
                    drop_last=False,
                )

    val_loader = create_data_loader(
                    val_data,
                    sampler=None,
                    length_batch=True,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_min, weight_decay=args.weight_decay)
    warmup_steps = len(trn_loader) * args.epochs
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=warmup_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    

    for epoch in range(args.epochs):
        ### Train
        print(f'#### TRAINING epoch {epoch}')
        total_loss, avg_sample_time, aa_loss, msa_loss, ec_loss, violation_loss, fape_loss, plddt_loss, tm_loss, pae_loss, rot_loss, trans_loss, bb_atom_loss, dist_mat_loss, inversefold_loss, struct_token_loss, seq_token_loss = train_epoch(args, model, flow_matcher, optimizer, lr_scheduler, warmup_scheduler, trn_loader)
        
        print(f'Train epoch: {epoch}, total_loss: {total_loss:.5f}, avg_time: {avg_sample_time:.5f}, aa_loss: {aa_loss:.5f}, msa_loss: {msa_loss:.5f}, ec_loss: {ec_loss:.5f}, violation_loss: {violation_loss:.5f}, fape_loss: {fape_loss:.5f}, plddt_loss: {plddt_loss:.5f}, tm_loss: {tm_loss:.5f}, pae_loss: {pae_loss:.5f}, rot_loss: {rot_loss:.5f}, trans_loss: {trans_loss:.5f}, bb_loss: {bb_atom_loss:.5f}, dist_mat_loss: {dist_mat_loss:.5f}, inverse_fold_loss: {inversefold_loss:.5f}, struct_token_loss: {struct_token_loss:.5f}, seq_token_loss: {seq_token_loss:.5f}')

        with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
            logger.write(f'Train epoch: {epoch}, total_loss: {total_loss:.5f}, avg_time: {avg_sample_time:.5f}, aa_loss: {aa_loss:.5f}, msa_loss: {msa_loss:.5f}, ec_loss: {ec_loss:.5f}, violation_loss: {violation_loss:.5f}, fape_loss: {fape_loss:.5f}, plddt_loss: {plddt_loss:.5f}, tm_loss: {tm_loss:.5f}, pae_loss: {pae_loss:.5f}, rot_loss: {rot_loss:.5f}, trans_loss: {trans_loss:.5f}, bb_loss: {bb_atom_loss:.5f}, dist_mat_loss: {dist_mat_loss:.5f}, inverse_fold_loss: {inversefold_loss:.5f}, struct_token_loss: {struct_token_loss:.5f}, seq_token_loss: {seq_token_loss:.5f}\n')
            logger.close()


        
        current_pointer += 1
        if total_loss < best_train_loss:
            best_train_loss = total_loss
            best_epoch = epoch
            current_pointer = 0

            torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        f'{args.checkpoint_dir}/alphaenzyme.ckpt',
                    )
            

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if current_pointer == args.early_stopping:
                break

    
if __name__ == "__main__":
    args = Args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.logger_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.eval.eval_dir, exist_ok=True)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        
    os.makedirs(args.logger_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.eval.eval_dir, exist_ok=True)
    
    args.flow_ec = False
    
    # uniform
    if args.discrete_flow_type == 'uniform':
        args.num_aa_type = 20
        args.masked_aa_token_idx = None


        if args.flow_msa:
            args.msa.num_msa_vocab = 64
            args.msa.masked_msa_token_idx = None

        if args.flow_ec:
            args.ec.num_ec_class = 7
            args.ec.masked_ec_token_idx = None
            

    # discrete
    elif args.discrete_flow_type == 'masking':
        args.num_aa_type = 21
        args.masked_aa_token_idx = 20
        args.aa_ot = False


        if args.flow_msa:
            args.msa.num_msa_vocab = 65
            args.msa.masked_msa_token_idx = 64
            args.msa_ot = False

        if args.flow_ec:
            args.ec.num_ec_class = 7
            args.ec.masked_ec_token_idx = 6

    else:
        raise ValueError(f'Unknown discrete flow type {args.discrete_flow_type}')
        

    args.date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    args.checkpoint_dir = os.path.join(args.ckpt_dir, args.date)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(f'{args.logger_dir}/{args.date}.txt', 'a') as logger:
        logger.write(f'{args}\n')
        logger.close()

    main(args)


