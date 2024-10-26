import argparse
import random
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import OrderedDict

import pandas as pd

import torch
import numpy as np
import pandas as pd

from eval_configs import *
from inference import *

from ofold.np import residue_constants

from flowmatch import flowmatcher

from model import main_network
from flowmatch.data import utils as du
from evaluation.metrics import *
from evaluation.loss import *
from data.utils import *
from data.loader import *
from data.data import *

from Bio.PDB import PDBParser, PDBIO, Superimposer

def align_structure(ref_structure, sample_structure, prot_path=None):
    ref_model = ref_structure[0]
    sample_model = sample_structure[0]
    
    ref_atoms = []
    sample_atoms = []
    
    for ref_chain in ref_model:
        for ref_res in ref_chain:
            ref_atoms.append(ref_res['N'])
            ref_atoms.append(ref_res['CA'])
            ref_atoms.append(ref_res['C'])
            ref_atoms.append(ref_res['O'])
    
    # Do the same for the sample structure
    for sample_chain in sample_model:
        for sample_res in sample_chain:
            if sample_res.resname == 'UNL': continue
            sample_atoms.append(sample_res['N'])
            sample_atoms.append(sample_res['CA'])
            sample_atoms.append(sample_res['C'])
            sample_atoms.append(sample_res['O'])
    
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    super_imposer.apply(sample_model.get_atoms())

    if prot_path is not None:
        io = PDBIO()
        io.set_structure(sample_structure) 
        io.save(prot_path)

    return super_imposer



def change_sequence_index(structure, seq_idx):
    idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                residue.id = (residue.id[0], seq_idx[idx], residue.id[2])
                idx += 1
                
    return structure

def find_sequence_index(structure):
    residue_index = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_index.append(residue.id[1])
                
    return residue_index

def parse_prot_structure(prot_path, name='example'):
    parser = PDBParser(QUIET=True)
    prot_structure = parser.get_structure(name, prot_path)
    return prot_structure
    

def write_pdb_traj(args, feats_0, feats_1, parent_dir, pdb_name, substrate_name, sample_id=0):
    final_prot = {
                "t_1": feats_1["t"][0],
                "pos_1": feats_1["coord_traj"][0],
                "aa_1": feats_1["aa_traj"][0],
                "ec_1": feats_1["ec_traj"][0],
            }
    
    CA_IDX = residue_constants.atom_order["CA"]
    res_mask = du.move_to_np(feats_0["res_mask"].bool())
    flow_mask = du.move_to_np(feats_0["flow_mask"].bool())
    gt_aatype = du.move_to_np(feats_0["aatype"])
    gt_protein_pos = du.move_to_np(all_atom.to_atom37(ru.Rigid.from_tensor_7(feats_0["rigids_1"].type(torch.float32)))[0])
    gt_ec = du.move_to_np(feats_0["ec_1"])
    
    ligand_pos = du.move_to_np(feats_0["ligand_pos"])
    ligand_atom = du.move_to_np(feats_0["ligand_atom"])
    ligand_mask = du.move_to_np(feats_0["ligand_mask"].bool())
    batch_size = res_mask.shape[0]
    
    for i in range(batch_size):
        num_res = int(np.sum(res_mask[i]).item())
        unpad_flow_mask = flow_mask[i][res_mask[i]]
        unpad_protein = {
            "pos": final_prot['pos_1'][i][res_mask[i]],
            "aatype": final_prot['aa_1'][i][res_mask[i]],
            "ec": final_prot['ec_1'][i],
        }
        
        pred_aatype = unpad_protein["aatype"]
        pred_ec = unpad_protein["ec"].item()
        pred_portein_pos = unpad_protein["pos"]
        
        unpad_gt_protein_pos = gt_protein_pos[i][res_mask[i]]
        unpad_gt_aatype = gt_aatype[i][res_mask[i]]
        unpad_gt_ec = gt_ec[i][0]
    
        unpad_gt_ligand_pos = ligand_pos[i][ligand_mask[i]]
        unpad_gt_ligand_atom = ligand_atom[i][ligand_mask[i]]
    
        generated_dir = parent_dir
        generated_prot = f'{pdb_name}_{substrate_name}'
        prot_dir = os.path.join(generated_dir, generated_prot)
        if not os.path.isdir(prot_dir):
            os.makedirs(prot_dir, exist_ok=True)
                    
        prot_path = os.path.join(prot_dir, f"sample_{sample_id}.pdb")
    
        saved_path = write_prot_to_pdb(
                        prot_pos=pred_portein_pos,
                        file_path=prot_path,
                        aatype=pred_aatype,
                        no_indexing=True,
                        b_factors=np.tile(unpad_flow_mask[..., None], 37) * 100,
                    )
        

def parse_sampling_feats(args, meta_csv, gen_model, eval_row_id=0):
    protein, ligand, identifier, ec_class, _ = get_csv_row(args, meta_csv, eval_row_id)
    gt_bb_rigid = ru.Rigid.from_tensor_7(protein["rigids_1"])
    gt_trans, gt_rot = extract_trans_rots_mat(gt_bb_rigid)
    protein["trans_1"] = gt_trans
    protein["rot_1"] = gt_rot
    
    n_res = protein["aatype"].size(0)
    protein["seq_idx"] = torch.arange(n_res) + 1
    
    t = 0.
    protein["t"] = t
    
    gen_feats_t = gen_model.sample_ref(
        n_samples=n_res,
        flow_mask=None,
        as_tensor_7=True,
        center_of_mass=None,
    )
    protein.update(gen_feats_t)
    
    aatype_0 = torch.rand(n_res)
    aatype_t = gen_model.forward_masking(
        feat_0=aatype_0,
        feat_1=None,
        t=0.,
        mask_token_idx=args.masked_aa_token_idx,
        flow_mask=None,
    )
    protein["aatype_t"] = aatype_t
    
    if args.flow_msa:
        msa_1 = F.one_hot(protein["msa_1"], num_classes=args.msa.num_msa_vocab)
        msa_0 = torch.ones_like(msa_1) / args.msa.num_msa_vocab
        n_msa, n_token, _ = msa_1.size()
            
        msa_0 = torch.rand(n_msa, n_token)
        msa_t = gen_model.forward_masking(
            feat_0=msa_0,
            feat_1=None,
            t=t,
            mask_token_idx=args.msa.masked_msa_token_idx,
            flow_mask=None,
        ).reshape(n_msa, n_token)
        protein["msa_t"] = msa_t
    
    if args.flow_ec: 
        ec_1 = F.one_hot(protein["ec_1"], num_classes=args.ec.num_ec_class).reshape(-1)
        ec_0 = torch.ones_like(ec_1) / args.ec.num_ec_class
    
        ec_0 = torch.rand(1).reshape(-1)
        ec_t = gen_model.forward_masking(
            feat_0=ec_0,
            feat_1=None,
            t=t,
            mask_token_idx=args.ec.masked_ec_token_idx,
            flow_mask=None,
        )
        protein["ec_t"] = ec_t
    
    
    final_feats = {}
    for k, v in protein.items():
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        
        if k in {"residx_atom14_to_atom37", "atom37_pos", "atom14_pos", "atom37_mask"}:
            continue
    
        else:
            final_feats[k] = v
    
    final_feats.update(ligand)
    return final_feats


def sampling_inference(
    args,
    init_feats,
    gen_model,
    main_network,
    min_t = 0.,
    max_t = 1.,
    num_t = 100,
    center = True,
    self_condition = True,
    aa_do_purity = True,
    msa_do_purity = True,
    ec_do_purity = True,
    rot_sample_schedule = 'exp',
    trans_sample_schedule = 'linear',
    
):

    sample_feats = copy.deepcopy(init_feats)
    if sample_feats["rigids_t"].ndim == 2:
        t_placeholder = torch.ones((1,)).to(args.device)
    else:
        t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(args.device)

    forward_steps = np.linspace(min_t, max_t, num_t)
    all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
    all_aa = [du.move_to_np(copy.deepcopy(sample_feats["aatype_t"]))]
    all_bb_atom37 = [du.move_to_np(all_atom.to_atom37(ru.Rigid.from_tensor_7(sample_feats["rigids_t"].type(torch.float32)))[0])]
    if args.flow_ec:
        all_ec = [du.move_to_np(copy.deepcopy(sample_feats["ec_t"]))]
    
    t_1 = forward_steps[0]
    with torch.no_grad():
        for t_2 in forward_steps[1:]:
            if args.embed.embed_self_conditioning and self_condition:
                sample_feats["t"] = t_1 * t_placeholder
                sample_feats = self_conditioning_fn(args, main_network, sample_feats)
            
            sample_feats["t"] = t_1 * t_placeholder
            dt = t_2 - t_1
            
            model_out = main_network(sample_feats)
            aa_pred = model_out["amino_acid"]
            rot_pred = model_out["rigids_tensor"].get_rots().get_rot_mats()
            trans_pred = model_out["rigids_tensor"].get_trans()
        
            if args.embed.embed_self_conditioning:
                sample_feats["sc_ca_t"] = model_out["rigids"][..., 4:]
                sample_feats["sc_aa_t"] = model_out["amino_acid"]
        
            if args.flow_msa:
                msa_pred = model_out["msa"]
            if args.flow_ec:
                ec_pred = model_out["ec"]

        
            rots_t, trans_t, rigids_t = gen_model.reverse_euler(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                        rot=du.move_to_np(rot_pred),
                        trans=du.move_to_np(trans_pred),
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        center=center,
                        center_of_mass=None,
                        rot_sample_schedule=rot_sample_schedule,
                        trans_sample_schedule=trans_sample_schedule,
                    )
        
            if args.eval.discrete_purity and aa_do_purity:
                aa_t = gen_model.reverse_masking_euler_purity(
                    feat_t=sample_feats["aatype_t"],
                    feat=aa_pred,
                    flow_mask=None,
                    t=t_1,
                    dt=dt,
                    n_token=args.num_aa_type,
                    mask_token_idx=args.masked_aa_token_idx,
                    temp=args.eval.discrete_temp,
                    noise=args.eval.aa_noise,
                )
        
            else:
                aa_t = gen_model.reverse_masking_euler(
                    feat_t=sample_feats["aatype_t"],
                    feat=aa_pred,
                    flow_mask=None,
                    t=t_1,
                    dt=dt,
                    n_token=args.num_aa_type,
                    mask_token_idx=args.masked_aa_token_idx,
                    temp=args.eval.discrete_temp,
                    noise=args.eval.aa_noise,
                )
        
            
            if args.flow_msa:
                if args.eval.discrete_purity and msa_do_purity:
                    msa_t = gen_model.reverse_masking_euler_purity(
                        feat_t=sample_feats["msa_t"],
                        feat=msa_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.msa.num_msa_vocab,
                        mask_token_idx=args.msa.masked_msa_token_idx,
                        temp=args.eval.discrete_temp,
                        noise=args.eval.msa_noise,
                    )
            
                else:
                    msa_t = gen_model.reverse_masking_euler(
                        feat_t=sample_feats["msa_t"],
                        feat=msa_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.msa.num_msa_vocab,
                        mask_token_idx=args.msa.masked_msa_token_idx,
                        temp=args.eval.discrete_temp,
                        noise=args.eval.msa_noise,
                    )
        
            if args.flow_ec:
                if args.eval.discrete_purity and ec_do_purity:
                    ec_t = gen_model.reverse_masking_euler_purity(
                        feat_t=sample_feats["ec_t"],
                        feat=ec_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.ec.num_ec_class,
                        mask_token_idx=args.ec.masked_ec_token_idx,
                        temp=args.eval.discrete_temp,
                        noise=args.eval.ec_noise,
                    )
        
                else:
                    ec_t = gen_model.reverse_masking_euler(
                        feat_t=sample_feats["ec_t"],
                        feat=ec_pred,
                        flow_mask=None,
                        t=t_1,
                        dt=dt,
                        n_token=args.ec.num_ec_class,
                        mask_token_idx=args.ec.masked_ec_token_idx,
                        temp=args.eval.discrete_temp,
                        noise=args.eval.ec_noise,
                    )
                    
        
            sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(args.device)
            sample_feats["aatype_t"] = aa_t.long().to(args.device)
            if args.flow_msa:
                sample_feats["msa_t"] = msa_t.long().to(args.device)
            if args.flow_ec:
                sample_feats["ec_t"] = ec_t.long().to(args.device)
        
            all_aa.append(du.move_to_np(aa_t))
            all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
            if args.flow_ec:
                all_ec.append(du.move_to_np(ec_t))
        
            atom37_t = all_atom.to_atom37(rigids_t)[0]
            all_bb_atom37.append(du.move_to_np(atom37_t))
            t_1 = t_2

    t_1 = max_t
    sample_feats["t"] = t_1 * t_placeholder
    with torch.no_grad():
        model_out = main_network(sample_feats)
        aa_logits = model_out['amino_acid']
        aa_logits[..., args.masked_aa_token_idx] = -1e10
        aa_pred = aa_logits.argmax(-1)
        rigid_pred = model_out['rigids_tensor']
        atom37_pred = all_atom.to_atom37(rigid_pred)[0]
        if args.flow_ec:
            ec_logits = model_out['ec']
            ec_logits[..., args.ec.masked_ec_token_idx] = -1e10
            ec_pred = ec_logits.argmax(-1).reshape(-1, 1)

        all_aa.append(du.move_to_np(aa_pred))
        all_bb_atom37.append(du.move_to_np(atom37_pred))
        all_rigids.append(du.move_to_np(rigid_pred.to_tensor_7()))
        if args.flow_ec:
            all_ec.append(du.move_to_np(ec_pred))

    # Flip trajectory
    flip = lambda x: np.flip(np.stack(x), (0,))
    time_steps = flip(forward_steps)
    all_bb_atom37 = flip(all_bb_atom37)
    all_aa = flip(all_aa)
    all_rigids = flip(all_rigids)
    if args.flow_ec:
        all_ec = flip(all_ec)
        

    out = {
        "t": time_steps, 
        "coord_traj": all_bb_atom37,
        "aa_traj": all_aa,
        "rigid_traj": all_rigids
        }
    if args.flow_ec:
        out["ec_traj"] = all_ec
        
    return out



if __name__ == "__main__":
    args = Args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.eval.eval_dir, exist_ok=True)

    # uniform
    if args.discrete_flow_type == 'uniform':
        args.num_aa_type = 20
        args.masked_aa_token_idx = None


        if args.flow_msa:
            args.msa.num_msa_vocab = 64
            args.msa.masked_msa_token_idx = None

        if args.flow_ec:
            args.ec.num_ec_class = 6
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


    # load model
    flow_model = flowmatcher.SE3FlowMatcher(args)
    model = main_network.ProteinLigandNetwork(args)
    model = model.to(args.device)
    
    ckpt_path = args.ckpt_path
    if ckpt_path:
        print(f'loading pretrained weights for enzymeflow {ckpt_path}')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_state_dict = checkpoint["model_state_dict"]
    
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k # remove `module.`
            new_state_dict[name] = v
    
        model.load_state_dict(new_state_dict, strict=False)


    # load csv
    meta_eval_csv = pd.read_csv(args.metadata_path)
    
    parent_dir = os.path.join(args.eval.eval_dir, 'eval_T10MSA')
    os.makedirs(parent_dir, exist_ok=True)

    n_samples = len(meta_eval_csv)
    print(f'starting sampling for {n_samples} items')
    for eval_id in range(n_samples):
        eval_csv_row = meta_eval_csv.iloc[eval_id]
        pdb_name = eval_csv_row['pdb_name']
        substrate_name = eval_csv_row['substrate_name']
        generated_prot = f'{pdb_name}_{substrate_name}'
        print(f'starting for Uniprot {pdb_name}, Substrate {substrate_name}, row {eval_id}')
    
        for seed in tqdm(range(0, args.eval.n_sample)):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
            feats_0 = parse_sampling_feats(args, meta_eval_csv, gen_model=flow_model, eval_row_id=eval_id)
            feats_0 = {key: value.unsqueeze(0).to(args.device) for key, value in feats_0.items()}
            
            feats_1 = sampling_inference(
                        args,
                        init_feats = feats_0,
                        gen_model = flow_model,
                        main_network = model,
                        min_t = 0.,
                        max_t = args.max_t,
                        num_t = args.num_t,
                        self_condition = args.eval.self_condition,
                        center = True,
                        do_purity = args.eval.discrete_purity,
                        rot_sample_schedule = args.eval.rot_sample_schedule,
                        trans_sample_schedule = args.eval.trans_sample_schedule,
                    )
            
            write_pdb_traj(
                args,
                feats_0=feats_0, 
                feats_1=feats_1, 
                parent_dir=parent_dir, 
                pdb_name=pdb_name, 
                substrate_name=substrate_name, 
                sample_id=seed,
            )                        

