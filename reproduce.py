import io, os, copy, re
import argparse

import math
import random
from tqdm import tqdm

from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from gen_configs import *

from data.data import process_pdb

from ofold.np import residue_constants
from ofold.utils import rigid_utils as ru
from ofold.np.protein import Protein, to_pdb

from flowmatch import flowmatcher
from flowmatch.data import utils as du
from flowmatch.data import all_atom
from flowmatch.utils.rigid_helpers import assemble_rigid_mat, extract_trans_rots_mat

from model import main_network, genzyme, folding_network, esm3_network

from Bio.PDB import PDBParser, PDBIO

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

from esm.sdk.api import ESMProtein, ESMProteinTensor, GenerationConfig
from esm.utils.decoding import decode_structure
from esm.utils.constants import esm3 as C
from esm.utils.generation import iterative_sampling_raw

from scipy.spatial import distance_matrix

BONDS = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}

RDLogger.DisableLog('rdApp.*')


RESTYPES = [
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
    " ", "_"
]
ALPHABET=''.join([i for i in RESTYPES])
N_radius = residue_constants.van_der_waals_radius["N"]
def reindexing(args, coords, temp=1.5, n_atom=5, start_idx=32):    
    n_res = args.n_pocket_res
    atom37_pos = copy.deepcopy(coords)
    dist = [np.min(distance_matrix(atom37_pos[res_id][:n_atom], atom37_pos[res_id+1][:n_atom])) for res_id in range(n_res - 1)]

    new_idx = [start_idx]
    for idx in range(len(dist)):
        new_idx.append(new_idx[idx] + math.ceil((dist[idx] / N_radius) ** temp))

    new_idx = np.array(new_idx, dtype=int)[None,...]
    return new_idx


def decode_protein_token(structure_tokens, sequence_tokens, decoding_network):
    # per-sample input!!
    assert len(structure_tokens) == len(sequence_tokens), f"{len(structure_tokens)} != {len(sequence_tokens)}"
    # add BOS and EOS to tensors
    sequence_tokens = torch.cat(
        [torch.LongTensor([C.SEQUENCE_BOS_TOKEN]), 
        sequence_tokens.cpu(), 
        torch.LongTensor([C.SEQUENCE_EOS_TOKEN])]
    )
    structure_tokens = torch.cat(
        [torch.LongTensor([C.STRUCTURE_BOS_TOKEN]), 
        structure_tokens.cpu(), 
        torch.LongTensor([C.STRUCTURE_EOS_TOKEN])]
    )
    
    prot = ESMProteinTensor(sequence=sequence_tokens, structure=structure_tokens)
    prot = prot.to(decoding_network.device)
    raw_protein = decoding_network.decode(prot)
    
    return raw_protein


def frames_to_inversefold(frames):
    device = frames["amino_acid"].device
    batch_size, num_res, _ = frames["amino_acid"].shape

    _, atom_mask, _, atom_pos = all_atom.to_atom37(frames["rigids_tensor"])
    atom_pos = atom_pos[:, :, :4]
    atom_mask = atom_mask[:, :, 0]

    atom_pos = atom_pos.to(device)
    score = torch.zeros([batch_size, num_res]).to(device) + 100.0
    atom_mask = atom_mask.to(dtype=torch.float32).to(device)
    return atom_pos, score, atom_mask


def forward_inversefold(inversefold_model, input_feats):
    pos, score, mask = input_feats
    X, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = inversefold_model._get_features(score, X=pos, mask=mask)
    aa_log_probs, aa_logits = inversefold_model(h_V, h_E, E_idx, batch_id, return_logit=True)
    return aa_logits

def smiles_to_sdf(smiles, destination):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol)
    
    writer = Chem.SDWriter(destination)
    writer.write(mol)
    writer.close()
    
def process_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)        
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        
    atom_feat = []
    for atom in mol.GetAtoms():
        atom_feat.append(atom.GetAtomicNum())
    
    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [BONDS[bond.GetBondType()]]
    
    edge_index = [rows, cols]
    atom_feat = np.array(atom_feat)
    edge_index = np.array(edge_index)
    edge_types = np.array(edge_types)
    
    perm = (edge_index[0] * atom_feat.size + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_types = edge_types[perm]
    
    final_feats = {
        "molecule_atom_feat": atom_feat,
        "molecule_edge_idx": edge_index,
        "molecule_edge_feat": edge_types,
    }
    
    return final_feats


def process_ligand(ligand_feats, guiding_mol):
    updated_ligand_feats = {}
    ligand_atom_feat = torch.tensor(ligand_feats["molecule_atom_feat"]).long()
    ligand_edge_feat = torch.tensor(ligand_feats["molecule_edge_feat"]).long()
    ligand_edge_index = torch.tensor(ligand_feats["molecule_edge_idx"]).long()
    ligand_atom_mask = torch.ones_like(ligand_atom_feat)
    ligand_edge_mask = torch.ones_like(ligand_edge_feat)
    updated_ligand_feats["ligand_atom"] = ligand_atom_feat
    updated_ligand_feats["ligand_edge"] = ligand_edge_feat
    updated_ligand_feats["ligand_edge_index"] = ligand_edge_index
    updated_ligand_feats["ligand_atom_mask"] = ligand_atom_mask
    updated_ligand_feats["ligand_edge_mask"] = ligand_edge_mask

    # guiding_mol = processed_feats["product"]
    guiding_atom_feat = torch.tensor(guiding_mol["molecule_atom_feat"]).long()
    guiding_edge_feat = torch.tensor(guiding_mol["molecule_edge_feat"]).long()
    guiding_edge_index = torch.tensor(guiding_mol["molecule_edge_idx"]).long()
    guiding_atom_mask = torch.ones_like(guiding_atom_feat)
    guiding_edge_mask = torch.ones_like(guiding_edge_feat)
    updated_ligand_feats["guide_ligand_atom"] = guiding_atom_feat
    updated_ligand_feats["guide_ligand_edge"] = guiding_edge_feat
    updated_ligand_feats["guide_ligand_edge_index"] = guiding_edge_index
    updated_ligand_feats["guide_ligand_atom_mask"] = guiding_atom_mask
    updated_ligand_feats["guide_ligand_edge_mask"] = guiding_edge_mask
    return updated_ligand_feats


def process_protein(args, chain_feats):
    gt_bb_rigid = ru.Rigid.from_tensor_4x4(chain_feats["rigidgroups_1"])[:, 0]
    flowed_mask = np.ones(args.n_pocket_res)
    flow_mask = np.ones(args.n_pocket_res)
    chain_feats["res_mask"] = flow_mask
    chain_feats["flow_mask"] = flow_mask
    chain_feats["rigids_1"] = gt_bb_rigid.to_tensor_7()
    chain_feats["sc_ca_t"] = torch.zeros(args.n_pocket_res, 3)
    chain_feats["sc_aa_t"] = torch.zeros(args.n_pocket_res, args.num_aa_type)

    #remove unused features
    del chain_feats["residx_atom14_to_atom37"], chain_feats["atom37_pos"], chain_feats["atom37_mask"], chain_feats["atom14_pos"], chain_feats["atom37_pos_before_com"], chain_feats["torsion_angles_sin_cos"]
    return chain_feats


def gen_data(args, gen_model, protein, ligand):
    gt_bb_rigid = ru.Rigid.from_tensor_7(protein["rigids_1"])
    gt_trans, gt_rot = extract_trans_rots_mat(gt_bb_rigid)
    protein["trans_1"] = gt_trans
    protein["rot_1"] = gt_rot

    if args.n_pocket_res != protein["aatype"].size(0):
        protein["seq_idx"] = torch.arange(args.n_pocket_res) + 1
        protein["residue_idx"] = torch.arange(args.n_pocket_res) + 1
        
    aatype_1 = F.one_hot(protein["aatype"], num_classes=args.num_aa_type)

    t = 0.
    gen_feats_t = gen_model.sample_ref(
        n_samples=args.n_pocket_res,
        flow_mask=None,
        as_tensor_7=True,
        center_of_mass=None,
    )

    aatype_0 = torch.rand(args.n_pocket_res)
    aatype_t = gen_model.forward_masking(
        feat_0=aatype_0,
        feat_1=None,
        t=0.,
        mask_token_idx=args.masked_aa_token_idx,
        flow_mask=None,
    )

    protein["aatype_t"] = aatype_t
    protein.update(gen_feats_t)
    protein["t"] = t

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



def gen_data_virtual_pocket(args, gen_model, ligand):
    n_res = args.n_pocket_res

    protein = {}
    aatype_0 = torch.rand(args.n_pocket_res)
    protein["res_mask"] = torch.ones(args.n_pocket_res)
    protein["flow_mask"] = torch.ones(args.n_pocket_res)
    protein["seq_idx"] = torch.arange(args.n_pocket_res) + 1
    protein["residue_index"] = torch.arange(args.n_pocket_res) + 1
        
    t = 0.
    gen_feats_t = gen_model.sample_ref(
        n_samples=args.n_pocket_res,
        flow_mask=None,
        as_tensor_7=True,
        center_of_mass=None,
    )

    protein["sc_aa_t"] = torch.zeros(args.n_pocket_res, args.num_aa_type)
    protein["sc_ca_t"] = torch.zeros_like(torch.tensor(gen_feats_t["trans_t"]))

    aatype_t = gen_model.forward_masking(
        feat_0=aatype_0,
        feat_1=None,
        t=0.,
        mask_token_idx=args.masked_aa_token_idx,
        flow_mask=None,
    )

    protein["aatype_t"] = aatype_t
    protein.update(gen_feats_t)
    protein["t"] = t

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


def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype=None,
    b_factors=None,
    residue_index=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    if residue_index is None:
        residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    aatype: np.ndarray = None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
    residue_index=None
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = file_path
    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            for t, pos14 in enumerate(prot_pos):
                atom14_mask = np.sum(np.abs(pos14), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos14, atom14_mask, aatype=aatype, b_factors=b_factors, residue_index=residue_index
                )
                pdb_prot = protein.to_pdb(prot, model=t+1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom14_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom14_mask, aatype=aatype, b_factors=b_factors, residue_index=residue_index
            )
            pdb_prot = to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path


def write_pdb_traj(args, feats_0, feats_1, parent_dir, pdb_name, substrate_name, sample_id=0):
    final_prot = {
                "t_1": feats_1["t"][0],
                "pos_1": feats_1["coord_traj"][0],
                "aa_1": feats_1["aa_traj"][0],
            }
    
    CA_IDX = residue_constants.atom_order["CA"]
    res_mask = du.move_to_np(feats_0["res_mask"].bool())
    flow_mask = du.move_to_np(feats_0["flow_mask"].bool())
    res_index = du.move_to_np(feats_0["residue_index"])
    
    batch_size = res_mask.shape[0]
    
    for i in range(batch_size):
        num_res = int(np.sum(res_mask[i]).item())
        unpad_flow_mask = flow_mask[i][res_mask[i]]
        unpad_protein = {
            "pos": final_prot['pos_1'][i][res_mask[i]],
            "aatype": final_prot['aa_1'][i][res_mask[i]],
        }
        
        pred_aatype = unpad_protein["aatype"]
        pred_portein_pos = unpad_protein["pos"]
            
        generated_dir = parent_dir
        generated_prot = pdb_name
        prot_dir = os.path.join(generated_dir, generated_prot)
        if not os.path.isdir(prot_dir):
            os.makedirs(prot_dir, exist_ok=True)
                    
        prot_path = os.path.join(prot_dir, f"pocket_{sample_id}.pdb")
    
        saved_path = write_prot_to_pdb(
                        prot_pos=pred_portein_pos,
                        file_path=prot_path,
                        aatype=pred_aatype,
                        no_indexing=True,
                        b_factors=np.tile(unpad_flow_mask[..., None], 37) * 100,
                        residue_index=res_index[i],
                    )
        
def tokenize_pocket(args, inpainting_model, prot_feats, pocket_feats, max_len=512):
    filled_ids = (prot_feats['residue_index']-1)
    protein_ids = torch.arange(max_len).long()

    pocket_mask = torch.zeros(max_len)
    pocket_mask[filled_ids] = True
    pocket_mask = pocket_mask.bool().to(args.device)

    pocket_coords = torch.tensor(pocket_feats['coord_traj'][0]).to(args.device)
    _, _structure_token = inpainting_model.net._structure_encoder.encode(
            pocket_coords, residue_index=prot_feats['residue_index']
        )

    structure_token = torch.ones(1, max_len) * C.STRUCTURE_MASK_TOKEN
    structure_token = structure_token.to(args.device)
    structure_token[:, pocket_mask] = _structure_token[0].float()
    structure_token = structure_token.long()

    sequence_token = torch.ones(1, max_len) * C.SEQUENCE_MASK_TOKEN
    sequence_token = sequence_token.long().to(args.device)
    return structure_token, sequence_token
        
    
def fill_pocket(args, prot_feats, pocket_feats, max_len=512):
    filled_ids = (prot_feats['residue_index'] - 1)
    protein_ids = torch.arange(max_len).long()

    pocket_mask = torch.zeros(max_len)
    pocket_mask[filled_ids] = True
    pocket_mask = pocket_mask.bool().to(args.device)

    protcoord = torch.ones(max_len, 37, 3) * float('inf')
    protcoord = protcoord.to(args.device)
    pocket_coords = torch.tensor(pocket_feats['coord_traj'][0]).to(args.device)
    protcoord[pocket_mask] = pocket_coords[0]

    
    protseq = torch.ones(max_len) * 21
    protseq = protseq.long().to(args.device)
    pocket_seqs = torch.tensor(pocket_feats['aa_traj'][0]).to(args.device)
    protseq[pocket_mask] = pocket_seqs[0]
    protseq = ''.join([ALPHABET[i] for i in protseq])
    
    # protseq = '_' * max_len
    # protseq = ''.join(protseq)

    return protcoord, protseq


def inverse_folding(prot, pocket_coord, n_invfold_sample=8):
    prot_chain = prot.to_protein_chain()

    prot_list = [prot for _ in range(n_invfold_sample)]
    cfg_seq_list  = [GenerationConfig(track="sequence", num_steps=args.num_inpaint_t, temperature=0.5) for _ in range(n_invfold_sample)]
    full_prot_list = iterative_sampling_raw(
                        inpainting_model.net, proteins=prot_list, configs=cfg_seq_list,
                    )
        
    full_prot_list = [ESMProtein(sequence=full_prot.sequence, coordinates=pocket_coord) for full_prot in full_prot_list]
    cfg_struct_list = [GenerationConfig(track="structure", num_steps=args.num_inpaint_t//2, temperature=0.5) for _ in range(len(prot_list))]
    full_prot_list = iterative_sampling_raw(
                        inpainting_model.net, proteins=full_prot_list, configs=cfg_struct_list,
                    )

    full_prot_chain_list = [full_prot.to_protein_chain().align(
                                prot_chain,
                                only_use_backbone=True,
                            ) for full_prot in full_prot_list]

    rmsd_list = [full_prot_chain.rmsd(
                    prot_chain,
                ) for full_prot_chain in full_prot_chain_list]

    prot_seq = full_prot_chain_list[np.argmin(rmsd_list)].sequence
    return prot_seq

        
def sampling_inference(
    args,
    init_feats,
    gen_model,
    main_network,
    invfold_network,
    min_t = 0.,
    max_t = 1.,
    num_t = 100,
    center = True,
    self_condition = False,
    aa_do_purity = True,
    aa_temp = 0.1,
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
        
                                
        
            sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(args.device)
            sample_feats["aatype_t"] = aa_t.long().to(args.device)
        
            all_aa.append(du.move_to_np(aa_t))
            all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
        
            atom37_t = all_atom.to_atom37(rigids_t)[0]
            all_bb_atom37.append(du.move_to_np(atom37_t))
            t_1 = t_2

    t_1 = max_t
    sample_feats["t"] = t_1 * t_placeholder
    with torch.no_grad():
        model_out = main_network(sample_feats)
        rigid_pred = model_out['rigids_tensor']
        atom37_pred = all_atom.to_atom37(rigid_pred)[0]
        
        # aa_logits = model_out['amino_acid']
        # aa_logits[..., args.masked_aa_token_idx] = -1e10
        # aa_pred = aa_logits.argmax(-1)

        inversefold_feats = frames_to_inversefold(model_out)
        aa_logits = forward_inversefold(invfold_network, inversefold_feats)
        aa_probs = F.softmax(aa_logits/aa_temp, dim=-1)
        aa_pred = torch.multinomial(aa_probs, 1).reshape(aa_t.shape)
        
        all_aa.append(du.move_to_np(aa_pred))
        all_bb_atom37.append(du.move_to_np(atom37_pred))
        all_rigids.append(du.move_to_np(rigid_pred.to_tensor_7()))

    # Flip trajectory
    flip = lambda x: np.flip(np.stack(x), (0,))
    time_steps = flip(forward_steps)
    all_bb_atom37 = flip(all_bb_atom37)
    all_aa = flip(all_aa)
    all_rigids = flip(all_rigids)
        

    out = {
        "t": time_steps, 
        "coord_traj": all_bb_atom37,
        "aa_traj": all_aa,
        "rigid_traj": all_rigids
        }
        
    return out


args = Args()
#args.device = torch.device('cpu')
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.discrete_flow_type == 'uniform':
    args.num_aa_type = 20
    args.masked_aa_token_idx = None


# discrete
elif args.discrete_flow_type == 'masking':
    args.num_aa_type = 21
    args.masked_aa_token_idx = 20
    args.aa_ot = False


else:
    raise ValueError(f'Unknown discrete flow type {args.discrete_flow_type}')


# Loading Model
flow_model = flowmatcher.SE3FlowMatcher(args)
gen_model = main_network.ProteinLigandNetwork(args)
inversefold_model = folding_network.ProDesign_Model(args.inverse_folding)

esm_model = esm3_network.CustomizedESM3(args.inpainting)
vqvae_model = genzyme.initialize_structure_encoder(args, pretrained_structure_encoder=esm_model.get_structure_encoder())
inpainting_model = genzyme.initialize_inpainting_module(args, esm_model, vqvae_model=vqvae_model)

GENzyme = genzyme.GENzyme(args, gen_model, inversefold_model, inpainting_model=inpainting_model)
GENzyme = GENzyme.float()

ckpt_path = 'genzyme_ckpt/genzyme.ckpt'
if ckpt_path:
    print(f'loading pretrained weights for GENzyme {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model_state_dict = checkpoint["model_state_dict"]
    GENzyme.load_state_dict(model_state_dict, strict=False)

GENzyme = GENzyme.to(args.device)

    
if __name__ == "__main__":
    protein_ids = [
        "Q8N4T8", "B2SUY7", "Q9WYD4", "Q4V8E6", "A6UP94",
        "Q8DBS9", "O84413", "B2HDR8", "F1RT67", "P61452",
        "A5N2N5", "C6DGZ5", "B7UVB1", "Q6DDT1", "P9WIQ1",
        "B3WC77", "Q8E0G6", "A9CB42", "Q86W10", "Q2TZB2",
        "A1JJD0", "C4JE77", "B3LNM2", "P0AA00", "Q9RVF5",
        "A7MKL4", "Q6CFH4", "Q23381", "Q6G456", "Q0SM64"
    ]

    substrate_ids = [
        "CC(=O)CC(=O)CC(=O)OC",
        "CC[C@@H](CCCCCC/C=C/C(=O)C)C",
        "OC[C@H]([C@H]([C@@H](C(=O)CO)O)O)O",
        "SC[C@@H](C(=O)NCC(=O)O)NC(=O)CC[C@@H](C(=O)O)N",
        "CC(=O)c1ccccc1",
        "OC[C@H]1O[C@@H](C[C@@H]1O)OP(=O)(O)O",
        "CN[C@H](C(=O)C)CSC[C@H](OC(=O)C)COC(=O)C",
        "O=CN(c1ccc(cc1)C(=O)N[C@H](C(=O)O)CCC(=O)C)C[C@H]1CNc2c(N1)c(=O)[nH]c(n2)N",
        "O[C@@H]1[C@H](OP(=O)(O)O)[C@H](O[C@H]1n1cnc2c1ncnc2N)COP(=O)(O)O",
        "O[C@H](C(C)C)CC(=O)C",
        "OC(=O)C(=C)OP(=O)(O)O",
        "O[C@@H]([C@H](c1c[nH]c2c1cccc2)O)COP(=O)(O)O",
        "CCCCCCCCCCCCCCCCC[C@H](CC(=O)C)O",
        "OC1O[C@H](COP(=O)(O)O)[C@H]([C@@H]([C@H]1O)O)O",
        "OC[C@H]1O[C@H](OP(=O)(OP(=O)(OC[C@H]2O[C@H]([C@@H]([C@@H]2O)O)n2ccc(=O)[nH]c2=O)O)O)[C@@H]([C@H]([C@H]1O)O)O",
        "C[C@H](C(=O)O)N",
        "C[C@H](C(=O)O)N",
        "COP(=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1OP(=O)(O)O)O)C)O",
        "CN1c2cc(C)c(cc2Nc2c1[nH]c(=O)[nH]c2=O)C",
        "N[C@H](C(=O)O)CCCN",
        "OC[C@@]1(O)O[C@@H]([C@H]([C@@H]1O)O)COP(=O)(O)O",
        "C[S@+](C[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N)CC[C@@H](C(=O)O)N",
        "CC(=O)[C@@H](NC(=O)[C@H](CCSC)N)C",
        "N[C@H](c1ccccc1)C(=O)N[C@@H]1C(=O)N2[C@@H]1SC([C@@H]2C(=O)O)(C)C",
        "COC(=O)CCC[C@H](CC(=O)C)O",
        "O[C@@H]([C@H]([C@@H](CC(=O)C(=O)O)O)O)COP(=O)(O)O",
        "OCC(=O)[C@@H]([C@@H](COP(=O)(O)O)O)O",
        "O=C(NCCSC(=O)[C@@H](C(=O)O)C)CCNC(=O)[C@@H](C(COP(=O)(OP(=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1OP(=O)(O)O)O)n1cnc2c1ncnc2N)O)O)(C)C)O",
        "OCC([C@H](C(=O)O)O)(C)C",
        "COP(=O)(O[C@@H]1[C@@H](COP(=O)(OC)O)O[C@H]([C@@H]1O)n1ccc(nc1=O)N)O"
    ]

    product_ids = [
        "CC(=O)C[C@@H](CC(=O)OC)O",
        "CC[C@@H](CCCCCCCCC(=O)C)C",
        "OC[C@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O",
        "CSC[C@@H](C(=O)NCC(=O)O)NC(=O)CC[C@@H](C(=O)O)N",
        "C[C@H](c1ccccc1)N",
        "OC[C@H]1O[C@H](C[C@@H]1O)n1cnc2c1ncnc2N",
        "CC(=O)O[C@H](COC(=O)C)CSC[C@@H](C(=O)C)N",
        "CC(=O)CC[C@@H](C(=O)O)NC(=O)c1ccc(cc1)N1C=[N+]2[C@@H](C1)CNc1c2c(=O)[nH]c(n1)N",
        "O[C@@H]1[C@H](O)[C@H](O[C@H]1n1cnc2c1ncnc2N)COP(=O)(O)O",
        "CC(/C=C/C(=O)C)C",
        "OC[C@H](C(=O)O)OP(=O)(O)O",
        "OC(=O)[C@H](Cc1c[nH]c2c1cccc2)N",
        "CCCCCCCCCCCCCCCCC/C=C/C(=O)C",
        "O[C@@H]1[C@H](OP(=O)(O)O)[C@@H](O)[C@@H]([C@H]([C@@H]1O)O)O",
        "OC[C@H]([C@@H]1O[C@@H]([C@@H]([C@H]1O)O)OP(=O)(OP(=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1ccc(=O)[nH]c1=O)O)O)O",
        "CC(=O)[C@H](N)C",
        "C[C@H](C(=O)O)NC(=O)[C@H](N)C",
        "COP(=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1OP(=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1OP(=O)(OC)O)O)C)O)O)C)O",
        "Cc1cc2nc3-c(n(c2cc1C)C)nc(=O)[nH]c3=O",
        "ONCCC[C@@H](C(=O)O)N",
        "O=C[C@@H](COP(=O)(O)O)O",
        "C[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N",
        "CSCC[C@@H](C(=O)O)N",
        "N[C@H](c1ccccc1)C(=O)N[C@H](C1N[C@H](C(S1)(C)C)C(=O)O)C(=O)O",
        "COC(=O)CCC/C=C/C(=O)C",
        "OP(=O)(O)O",
        "O=C[C@@H]([C@@H]([C@@H](COP(=O)(O)O)O)O)O",
        "O=C(NCCSC(=O)CCC(=O)O)CCNC(=O)[C@@H](C(COP(=O)(OP(=O)(OC[C@H]1O[C@H]([C@@H]([C@@H]1OP(=O)(O)O)O)n1cnc2c1ncnc2N)O)O)(C)C)O",
        "OCC([C@H](C(=O)NCCC(=O)O)O)(C)C",
        "COP(=O)(O[C@@H]1[C@@H](COP(=O)(OC)O)O[C@H]([C@@H]1O)n1ccc(=N)nc1NCCCC[C@@H](C(=O)O)N)O"
    ]

    for s_idx in range(len(protein_ids)):
        pdb_name = protein_ids[s_idx]
        substrate_smiles = substrate_ids[s_idx]
        product_smiles = product_ids[s_idx]
    
        # write down reaction molecules
        smiles_to_sdf(substrate_smiles, destination=f'data/ligand/{pdb_name}_substrate.sdf')
        smiles_to_sdf(product_smiles, destination=f'data/ligand/{pdb_name}_product.sdf')

        print('starting processing data...')
        # process reaction
        substrate = process_smiles(substrate_smiles)
        product = process_smiles(product_smiles)
        substrate_mol, product_mol = copy.copy(substrate), copy.copy(product)
        reaction_feats = process_ligand(substrate_mol, product_mol)

        # process protein
        pocket_path = None
        protein = None
        try:
            pocket_path = f'data/ground_truth/pocket/{pdb_name}.pdb'
            print(f'Loading catalytic pocket for {pdb_name}')
            protein = process_pdb(pocket_path)
            chain_feats = copy.copy(protein)
            protein = process_protein(args, chain_feats)

        except:
            print(f'Unknown pocket file for {pdb_name}, using virtual residue index...')


        # load AFDB structure
        n_prot_res = args.n_protein_res
        try:
            full_protein_path = f'data/ground_truth/protein/{pdb_name}.pdb'
            print(f'Loading enzyme for {pdb_name}')
            full_prot = ESMProtein.from_pdb(full_protein_path)
            n_prot_res = len(full_prot.sequence)

        except:
            print(f'Unknown AFDB file for {pdb_name}, generating enzyme with {n_prot_res} residues...')


        if protein is None:
            pdb_name = f'{substrate_smiles}_{product_smiles}'


        args.eval.aa_temp = 0.1
        args.eval.aa_noise = 20.
        parent_dir = os.path.join('fuck')
        os.makedirs(parent_dir, exist_ok=True)

        print('Starting sampling enzyme...')
        n_sample = 1000000
        generated_sample = 0
        for _ in tqdm(range(n_sample)):
            sample_idx = generated_sample
            seed = random.randint(0, 1000000)
            #seed = sample_idx
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            if protein:
                feats_0 = gen_data(args, flow_model, protein, reaction_feats)

            else:
                feats_0 = gen_data_virtual_pocket(args, flow_model, reaction_feats)

            feats_0 = {
                    k: v.unsqueeze(0).to(args.device) if torch.is_tensor(v) else v for k, v in feats_0.items()
                }

            feats_1 = sampling_inference(
                            args,
                            init_feats = feats_0,
                            gen_model = flow_model,
                            main_network = GENzyme.generation_model,
                            invfold_network = GENzyme.inversefold_model,
                            min_t = args.min_t,
                            max_t = args.max_t,
                            num_t = args.num_pocket_design_t,
                            self_condition = False,
                            center = True,
                            aa_do_purity = True,
                            aa_temp = 0.1,
                            rot_sample_schedule = 'exp',
                            trans_sample_schedule = 'linear',
                        )


            # reindexing pocket residue index
            if protein is None:
                coords = feats_1["coord_traj"][0][0]
                residue_index = reindexing(args, coords, start_idx=n_prot_res//3)

                if max(residue_index[0]) >= n_prot_res:
                    residue_index = reindexing(args, coords, start_idx=n_prot_res//4)

                    if max(residue_index[0]) >= n_prot_res:
                        residue_index = reindexing(args, coords, temp=1.0, start_idx=1)

                        if max(residue_index[0]) >= n_prot_res:
                            continue

                feats_0['residue_index'] = residue_index


            if args.inpaint_pocket:
                prot_path = os.path.join(parent_dir, pdb_name, f'protein_{sample_idx}.pdb')

                if args.inpaint_method == 'ddpm':
                    ## DDPM sampling
                    structure_token, sequence_token = tokenize_pocket(
                                                        args, 
                                                        GENzyme.inpainting_model, 
                                                        prot_feats=feats_0, 
                                                        pocket_feats=feats_1, 
                                                        max_len=n_prot_res
                                                    )

                    structure_token_traj = GENzyme.inpainting_model.ddpm_sample_trajectory(
                                                        num_steps=args.num_inpaint_t, 
                                                        sequence_tokens=sequence_token, 
                                                        input_prior=structure_token, 
                                                        sample_max_t=args.max_inpaint_t,
                                                    )

                    structure_token = structure_token_traj[-1]
                    prot = decode_protein_token(
                                    structure_tokens=structure_token[0], 
                                    sequence_tokens=sequence_token[0], 
                                    decoding_network=GENzyme.inpainting_model.net,
                                )

                    prot.to_pdb(prot_path)


                elif args.inpaint_method == 'gibbs':
                    ## Gibbs Sampling
                    protcoord, protseq = fill_pocket(
                                            args, 
                                            prot_feats=feats_0, 
                                            pocket_feats=feats_1, 
                                            max_len=n_prot_res
                                        )


                    for _ in range(8):
                        prot_list = [ESMProtein(sequence=protseq, coordinates=protcoord)]
                        cfg_struct_list = [GenerationConfig(track="structure", num_steps=args.num_inpaint_t, temperature=0.5)]
                        prot = iterative_sampling_raw(
                                    GENzyme.inpainting_model.net, proteins=prot_list, configs=cfg_struct_list,
                                )[0]

                        ptm = prot.ptm.mean().item()
                        plddt = prot.plddt
                        protein_plddt = prot.plddt.mean().item()

                        print(f'pTM:{ptm}, protein-pLDDT:{protein_plddt}, seed:{seed}, pdb: {pdb_name}')
                        if (ptm > args.ptm_filter) or (protein_plddt > args.plddt_filter):
                            print(f'pTM:{ptm}, protein-pLDDT:{protein_plddt}, seed:{seed}, sample_idx: {sample_idx}')
                            write_pdb_traj(
                                args,
                                feats_0=feats_0, 
                                feats_1=feats_1, 
                                parent_dir=parent_dir, 
                                pdb_name=pdb_name, 
                                substrate_name=None, 
                                sample_id=sample_idx,
                            )

                            prot_seq = inverse_folding(copy.deepcopy(prot), pocket_coord=protcoord)
                            prot.sequence = prot_seq
                            prot.to_pdb(prot_path)

                            generated_sample += 1
                            break


                if generated_sample == args.n_sample_enzyme:
                    print(f'Done sampling {args.n_sample_enzyme} enzymes...')
                    break







