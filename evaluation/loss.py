import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ofold.np import residue_constants
from ofold.utils import rigid_utils as ru
from ofold.utils.tensor_utils import permute_final_dims

from flowmatch.data import all_atom
from flowmatch.utils.so3_helpers import hat_inv, pt_to_identity
from flowmatch.utils.rigid_helpers import extract_trans_rots_mat

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss
    
def multiclass_posterior(aa_t, aa_vectorfield, t):
    if aa_t.dim() == 2:
        theta = aa_t + aa_vectorfield * (1-t[..., None])
        
    elif aa_t.dim() == 3:
        theta = aa_t + aa_vectorfield * (1-t[..., None, None])   #(N, L, K)
        
    elif aa_t.dim() == 4:
        theta = aa_t + aa_vectorfield * (1-t[..., None, None, None])
        
    theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-10)
    return theta


def intersection_loss(protein_atom, protein_pos, ligand_pos, ligand_mask, rho=2., gamma=6.):
    n_batch, n_res, n_atom = protein_atom.shape
    aa_pos = protein_pos.reshape(n_batch, n_res*n_atom, 3).unsqueeze(2)
    ligand_pos = ligand_pos.unsqueeze(1)
    dist_mask = protein_atom.reshape(
                        n_batch, n_res*n_atom, 1
                ) * ligand_mask.reshape(n_batch, 1, -1)
    dist2 = (torch.square(aa_pos - ligand_pos).sum(dim=-1) + 1e-10)
    exp_dist2 = torch.divide(-dist2, rho).exp() * dist_mask
    loss_per_aa = -rho * exp_dist2.sum(dim=-1).clamp(min=1e-20).log()
    loss_per_aa = gamma - loss_per_aa
    interior_loss = torch.clamp(loss_per_aa, min=1e-10)
    return interior_loss.reshape(n_batch, n_res, n_atom)


def loss_fape(
    pred_frames,
    target_frames,
    frames_mask,
    pred_positions,
    target_positions,
    positions_mask,
    length_scale,
    pair_mask=None,
    l1_clamp_distance=None,
    eps=1e-10,
) -> torch.Tensor:
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = (
            normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        )
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def compute_plddt(logits):
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos,
    all_atom_positions,
    all_atom_mask,
    cutoff=10.0,
    eps=1e-10,
    per_residue=True,
):
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def loss_lddt(
    logits,
    all_atom_pred_pos,
    all_atom_positions,
    all_atom_mask,
    resolution=1.0,
    cutoff=10.0,
    no_bins=22,
    min_resolution=0.1,
    max_resolution=3.0,
    eps=1e-10,
    **kwargs,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]

    ca_pos = 1 # use 14-atom frame
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos, :]  # keep dim

    score = lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps
    )

    # TODO: Remove after initial pipeline testing
    score = torch.nan_to_num(score, nan=torch.nanmean(score))
    score[score < 0] = 0

    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )


    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    return loss


def loss_tm(
    logits,
    pred_affine,
    backbone_rigid,
    backbone_rigid_mask,
    resolution=1.0,
    max_bin=31,
    no_bins=64,
    min_resolution=0.1,
    max_resolution=3.0,
    eps=1e-10,
    **kwargs,
):
    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum(
        (_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1
    )

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )
    boundaries = boundaries ** 2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, no_bins)
    )

    square_mask = (
        backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]
    )

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    return loss


def between_residue_bond_loss(
    pred_atom_positions,  # (*, N, 37/14, 3)
    pred_atom_mask,  # (*, N, 37/14)
    residue_index,  # (*, N)
    aatype,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-10,
):
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = aatype[..., 1:] == residue_constants.resname_to_idx["PRO"]
    gt_length = (
                    ~next_is_proline
                ) * residue_constants.between_res_bond_length_c_n[
                    0
                ] + next_is_proline * residue_constants.between_res_bond_length_c_n[
                    1
                ]
    gt_stddev = (
                    ~next_is_proline
                ) * residue_constants.between_res_bond_length_stddev_c_n[
                    0
                ] + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[
                    1
                ]

    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom14_pred_positions,
    atom14_atom_exists,
    atom14_atom_radius,
    residue_index,
    asym_id=None,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
):
    fp_type = atom14_pred_positions.dtype
    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residue_index[..., :, None] + 1) == residue_index[..., None, :]

    if asym_id is not None:
        neighbour_mask = neighbour_mask & (asym_id[..., :, None] == asym_id[..., None, :])

    neighbour_mask = neighbour_mask[..., None, None]

    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = residue_constants.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, dim=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    per_atom_num_clash = torch.sum(clash_mask, dim=(-4, -2)) + torch.sum(clash_mask, dim=(-3, -1))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, dim=(-4, -2)),
        torch.amax(clash_mask, dim=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
        "per_atom_num_clash": per_atom_num_clash  # shape (N, 14)
    }


def within_residue_violations(
    atom14_pred_positions,
    atom14_atom_exists,
    atom14_dists_lower_bound,
    atom14_dists_upper_bound,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
):
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    per_atom_num_clash = torch.sum(violations, dim=-2) + torch.sum(violations, dim=-1)

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
        "per_atom_num_clash": per_atom_num_clash
    }
    

def find_structural_violations(
    batch,
    atom_mask,
    atom14_pred_positions,
    violation_tolerance_factor=12.0,
    clash_overlap_tolerance=1.5,
    **kwargs,
):
    """Computes several checks for structural violations."""
    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=atom_mask,
        residue_index=batch["seq_idx"],
        aatype=batch["aatype"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]

    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)

    # TODO: Consolidate monomer/multimer modes
    asym_id = None
    if asym_id is not None:
        residx_atom14_to_atom37 = get_rc_tensor(
            residue_constants.RESTYPE_ATOM14_TO_ATOM37, batch["aatype"]
        )
        atom14_atom_radius = (
            atom_mask
            * atomtype_radius[residx_atom14_to_atom37.long()]
        )
    else:
        atom14_atom_radius = (
            atom_mask
            * atomtype_radius[batch["residx_atom14_to_atom37"]]
        )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom_mask,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["seq_idx"],
        asym_id=asym_id,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = atom_mask
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["aatype"]]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom_mask,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(
                    between_residue_clashes["per_atom_clash_mask"], dim=-1
                )[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations[
                "ca_c_n_loss_mean"
            ],  # ()
            "angles_c_n_ca_loss_mean": connection_violations[
                "c_n_ca_loss_mean"
            ],  # ()
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (N, 14)
            "clashes_per_atom_num_clash": between_residue_clashes[
                "per_atom_num_clash"
            ],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (N, 14),
            "per_atom_num_clash": residue_violations[
                "per_atom_num_clash"
            ],  # (N, 14)
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }



def loss_violation(
    violations,
    atom14_atom_exists,
    average_clashes: bool = False,
    eps=1e-10,
    **kwargs,
):
    num_atoms = torch.sum(atom14_atom_exists)

    per_atom_clash = (violations["between_residues"]["clashes_per_atom_loss_sum"] +
                      violations["within_residues"]["per_atom_loss_sum"])

    if average_clashes:
        num_clash = (violations["between_residues"]["clashes_per_atom_num_clash"] +
                     violations["within_residues"]["per_atom_num_clash"])
        per_atom_clash = per_atom_clash / (num_clash + eps)

    l_clash = torch.sum(per_atom_clash) / (eps + num_atoms)
    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )
    
    return loss
    

def compute_dist(protein_atom, protein_pos, ligand_pos, ligand_mask):
    n_batch, n_res, n_atom = protein_atom.shape
    aa_pos = protein_pos.reshape(n_batch, n_res*n_atom, 3).unsqueeze(2)
    ligand_pos = ligand_pos.unsqueeze(1)
    dist_mask = protein_atom.reshape(
                        n_batch, n_res*n_atom, 1
                ) * ligand_mask.reshape(n_batch, 1, -1)
    dist2 = torch.square(aa_pos - ligand_pos).sum(dim=-1)
    dist = torch.sqrt(
                dist2 + 1e-10
            ) * dist_mask
    return dist, dist_mask


def loss_inversefold(args, batch, pred_aa_logits):
    device = pred_aa_logits.device
    bb_mask = batch["res_mask"].to(device)
    batch_size, num_res = bb_mask.shape
    batch_loss_mask = torch.any(bb_mask, dim=-1)
    
    gt_aa = batch["aatype"]
    aa_loss = F.cross_entropy(
                            input=pred_aa_logits.reshape(-1, 20), 
                            target=gt_aa.flatten().long(), 
                            reduction="none"
                ).reshape(batch_size, num_res)
    aa_loss = (aa_loss * bb_mask).sum(dim=-1) / (bb_mask.sum(dim=-1) + 1e-10)

    def normalize_loss(x):
        return x.sum() / (batch_loss_mask.sum() + 1e-10)

    return normalize_loss(aa_loss)

    
def loss_fn(args, batch, model_out, flow_matcher):
    """Computes loss and auxiliary data.
    
    Args:
        batch: Batched data.
        model_out: Output of model ran on batch.
    
    Returns:
        loss: Final training loss scalar.
        aux_data: Additional logging data.
    """    
    device = model_out["amino_acid"].device
    bb_mask = batch["res_mask"].to(device)
    flow_mask = batch["flow_mask"].to(device)
    lig_mask = batch["ligand_atom_mask"].to(device)
    loss_mask = bb_mask * flow_mask
    batch_size, num_res = bb_mask.shape
    
    rot_vectorfield_scaling = batch["rot_vectorfield_scaling"]
    trans_vectorfield_scaling = batch["trans_vectorfield_scaling"]
    batch_loss_mask = torch.any(bb_mask, dim=-1)

    pred_rot = model_out["rigids_tensor"].get_rots().get_rot_mats()
    pred_trans = model_out["rigids_tensor"].get_trans()
    
    gt_rot_u_t = flow_matcher._so3_fm.vectorfield(
        batch["rot_t"], batch["rot_1"], batch["t"],
    )

    pred_rots_u_t = flow_matcher._so3_fm.vectorfield(
        batch["rot_t"], pred_rot, batch["t"]
    )
    
    rot_mse = (gt_rot_u_t - pred_rots_u_t) ** 2 * loss_mask[..., None]
    rot_loss = torch.sum(
        rot_mse / rot_vectorfield_scaling[:, None, None] ** 2,
        dim=(-1, -2),
    ) / (loss_mask.sum(dim=-1) + 1e-10)
    rot_loss *= args.exp.rot_loss_weight
    rot_loss *= int(args.flow_rot)


    gt_trans_x1 = batch["trans_1"] * args.exp.coordinate_scaling
    pred_trans_x1 = pred_trans * args.exp.coordinate_scaling

    trans_loss = torch.sum(
        (gt_trans_x1 - pred_trans_x1) ** 2 * loss_mask[..., None], dim=(-1, -2)
    ) / (loss_mask.sum(dim=-1) + 1e-10)
    
    trans_loss *= args.exp.trans_loss_weight
    trans_loss *= int(args.flow_trans)
    
    
    # Backbone atom loss
    _, _, _, pred_atom_pos = all_atom.to_atom37(model_out["rigids_tensor"])
    gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_1"].type(torch.float32))
    _, atom_mask, _, gt_atom_pos = all_atom.to_atom37(gt_rigids)
    
    violation_loss = torch.zeros([batch_size]).to(device)
    if args.use_struct_violation:
        violations = find_structural_violations(
                     batch,
                     atom_mask[:, :, :14].to(device),
                     pred_atom_pos,
                    )
        violation_loss = loss_violation(violations, atom_mask[:, :, :14].to(device))
        violation_loss *= batch["t"] > args.exp.aux_loss_t_filter
        violation_loss *= args.exp.violation_loss_weight
        
    n_res_atom = 4
    gt_atom_pos = gt_atom_pos[:, :, :n_res_atom]
    atom_mask = atom_mask[:, :, :n_res_atom]
    pred_atom_pos = pred_atom_pos[:, :, :n_res_atom]
    
    gt_atom_pos = gt_atom_pos.to(device)
    atom_mask = atom_mask.to(device)
    bb_atom_loss_mask = atom_mask * loss_mask[..., None]
    bb_atom_loss = torch.sum(
        (pred_atom_pos - gt_atom_pos) ** 2 * bb_atom_loss_mask[..., None],
        dim=(-1, -2, -3),
    ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
    bb_atom_loss *= args.exp.bb_atom_loss_weight
    bb_atom_loss *= batch["t"] > args.exp.aux_loss_t_filter
    bb_atom_loss *= args.exp.aux_loss_weight

    
    # Pairwise distance loss
    gt_flat_atoms = gt_atom_pos.reshape([batch_size, num_res * n_res_atom, 3])
    gt_pair_dists = torch.linalg.norm(
        gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
    )
    pred_flat_atoms = pred_atom_pos.reshape([batch_size, num_res * n_res_atom, 3])
    pred_pair_dists = torch.linalg.norm(
        pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
    )
    
    flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_res_atom))
    flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * n_res_atom])
    flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, n_res_atom))
    flat_res_mask = flat_res_mask.reshape([batch_size, num_res * n_res_atom])
    
    gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
    pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
    pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]
    
    # No loss on anything >10A
    proximity_mask = gt_pair_dists < args.exp.dist_loss_filter
    pair_dist_mask = pair_dist_mask * proximity_mask
    
    dist_mat_loss = torch.sum(
        (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
    )
    dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1e-10) - num_res
    dist_mat_loss *= args.exp.dist_mat_loss_weight
    dist_mat_loss *= batch["t"] > args.exp.aux_loss_t_filter
    dist_mat_loss *= args.exp.aux_loss_weight

    # Amino acid loss
    aa_loss = torch.zeros([batch_size]).to(device)
    if args.use_aa:
        gt_aa = batch["aatype"]
        pred_aa = model_out["amino_acid"]
        aa_loss = F.cross_entropy(
                                input=pred_aa.reshape(-1, args.num_aa_type), 
                                target=gt_aa.flatten().long(), 
                                reduction="none"
                    ).reshape(batch_size, num_res)

        aa_loss = (aa_loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-10)
        aa_loss *= args.exp.aa_loss_weight


    # FAPE loss
    fape_loss = torch.zeros([batch_size]).to(device)
    if args.use_fape:
        fape_loss = loss_fape(
                    model_out["rigids_tensor"], 
                    gt_rigids, 
                    bb_mask, 
                    pred_flat_atoms, 
                    gt_flat_atoms, 
                    flat_loss_mask, 
                    length_scale=args.exp.dist_loss_filter,
                )
        fape_loss *= batch["t"] > args.exp.aux_loss_t_filter
        fape_loss *= args.exp.fape_loss_weight


    
    plddt_loss = torch.zeros([batch_size]).to(device)
    if args.use_plddt:
        plddt_loss = loss_lddt(
                    model_out["plddt"], 
                    pred_atom_pos, 
                    gt_atom_pos, 
                    bb_atom_loss_mask[...,None], 
                    cutoff=args.exp.dist_loss_filter, 
                    no_bins=args.embed.num_lddt_bins
                )
        plddt_loss *= batch["t"] > args.exp.aux_loss_t_filter
        plddt_loss *= args.exp.plddt_loss_weight



    tm_loss = torch.zeros([batch_size]).to(device)
    if args.use_tm:
        tm_loss = loss_tm(
                  model_out["tm"], 
                  model_out["rigids_tensor"],
                  gt_rigids,
                  bb_mask,
                )
        tm_loss *= batch["t"] > args.exp.aux_loss_t_filter
        tm_loss *= args.exp.tm_loss_weight

    
    pae_loss = torch.zeros([batch_size]).to(device)
    if args.use_pae:
        gt_pae = batch["pae"].to(device)
        pred_ca_pos = pred_atom_pos[:, :, 1, :]
        pred_ca_dists = torch.linalg.norm(
            pred_ca_pos[:, :, None, :] - pred_ca_pos[:, None, :, :], dim=-1
        )
        pae_mask = bb_mask[:, :, None] * bb_mask[:, None, :]
        pae_loss = torch.sum(
            (gt_pae - pred_ca_dists) ** 2  * pae_mask, dim=(1, 2)
        )

        pae_loss = pae_loss / (pae_mask.sum(dim=(1, 2)) + 1e-10)
        pae_loss *= batch["t"] > args.exp.aux_loss_t_filter
        pae_loss *= args.exp.pae_loss_weight


    msa_loss = torch.zeros([batch_size]).to(device)
    if args.flow_msa:
        msa_mask = batch["msa_mask"][:, 0].to(device)
        gt_msa = batch["msa_1"][:, 0]
        pred_msa = model_out["msa"]
        msa_loss = F.cross_entropy(
                            input=pred_msa.reshape(-1, args.msa.num_msa_vocab), 
                            target=gt_msa.flatten().long(), 
                            reduction="none"
                ).reshape(batch_size, args.msa.num_msa_token)
        msa_loss = (msa_loss * msa_mask).sum(dim=1) / (msa_mask.sum(dim=1) + 1e-10)
        msa_loss *= args.exp.msa_loss_weight

    
    ec_loss = torch.zeros([batch_size]).to(device)
    if args.flow_ec:
        gt_ec = batch["ec_1"]
        pred_ec = model_out["ec"]
        ec_loss = F.cross_entropy(
                            input=pred_ec.reshape(-1, args.ec.num_ec_class), 
                            target=gt_ec.flatten().long(), 
                            reduction="none"
                ).reshape(batch_size)
        ec_loss *= args.exp.ec_loss_weight
    

    final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

    if args.flow_msa:
        final_loss += msa_loss

    if args.flow_ec:
        final_loss += ec_loss
        
    if args.use_aa:
        final_loss += aa_loss
        
    if args.use_fape:
        final_loss += fape_loss
        
    if args.use_plddt:
        final_loss += plddt_loss

    if args.use_tm:
        final_loss += tm_loss

    if args.use_pae:
        final_loss += pae_loss

    if args.use_struct_violation:
        final_loss += violation_loss
    
    def normalize_loss(x):
        return x.sum() / (batch_loss_mask.sum() + 1e-10)
    
    aux_data = {
        "batch_time": batch["t"],
        "batch_train_loss": final_loss,
        "batch_aa_loss": aa_loss,
        "batch_msa_loss": msa_loss,
        "batch_ec_loss": ec_loss,
        "batch_rot_loss": rot_loss,
        "batch_trans_loss": trans_loss,
        "batch_bb_atom_loss": bb_atom_loss,
        "batch_dist_mat_loss": dist_mat_loss,
        "total_loss": normalize_loss(final_loss).item(),
        "aa_loss": normalize_loss(aa_loss).item(),
        "msa_loss": normalize_loss(msa_loss).item(),
        "ec_loss": normalize_loss(ec_loss).item(),
        "violation_loss": normalize_loss(violation_loss).item(),
        "fape_loss": normalize_loss(fape_loss).item(),
        "plddt_loss": normalize_loss(plddt_loss).item(),
        "tm_loss": normalize_loss(tm_loss).item(),
        "pae_loss": normalize_loss(pae_loss).item(),
        "rot_loss": normalize_loss(rot_loss).item(),
        "trans_loss": normalize_loss(trans_loss).item(),
        "bb_atom_loss": normalize_loss(bb_atom_loss).item(),
        "dist_mat_loss": normalize_loss(dist_mat_loss).item(),
        "examples_per_step": torch.tensor(batch_size).item(),
        "res_length": torch.mean(torch.sum(bb_mask, dim=-1)).item(),
    }
    
    
    assert final_loss.shape == (batch_size,)
    assert batch_loss_mask.shape == (batch_size,)
    return normalize_loss(final_loss), aux_data