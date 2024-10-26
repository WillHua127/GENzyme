class Args:
    ckpt_path = 'checkpoint/gen_model.ckpt'
    
    #1 data arguments
    min_t = 0.0
    max_t = 1.0
    num_t = 20

    #2 flow matcher arguments
    ot_plan = False
    flow_trans = True
    flow_rot = True

    # amino-acid flow
    flow_aa = True
    discrete_flow_type = 'masking'
    # msa flow
    flow_msa = False

    # alphafold loss
    use_aa = False
    use_fape = True
    use_plddt = False
    use_pae = False
    use_tm = False
    use_struct_violation = False
    class r3:
        min_b = 0.01
        min_sigma = 0.01
        max_b = 20.0
        coordinate_scaling = 1. #dont scale coordinates
        g = 0.1

    class so3:
        min_sigma = 0.01
        max_sigma = 1.5
        axis_angle = True
        inference_scaling = 0.01
        g = 0.1

    #3 model arguments
    guide_by_condition = True
    pretrain_kd_pred = True
    num_aa_type = 20 #fixed
    num_atom_type = 95 #fixed
    node_embed_size = 256
    edge_embed_size = 128
    dropout = 0.
    num_rbf_size = 16
    ligand_rbf_d_min = 0.05
    ligand_rbf_d_max = 6.
    bb_ligand_rbf_d_min = 0.5
    bb_ligand_rbf_d_max = 6.

        
    class mpnn:
        num_edge_type = 4
        dropout = 0.
        n_timesteps = 16
        mpnn_layers = 3
        mpnn_node_embed_size = 256 #node_embed_size
        mpnn_edge_embed_size = 128 #edge_embed_size

    class embed:
        c_s = 256 #node_embed_size
        c_pos_emb = 128
        c_timestep_emb = 128
        timestep_int = 1000
    
        c_z = 128 #edge_embed_size
        embed_self_conditioning = True
        relpos_k = 64
        feat_dim = 64
        num_bins = 22
        
    class ipa:
        c_s = 256 #node_embed_size
        c_z = 128 #edge_embed_size
        c_hidden = 16
        # c_skip = 16
        no_heads = 8
        no_qk_points = 8
        no_v_points = 12
        seq_tfmr_num_heads = 4
        seq_tfmr_num_layers = 4
        num_blocks = 20
        coordinate_scaling = .1 #r3.coordinate_scaling


    #5 evaluation arguments
    class eval:
        noise_scale = 1.
        dist_loss_filter = 8.
        sample_from_multinomial = False
        eval_dir = 'generated'
        record_extra = False
        discrete_purity = True
        self_condition = True
        discrete_temp = 5.
        aa_noise = 40.0
        msa_noise = 40.0
        rot_sample_schedule = 'exp'
        trans_sample_schedule = 'linear'
        n_sample = 100