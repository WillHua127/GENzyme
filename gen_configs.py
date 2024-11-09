class Args:
    #1 data arguments
    pdb_name = 'Q8N4T8'
    substrate_smiles = 'CC(=O)CC(=O)CC(=O)OC'
    product_smiles = 'CC(=O)C[C@@H](CC(=O)OC)O'
    
    n_pocket_res = 64
    n_protein_res = 300
    min_t = 0.0
    max_t = 1.0
    num_pocket_design_t = 50
    inpaint_pocket = True
    
    
    # import args to load inpainting module
    n_sample_enzyme = 8
    inpaint_method = 'gibbs'
    max_inpaint_t = 1.0
    num_inpaint_t = 100
    ptm_filter = 0.65
    plddt_filter = 0.65

    
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
    use_aa = True
    use_fape = True
    use_plddt = True
    use_tm = True
    use_pae = False
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


    #4 MSA arguments
    class msa:
        num_msa = 1
        num_msa_vocab = 64 #fixed
        num_msa_token = 500
        msa_layers = 2
        msa_heads = 4
        msa_hidden_size = 128
        msa_embed_size = 32 #node_embed_size

        
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
        embed_self_conditioning = False
        relpos_k = 64
        feat_dim = 64
        num_bins = 22
        num_lddt_bins = 50
        num_tm_bins = 64
        
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


    class inverse_folding:
        hidden_dim = 128 #node_embed_size
        node_features = 128
        edge_features = 128
        k_neighbors = 30
        dropout = .1
        num_encoder_layers = 10
        updating_edges = 4
        node_dist = 1
        node_angle = 1
        node_direct = 1
        edge_dist = 1
        edge_angle = 1
        edge_direct = 1
        virtual_num = 3
        
    class inpainting:
        noise_type = 'loglinear'
        noise_removal = True
        time_conditioning = True
        change_of_variables = False
        importance_sampling = False
        sequence_prediction = True
        condition_dropout = 0.
        condition_mask_rate = 0.
        coupled_condition_mask = False
        structure_only = False
        antithetic_sampling = True
        structure_encoder_hidden = 1024
        structure_encoder_layer = 2
        structure_encoder_out = 128
        structure_encoder_v_heads = 128
        sigma_embedder_hidden = 1536
        n_structure_heads = 4101
        n_sequence_heads = 32
        attn_n_heads = 24
        attn_v_heads = 256
        n_layers = 48
        pretrained=True
        freeze_codebook=True


    #5 evaluation arguments
    class eval:
        sample_from_multinomial = False
        eval_dir = 'generated'
        record_extra = False
        discrete_purity = True
        discrete_temp = 5.
