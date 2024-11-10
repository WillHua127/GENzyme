# Reaction-conditioned De Novo Enzyme Design with GENzyme

GENzyme enables de novo design of catalytic pockets, enzymes, and enzyme-substrate complexes for any reaction. Simply change ```args.substrate_smiles``` and  ```args.product_smiles``` to customized substrate SMILES and product SMILES in ```gen_configs.py```, then run ```python generate.py```, you can design your own enzymes. 

#### Please make sure you have [ESM3](https://github.com/evolutionaryscale/esm/tree/main) installed or have access to [ESM3](https://github.com/evolutionaryscale/esm/tree/main).

![genzyme](./image/genzyme.jpg)

![workflow](./image/workflow.jpg)


### Requirement
```
python>=3.11
CUDA=12.1
torch==2.4.1 (>=2.0.0)
torch_geometric==2.4.0
torch_scatter==2.1.2

pip install mdtraj==1.10.0 (do first will install numpy, scipy as well, install later might raise dependency issues)
pip install esm==3.0.7.post1
pip install pytorch-warmup==0.1.1
pip install POT==0.9.4
pip install rdkit==2023.9.5
pip install biopython==1.84
pip install tmtools==0.2.0
pip install geomstats==2.7.0
pip install dm-tree==0.1.8
pip install ml_collections==0.1.1
pip install torchmetrics==0.11.4
pip install OpenMM
pip install einx
pip install einops

conda install conda-forge::pdbfixer
```

In case if you want to use the pocket-specific binding module:
```
For binding module, we use UniMol Docking v2, you need to install [UniCore](https://github.com/dptech-corp/Uni-Core)
```


## Model Weights

You should download GENzyme checkpoint at [Google drive](https://drive.google.com/file/d/1R39bvQwUKqIXeqf4RIsuK-K6RWq4P1gj/view?usp=sharing). Once you download it, put it under ```genzyme_ckpt``` folder, namely ```genzyme_ckpt/genzyme.ckpt```.

## Model Inference
1. ```gen_configs.py``` contain all inference configurations and hyperparameters. In ```gen_configs.py```, change ```args.pdb_name``` to one pdb file (set to None for de novo design ```args.pdb_name = None```), ```args.substrate_smiles``` to one substrate SMILES, and ```args.product_smiles``` to one product SMILES, to customize reaction.

2. GENzyme inference script ```generate.py``` is provided. Run ```python generate.py``` for de novo enzyme design. To customize catalytic reaction, remeber to change the subsrtate SMILES and product SMILES in ```gen_configs.py```.

3. GENzyme reproduce script ```reproduce.py``` is provided. Run ```python reproduce.py``` for reproduction.

## Model Training

1. ```configs.py``` contain all training configurations and hyperparameters.

2. Train model using ```train.py``` for single GPU training. Run ```python train.py``` for training.

   
## License
No Commercial use of either the model nor generated data, details to be found in license.md.
