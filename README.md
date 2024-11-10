# Reaction-conditioned De Novo Enzyme Design with GENzyme

![genzyme](./image/genzyme.jpg)

![workflow](./image/workflow.jpg)


### Requirement
```
python>=3.11
CUDA=12.1
torch==2.4.1 (>=2.0.0)
torch_geometric==2.4.0

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
pip install OpenMM
pip install einx
pip install einops

conda install conda-forge::pdbfixer

For binding module, we use UniMol Docking v2, you need to install [UniCore](https://github.com/dptech-corp/Uni-Core)
```
## Model Weights

## Model Inference

1. GENzyme inference script ```generate.py``` is provided. Change ```args.pdb_name``` ```args.substrate_smiles``` and ```args.product_smiles``` in ```gen_configs.py``` to customize reaction. Run ```python generate.py```.

2. GENzyme reproduce script ```reproduce.py``` is provided. Run ```python reproduce.py```.

## Model Training

1. ```configs.py``` contain all training configurations and hyperparameters.

2. Train model using ```train.py``` for single GPU training. Run ```python train.py```.

   
## License
No Commercial use of either the model nor generated data, details to be found in license.md.
