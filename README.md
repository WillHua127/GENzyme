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

GENzyme inference [demo](https://github.com/WillHua127/GENzyme/blob/main/generate.py) is provided. Change ```args.pdb_name args.substrate_smiles args.product_smiles``` in gen_configs.py to customize reaction.

## Model Training

## License
No Commercial use of either the model nor generated data, details to be found in license.md.
