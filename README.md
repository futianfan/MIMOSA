# MIMOSA


## Install & Usage


```bash

cd MIMOSA
# Change directory to MIMOSA

conda env create -f mimosa.yml   
## Build virtual environment with all packages installed using conda

conda activate mimosa
## Activate conda environment (use "source activate mimosa" for anaconda 4.4 or earlier) 


## pretrain mGNN   -> the trained model is saved at ./trained_model
python pretrain_mGNN.py 

## pretrain bGNN   -> the trained model is saved at ./trained_model 
python pretrain_mGNN.py 


###### Evaluation code ######
## jointly optimize DRD and PLogP 
python test_drd_logp.py

## jointly optimize QED and PLogP 
python test_qed_logp.py





conda deactivate 
## Exit conda environment 

```



## Related Projects

This repository relies on following repositories: 
* [VJTNN](https://github.com/wengong-jin/iclr19-graph2graph) (Graph-to-Graph Translation, ICLR 19) 
* [Pretrain GNN](https://github.com/snap-stanford/pretrain-gnns) (Strategies for Pre-training Graph Neural Networks, ICLR 20)


## Dataset

### pretraining data: property-agnostic

* ./dataset/zinc_standard_agent: not uploaded(4.5G); publicly available at http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip  




## Util File 

* ./trained_model/   directory to save the pretrained model 
* pretrain_mGNN.py  (pretrain_masking.py)
* test_qed_logp.py  jointly optimize QED and PLogP 
* test_drd_logp.py  jointly optimize DRD and PLogP  
* props   not uploaded because the size is too large. 



















