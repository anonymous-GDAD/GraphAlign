# Anonymous code for GDAD
Anonymous Code Submission for KDD'24: 'Can Modifying Data Address Graph Domain Adaptation?'

## Requirements
Please see [requirements.txt](https://github.com/ChandlerBang/GCond/blob/main/requirements.txt).
```
torch==1.7.0
torch_geometric==1.6.3
scipy==1.6.2
numpy==1.19.2
ogb==1.3.0
tqdm==4.59.0
torch_sparse==0.6.9
deeprobust==0.2.4
scikit_learn==1.0.2
```

## Download Datasets
For ACMv9, DBLPv7, Citationv1, datasets are available for download from the link by the repository of [AdaGCN](https://github.com/daiquanyu/AdaGCN_TKDE). For ACM_small and DBLP_small, the dataset are available for download in the repository of [AdaGCN](https://github.com/GRAND-Lab/UDAGCN). The remaining datasets, such as Cora-degree, Cora-word, Arxiv-degree, and Arxiv-time, will be downloaded directly from GOOD using the code in [GOOD](https://github.com/divelab/GOOD).

- `dual_gnn&data.py`: contains data processing for Acm, dblp, and citation.
- `models/`: includes GNN prototypes used by the GDAD model.
- `Utils/`: contains data processing for cora and arxiv.
- `generator/`: graph generator used in the GDAD model for graph generation.
- `train.py`: main program for the GDAD model.



## Run the code
For unsupervised domain adaptation task (A-D), please run the following command:
```
python train.py --dataset acm --source acm --target dblp --epoch 500 --dis_metric ours --alpha 30 --method mmd --reduction_rate 0.01 --gpu_id 0 
```

For unsupervised domain adaptation task (Asmall-Dsmall), please run the following command:
```
python train.py --dataset acm --source acm --target dblp --epoch 500 --dis_metric ours --alpha 30 --method mmd --reduction_rate 0.01 --version old --gpu_id 0 
```


