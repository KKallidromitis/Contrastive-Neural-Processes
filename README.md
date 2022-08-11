# Contrastive Neural Processes
PyTorch implementation of "[Contrastive Neural Processes for Self-Supervised Learning](https://arxiv.org/abs/2110.13623)"

## Implementation Details

This folder includes the code for Contrastive Neural Processes and Baselines.
Code has been modified accordingly to the needs of the project. Original sources are cited here:

Folders:
- Baselines : Includes all baselines, hyperparameters and evaluation metrics used for base experiments
- ContrNP : Includes code for ContrNP method and resources
- Results : Location where weights and results are saved
- Data : Location where datasets are located. Use *data_name*_load.py to download and extract data. 
(Please Note: Some commands for data extraction are Ubuntu specific)



Baselines includes implementations for Tloss [[1]](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries), CPC [[2]](https://openreview.net/forum?id=8qDwejCuCN), TNC [[2]](https://openreview.net/forum?id=8qDwejCuCN) and SimCLR [[3]](https://github.com/Spijkervet/SimCLR).

Folders: npf, utils are for the implementation of Neural Processes [[4]](https://github.com/YannDubs/Neural-Process-Family).
- Main Implementation of contrastive convolutional cnp: Contrastive-ConvCNP-SSL.ipynb
- Implementation of Self supervised convolutional cnp: ConvCNP-SSL.ipynb
- Implementation of Self supervised cnp: CNP-SSL.ipynb


## Citing this work


[[arXiv]](https://arxiv.org/abs/2110.13623) [[PMLR]](https://proceedings.mlr.press/v157/kallidromitis21a) [[ACML2021]](http://www.acml-conf.org/2021/conference/accepted-papers/266/) 

```
@misc{kallidromitis2021contrastive,
      title={Contrastive Neural Processes for Self-Supervised Learning}, 
      author={Konstantinos Kallidromitis and Denis Gudovskiy and Kazuki Kozuka and Ohama Iku and Luca Rigazio},
      year={2021},
      eprint={2110.13623},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
## Requirements
Full list [here](./requirements.txt)
```
python>=3.6.9
skorch==0.8
pytorch>=1.3.1
wfdb
scikit-image
```
