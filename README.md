# contr_np
Code for paper "Contrastive Neural Processes for Time Series"

This folder includes the code for Contrastive Neural Processes and Baselines.
Code has been modified accordingly to the needs of the project. Original sources are cited here:

Folders:
- Baselines : Includes all baselines, hyperparameters and evaluation metrics used for base experiments
- ContrNP : Includes code for ContrNP method and resources
- Results : Location where weights and results are saved
- Data : Location where datasets are located. Use *data_name*_load.py to download and extract data. 
(Please Note: Some commands for data extraction are Ubuntu specific)



In folder baselines please find implementations for Tloss [1], CPC [2], TNC [2] and SimCLR [3].
Folders: npf, utils are for the implementation of Neural Processes [4].
Main Implementation of contrastive convolutional cnp: Contrastive-ConvCNP-SSL.ipynb
Implementation of Self supervised convolutional cnp: ConvCNP-SSL.ipynb
Implementation of Self supervised cnp: CNP-SSL.ipynb


[1]https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
[2]https://openreview.net/forum?id=8qDwejCuCN
[3]https://github.com/Spijkervet/SimCLR
[4]https://github.com/YannDubs/Neural-Process-Family


### Reference



[[ACML2021]](http://www.acml-conf.org/2021/conference/accepted-papers/266/) [[arXiv]](https://arxiv.org/abs/2110.13623)

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
