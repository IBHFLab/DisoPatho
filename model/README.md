## Model Files

### Encoder.py
Transformer encoder module.

### GateDXHCEncoder.py
Head-wise DHC framework, initialized with **global context**; Gate is configured per attention **head**.

### GateRDXHCEncoder.py
Element-wise DHC framework, initialized at the **token level**; Gate is configured per **token**.

### MergeAUC_Single.py
Reads files within `/data/FPR_TPR` to obtain the average results of the five-fold cross-validation presented in the paper.

### model.py
model.

### predict.py test_indep_modify.py train.py
predict; test; train, and test in independent test set. 

## Dataset
We utilize the dataset created by Fawzy et al. Please refer to **Text S1-S3** for details.
Associated datasets are available at https://doi.org/10.6084/m9.figshare.c.7747895.v1 and the pipeline code is shared at https://github.com/drsamibioinfo/VEPS_IN_DISORDER/.

## independent test file
Due to GitHub's file size limits, please download from Zenodo (https://zenodo.org/records/19702412) and overwrite the `/features_npy/` directory.

## Weights
To ensure reproducibility, we provide 5 weight files trained on Data1 (`/save_model/ten/idrs/`). The results listed below serve as baselines for subsequent work. You can use the `/model/test.py` script to generate the prediction outputs.

| **MCC** | **ACC** | **AUC** | **recall** | **Specificity** | **Precision** | **F1** | **AUPRC** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.6913 | 0.8690 | 0.9092 | 0.7065 | 0.9459 | 0.8606 | 0.7760 | 0.8307 |
| 0.6923 | 0.8690 | 0.9137 | 0.7313 | 0.9341 | 0.8400 | 0.7819 | 0.8288 |
| 0.6934 | 0.8690 | 0.9124 | 0.6517 | 0.9718 | 0.9161 | 0.7616 | 0.8398 |
| 0.6955 | 0.8706 | 0.9092 | 0.6816 | 0.9600 | 0.8896 | 0.7718 | 0.8367 |
| 0.7011 | 0.8722 | 0.9098 | 0.6617 | 0.9718 | 0.9172 | 0.7688 | 0.8438 |

### Citation
Fawzy, M., & Marsh, J. A. (2025). Assessing variant effect predictors and disease mechanisms in intrinsically disordered proteins. *PLOS Computational Biology*, *21*(8), e1013400. https://doi.org/10.1371/journal.pcbi.1013400
