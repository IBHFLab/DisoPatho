# DisoPatho
Accurate prediction of mutations within intrinsically disordered regions (IDRs) is crucial for advancing disease diagnosis and drug discovery. However, the intrinsic lack of stable structural conformations and the high sequence variability of IDRs make it challenging for existing predictors to achieve robust performance in these regions. Here, we introduce DisoPatho, a deep learning framework specifically tailored for predicting disease-associated variants in IDRs. DisoPatho features a novel mutation-centric architecture that utilizes the variant site as an anchor for feature construction and interaction. The core innovation lies in a cross-view adaptive feature interaction mechanism, which synergistically integrates IDR-specific energy representations with embeddings from protein language models, including xTrimoPGLM and Evolutionary Scale Modeling. This strategy enables the comprehensive capture of evolutionary constraints and physicochemical patterns without explicitly relying on structural data or homologous sequences. Consequently, DisoPatho exhibits enhanced discriminative power perfectly suited to the highly flexible nature of IDRs. Comprehensive evaluations across multiple IDR datasets demonstrate that DisoPatho substantially outperforms existing methods. In 5-fold cross-validation, it achieves average AUCs of 0.899 and 0.840 on two datasets constructed under different evolutionary conservation constraints. Notably, on a highly confounded independent test set where phylogenetic constraints offer limited discriminative signals, DisoPatho yields a 50.2% improvement in MCC over AlphaMissense on their respective predictable variants, while achieving broader prediction coverage. In-depth analyses of the prediction results further confirm the effectiveness and stability of the framework in IDR-specific scenarios. 

## Overall architecture of DisoPatho
![image](https://github.com/IBHFLab/DisoPatho/blob/main/pic/disopatho.png)

## Install Dependencies
Python ver. == 3.9  
For others, run the following command:  
```Python
conda install tensorflow-gpu==2.5.0
pip install numpy==1.23.5
```
## Run
We provide results of two data sets. Run `/model/predict.py`; To train your own data, use the `/model/train.py` file.

## Weights
To ensure reproducibility, we provide 5 weight files trained on Data1 (`/save_model/ten/idrs/`). The results listed below serve as baselines for subsequent work. You can use the `/model/test.py` script to generate the prediction outputs.

## Contact
If you are interested in our work, OR, if you have any suggestions/questions about our work, PLEASE contact with us. E-mail: 202519159678@stu.cqu.edu.cn

