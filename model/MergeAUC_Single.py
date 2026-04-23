import pandas as pd
from sklearn import metrics
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATASET='idr2'

FPR_TPR_path = 'insert your path \\DisoPatho\\data\\FPR_TPR\\'

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # file_path0 = os.path.join(FPR_TPR_path,'train','589idrs_tpr_fpr_single_eng_neo_0.xlsx')  
    file_path0 = os.path.join(FPR_TPR_path,'train2','561idrs2_tpr_fpr_single_eng_0.xlsx')
    df0 = pd.read_excel(file_path0)
    mt0 = 'mean_tpr'
    mf0 = 'mean_fpr'
    # 将某列数据输入到一个numpy数组中
    tpr0_mean=df0[mt0].values
    fpr0_mean=df0[mf0].values
    roc_auc0_mean = metrics.auc(fpr0_mean, tpr0_mean)
  
    # file_path1 = os.path.join(FPR_TPR_path,'train','872idrs_tpr_fpr_single_esmc_site_neo_0.xlsx')  
    file_path1 = os.path.join(FPR_TPR_path,'train2','807idrs2_tpr_fpr_single_esmc_site_4.xlsx')
    df = pd.read_excel(file_path1)
    mt = 'mean_tpr'
    mf = 'mean_fpr'
    # 将某列数据输入到一个numpy数组中
    tpr_mean=df[mt].values
    fpr_mean=df[mf].values
    roc_auc_mean = metrics.auc(fpr_mean, tpr_mean)

    # 读取xlsx文件
    # file_path2 = os.path.join(FPR_TPR_path,'train','873idrs_tpr_fpr_single_esmc_neo2_0.xlsx')  
    file_path2 = os.path.join(FPR_TPR_path,'train2','807idrs2_tpr_fpr_single_esmc_neo2_4.xlsx')
    df2 = pd.read_excel(file_path2)
    mt2 = 'mean_tpr'
    mf2 = 'mean_fpr'
    # 将某列数据输入到一个numpy数组中
    tpr2_mean=df2[mt2].values
    fpr2_mean=df2[mf2].values
    roc_auc2_mean = metrics.auc(fpr2_mean, tpr2_mean)

    # file_path3 = os.path.join(FPR_TPR_path,'train','875idrs_tpr_fpr_single_esmc_dhc_8head_engm32_1.xlsx')  
    file_path3 = os.path.join(FPR_TPR_path,'train2','809idrs2_tpr_fpr_single_esmc_engm32_0.xlsx')
    df3 = pd.read_excel(file_path3)
    mt3 = 'mean_tpr'
    mf3 = 'mean_fpr'
    # 将某列数据输入到一个numpy数组中
    tpr3_mean = df3[mt3].values
    fpr3_mean = df3[mf3].values
    roc_auc3_mean = metrics.auc(fpr3_mean, tpr3_mean)


    # file_path4 = os.path.join(FPR_TPR_path,'train','892idrs_tpr_fpr_single_pglm_0.xlsx')  
    file_path4 = os.path.join(FPR_TPR_path,'train2','836idrs2_tpr_fpr_single_pglm_1.xlsx')
    df4 = pd.read_excel(file_path4)
    mt4 = 'mean_tpr'
    mf4 = 'mean_fpr'
    # 将某列数据输入到一个numpy数组中
    tpr4_mean = df4[mt4].values
    fpr4_mean = df4[mf4].values
    roc_auc4_mean = metrics.auc(fpr4_mean, tpr4_mean)

    # 读取xlsx文件
    # file_path5 = os.path.join(FPR_TPR_path,'train','891idrs_tpr_fpr_ori.xlsx')  
    file_path5 = os.path.join(FPR_TPR_path,'train2','834idrs2_tpr_fpr_ori_0.xlsx')
    df5 = pd.read_excel(file_path5)
    mt5 = 'mean_tpr'
    mf5 = 'mean_fpr'
    # 将某列数据输入到一个numpy数组中
    tpr5_mean = df5[mt5].values
    fpr5_mean = df5[mf5].values
    roc_auc5_mean = metrics.auc(fpr5_mean, tpr5_mean)

    # 读取xlsx文件
    # file_path6 = os.path.join(FPR_TPR_path,'train','899idrs_tpr_fpr_ori_gdxpglm_dhc_8head_engm32.xlsx')  
    file_path6 = os.path.join(FPR_TPR_path,'train2','840idrs2_tpr_fpr_ori_gdxpglm_dhc_8head_engm32_0.xlsx')
    df6 = pd.read_excel(file_path6)
    mt6 = 'mean_tpr'
    mf6 = 'mean_fpr'
    # 将某列数据输入到一个numpy数组中
    tpr6_mean = df6[mt6].values
    fpr6_mean = df6[mf6].values
    roc_auc6_mean = metrics.auc(fpr6_mean, tpr6_mean)


    # 绘制ROC曲线图
    plt.figure(figsize=(9, 8))
    plt.rcParams['font.family'] = 'Arial'
   
    plt.plot(fpr6_mean, tpr6_mean, color='#e84d84', label='$\mathregular{ESM+Eng+xTP_{E}}$ ROC curve (AUC = %0.3f)' % roc_auc6_mean)
    plt.plot(fpr5_mean, tpr5_mean, color='#EDB31E', label='ESM+Eng+xTP ROC curve (AUC = %0.3f)' % roc_auc5_mean)

    plt.plot(fpr3_mean, tpr3_mean,  color='#0e8585', label='ESM+Eng ROC curve (AUC = %0.3f)' % roc_auc3_mean)
    
    plt.plot(fpr2_mean, tpr2_mean,  color='#4c9be6', label='ESM Feature ROC curve (AUC = %0.3f)' % roc_auc2_mean)
    

    # plt.plot(fpr4_mean, tpr4_mean,  color='#e84d84', label='Single xTP Feature ROC curve (AUC = %0.3f)' % roc_auc4_mean)
    
    # plt.plot(fpr2_mean, tpr2_mean,  color='#EDB31E', label='Single ESM Feature ROC curve (AUC = %0.3f)' % roc_auc2_mean)

    # plt.plot(fpr_mean, tpr_mean, color='#0e8585', label='Single $\mathregular{ESM_{Site}}$ Feature ROC curve (AUC = %0.3f)' % roc_auc_mean)    
    
    # plt.plot(fpr0_mean, tpr0_mean, color='#4c9be6', label='Single Eng Feature ROC curve (AUC = %0.3f)' % roc_auc0_mean)    

    
    plt.plot([0, 1], [0, 1], 'darkgray', linestyle='--')
    plt.xlim([-.05, 1.05])
    plt.ylim([-.05, 1.05])
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel('False Positive Rate',fontsize=19)
    plt.ylabel('True Positive Rate',fontsize=19)
    plt.title('Samples Ablation Study ROC Curves', fontsize=20)
    plt.legend(loc='lower right',fontsize=13.5)
    plt.savefig(f'pic/{DATASET}_as_MergeCurve.png', dpi=300)
    # plt.savefig(f'pic/{DATASET}_Single_as_MergeCurve.png', dpi=300)
    plt.show()

