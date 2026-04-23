import tensorflow as tf
import numpy as np
from model import get_model
import os
import gc
from scipy import interp
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef, f1_score, roc_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve
import openpyxl as op
from roc_utils import *
import warnings
warnings.filterwarnings("ignore")

Project_Path='insert your path'
SEQUENCE_LEN = 221
DATANAME = 'idrs'
'''
idrs->Data1
idrs2->Data2
'''
SUFFIX='neo_test_ex'

filename =Project_Path + f'/result/ten/{DATANAME}/ten_test_dhc_8head_qpv_meng32_{SUFFIX}.xlsx'

def op_toexcel(data, filename): # openpyxl库储存 数据到excel

    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]
        ws.append(data) # 每次写入一行
        wb.save(filename)
    else:
        wb = op.Workbook()  # 创建工作簿对象
        ws = wb['Sheet']  # 创建子表
        ws.append(['MCC', 'ACC', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1', 'FPR', 'FNR',
                  'TN', 'FP', 'FN', 'TP', 'AUPRC', 'Threshold'])  # 添加表头
        ws.append(data) # 每次写入一行
        wb.save(filename)

def data_generator(train_pglm, train_esmc, train_eng,train_y, batch_size):

    L = train_esmc.shape[0]

    while True:
        for i in range(0, L, batch_size):

            batch_esmc = train_esmc[i:i + batch_size].copy()
            batch_pglm = train_pglm[i:i + batch_size].copy()
            batch_eng = train_eng[i:i + batch_size].copy()
            batch_y = train_y[i:i + batch_size].copy()

            yield ([batch_pglm, batch_esmc, batch_eng], batch_y)

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    all_esmc = np.lib.format.open_memmap(Project_Path + f'/features_npy/esm/{DATANAME}/idr{SEQUENCE_LEN}.npy')
    all_pglm = np.lib.format.open_memmap(Project_Path + f'/features_npy/xtrimopglm/{DATANAME}/idr{SEQUENCE_LEN}.npy')
    all_eng = np.lib.format.open_memmap(Project_Path + f'/features_npy/energy/{DATANAME}/idr{SEQUENCE_LEN}_eng.npy')
    all_label = np.lib.format.open_memmap(Project_Path + f'/features_npy/labels/{DATANAME}/idr{SEQUENCE_LEN}_labels.npy')
   
    for i in range(10):
            
        tt = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

        k = i

        for train_index, test_index in tt.split(all_esmc, all_label):

            # 训练集
            train_esmc = all_esmc[train_index]
            train_pglm = all_pglm[train_index]
            train_eng = all_eng[train_index]
            train_label = all_label[train_index]

            # 打乱训练集顺序并划分出验证集
            # （1）分层打乱
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
            for train_inx, valid_inx in split.split(train_esmc, train_label):
                # 验证集
                valid_esmc = train_esmc[valid_inx]
                valid_pglm = train_pglm[valid_inx]
                valid_eng = train_eng[valid_inx]
                valid_label = train_label[valid_inx]

                # 训练集
                train_esmc = train_esmc[train_inx]
                train_pglm = train_pglm[train_inx]
                train_eng = train_eng[train_inx]
                train_label = train_label[train_inx]


            # 测试集
            test_esmc = all_esmc[test_index]
            test_pglm = all_pglm[test_index]
            test_eng = all_eng[test_index]
            test_label = all_label[test_index]
            
            # 训练、验证each epoch的步长
            train_size = train_label.shape[0]
            val_size = valid_label.shape[0]
            batch_size = 32
            train_steps = train_size // batch_size
            val_steps = val_size // batch_size

            print(f"Cycle {k} - Training samples: {train_esmc.shape[0]}, Test samples: {test_esmc.shape[0]}")

            qa_model = get_model()

            valiBestModel = Project_Path + f'/save_model/fivecv_model/ten/{DATANAME}/model_{SEQUENCE_LEN}_{k}.h5'
            lastEpochModel = Project_Path + f'/save_model/fivecv_model/ten/{DATANAME}/model_{SEQUENCE_LEN}_{k}_last.h5'
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
                filepath=valiBestModel,
                monitor='val_loss',
                save_weights_only=True,
                verbose=1,
                save_best_only=True
            )
            save_last_epoch = tf.keras.callbacks.ModelCheckpoint(
                filepath=lastEpochModel,
                save_weights_only=True,
                verbose=1,
                save_freq='epoch'  
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=0,
                mode='auto'
            )

            train_generator = data_generator(train_pglm, train_esmc, train_eng, train_label, batch_size)
            val_generator = data_generator(valid_pglm, valid_esmc, valid_eng, valid_label, batch_size)

            history_callback = qa_model.fit_generator(
                train_generator,
                steps_per_epoch=train_steps,
                epochs=500,
                verbose=1,
                callbacks=[checkpointer, save_last_epoch, early_stopping],
                validation_data=val_generator,
                validation_steps=val_steps,
                shuffle=True,
                workers=1
            )

            train_generator.close()
            val_generator.close()

            print(f"\nCycle {k} - Validation Loss: {history_callback.history['val_loss'][-1]:.4f}, " +
                f"Validation Accuracy: {history_callback.history['val_accuracy'][-1]:.4f}")

            print(f"Cycle {k} - Testing:")

            test_pred = qa_model.predict([test_pglm, test_esmc, test_eng]).reshape(-1, )

            y_pred = test_pred
            y_true = test_label
            y_pred_new = []

            best_f1 = 0
            best_threshold = 0.5
            for threshold in range(0, 1000):
                threshold = threshold / 1000
                binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
                f1 = metrics.f1_score(y_true, binary_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            for value in y_pred:
                if value < best_threshold:
                    y_pred_new.append(0)
                else:
                    y_pred_new.append(1)
            y_pred_new = np.array(y_pred_new)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()
            fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_new)
            auprc = metrics.auc(recall, precision)
            thd = best_threshold

            print("Matthews相关系数: " + str(matthews_corrcoef(y_true, y_pred_new)))
            print("ACC: ", (tp + tn) / (tp + tn + fp + fn))
            print("AUC: ", roc_auc)
            print('sensitivity/recall:', tp / (tp + fn))
            print('specificity:', tn / (tn + fp))
            print('precision:', tp / (tp + fp))
            print('negative predictive value:', tn / (tn + fn))
            print("F1值: " + str(f1_score(y_true, y_pred_new)))
            print('error rate:', fp / (tp + tn + fp + fn))
            print('false positive rate:', fp / (tn + fp))
            print('false negative rate:', fn / (tp + fn))
            print('TN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)
            print('AUPRC: ' + str(auprc))
            print('best_threshold: ' + str(best_threshold))

            mcc = float(format((matthews_corrcoef(y_true, y_pred_new)), '.4f'))
            acc = float(format((tp + tn) / (tp + tn + fp + fn), '.4f'))
            auc = float(format(roc_auc, '.4f'))
            sen = float(format(tp / (tp + fn), '.4f'))
            spe = float(format(tn / (tn + fp), '.4f'))
            pre = float(format(tp / (tp + fp), '.4f'))

            npv = float(format(tn / (tn + fn), '.4f'))
            f1 = float(format(f1_score(y_true, y_pred_new), '.4f'))
            fpr = float(format(fp / (tn + fp), '.4f'))
            fnr = float(format(fn / (tp + fn), '.4f'))
            auprc = float(format(auprc, '.4f'))

            #保存每一次跑的结果到excel表格
            result = mcc, acc, auc, sen, spe, pre, npv, f1, fpr, fnr, tn, fp, fn, tp, auprc, thd
            op_toexcel(result, filename)

            k += 1
            
            del qa_model
            tf.keras.backend.clear_session()
            gc.collect()







