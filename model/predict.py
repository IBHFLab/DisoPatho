from test_indep_modify import fcvtest

Project_Path='insert your path'
'''
idrs->Data1
idrs2->Data2
'''
DATASET = 'idrs'

fcvtest(Project_Path + f'save_model/fivecv_model/ten/{DATASET}/model_221_82_last_7011_neo.h5',Project_Path + f'save_model/fivecv_model/ten/{DATASET}/model_221_82_last_7011_neo_metrics_neo.xlsx',Project_Path + f'save_model/fivecv_model/ten/{DATASET}/model_221_82_last_7011_neo_all.xlsx',datasets=DATASET)
print("Done")