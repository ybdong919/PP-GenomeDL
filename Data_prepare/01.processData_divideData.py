# a script to divide all data to train and test 

import numpy as np
import pandas as pd

all_df=pd.read_csv('data/data.tsv',sep='\t',index_col=False)
#check shape, column names and first column values
#all_df.shape
#all_df.columns
#all_df.iloc[:,0]

samples_df=all_df.iloc[:,2:]
# check shape, column names and first column values
# samples_df.shape
# samples_df.columns
# samples_df.iloc[:,0]

features_df=all_df.iloc[:,:2]
# check shape, column names and first column values
# features_df.shape
# features_df.columns
# features_df.iloc[:,0]

# shuffle the list of columns
columns=list(samples_df.columns)
np.random.shuffle(columns)

# reindex the dataframe with the shuffled columns
samples_df_shuffled=samples_df.reindex(columns=columns)

# check column number
# samples_df_shuffled.shape[1]

#select 80% samples as train, 20% as test
divide_index=int(samples_df_shuffled.shape[1]*0.8)
samples_df_shuffled_train=samples_df_shuffled.iloc[:,:divide_index]
samples_df_shuffled_test=samples_df_shuffled.iloc[:,divide_index:]

#check shap of train
#samples_df_shuffled_train.shape

#combine samples with features
train_df=pd.concat([features_df, samples_df_shuffled_train], axis=1)
test_df=pd.concat([features_df, samples_df_shuffled_test], axis=1)

# check shape and columns
# train_df.columns
# train_df.shape

#######################################################################################
########### the following parts are similar with  01.processData.py  line58-end   #####
#######################################################################################

################## Part1: generate log10 normalized train data #########################################

train_df=train_df.set_index('features')


##### According to all_df output, 'cancer_type''day2birth','day2death','dayrecord' are located in the row 20529 to 20532 (Note: the first row is row 0).
train_df_without_labels=train_df.drop(['cancer_type','day2birth','day2death','dayrecord'], axis=0)
train_data_dict={'features':list(train_df_without_labels.index),'samples':{}}

for sample in train_df.columns:
    if 'TCGA' in sample:
        tmp=list(train_df.loc[:,sample])
        day2birth=tmp[20530]
        day2death=tmp[20531]
        dayrecord=tmp[20532]
        cancertype=tmp[20529]
        #print(cancertype, day2birth, day2death, dayrecord)
        #print("\n")
        del tmp[20529:20533]
        genes=tmp
        train_data_dict['samples'][sample]={'day2birth':day2birth,'day2death':day2death,'dayrecord':dayrecord,'cancertype':cancertype,'genes':genes}   #####Note: it is a dictionary including more dictionaries inside.
np.save('data/train_data.npy', train_data_dict)

#### Note: numpy just save a dictorary with multi-levels. 
####       The fist-level dictionay have keys of 'features' and 'samples'. The value of the key "features" is a list of gene names. The value of the key "samples" is the another dictionary.
####       The second-level dictionary is the value of the key "samples". The key of this dictionary is the sample names. The value is third dictionary.
####       The third-level dictionary include keys of 'day2birth','day2death', 'dayrecord','cacertype','genes'. Among them, the value of the key "genes" is a list of gene expression values.   

train_data=np.load('data/train_data.npy',allow_pickle=True)
train_data=train_data.item()
for sample in train_data['samples']:
    tmp=[float(x) for x in train_data['samples'][sample]['genes']]
    tmp=np.log10(np.asarray(tmp)+0.0001)
    _min=np.min(tmp)
    _max=np.max(tmp)
    train_data['samples'][sample]['genes']=(np.asarray(tmp)-_min)/(_max-_min)
np.save('data/train_data_norm_log10.npy', train_data)
#print(train_data)


############################### Part2: generate log10 normalized test data ###########################################################################

test_df=test_df.set_index('features')


##### According to all_df output, 'cancer_type''day2birth','day2death','dayrecord' are located in the row 20529 to 20532 (Note: the first row is row 0).
test_df_without_labels=test_df.drop(['cancer_type','day2birth','day2death','dayrecord'], axis=0)
test_data_dict={'features':list(test_df_without_labels.index),'samples':{}}

for sample in test_df.columns:
    if 'TCGA' in sample:
        tmp=list(test_df.loc[:,sample])
        day2birth=tmp[20530]
        day2death=tmp[20531]
        dayrecord=tmp[20532]
        cancertype=tmp[20529]
        #print(cancertype, day2birth, day2death, dayrecord)
        #print("\n")
        del tmp[20529:20533]
        genes=tmp
        test_data_dict['samples'][sample]={'day2birth':day2birth,'day2death':day2death,'dayrecord':dayrecord,'cancertype':cancertype,'genes':genes}   #####Note: it is a dictionary including more dictionaries inside.
np.save('data/test_data.npy', test_data_dict)

#### Note: numpy just save a dictorary with multi-levels. 
####       The fist-level dictionay have keys of 'features' and 'samples'. The value of the key "features" is a list of gene names. The value of the key "samples" is the another dictionary.
####       The second-level dictionary is the value of the key "samples". The key of this dictionary is the sample names. The value is third dictionary.
####       The third-level dictionary include keys of 'day2birth','day2death', 'dayrecord','cacertype','genes'. Among them, the value of the key "genes" is a list of gene expression values.   

test_data=np.load('data/test_data.npy',allow_pickle=True)
test_data=test_data.item()
for sample in test_data['samples']:
    tmp=[float(x) for x in test_data['samples'][sample]['genes']]
    tmp=np.log10(np.asarray(tmp)+0.0001)
    _min=np.min(tmp)
    _max=np.max(tmp)
    test_data['samples'][sample]['genes']=(np.asarray(tmp)-_min)/(_max-_min)
np.save('data/test_data_norm_log10.npy', test_data)
#print(test_data)