import pandas as pd
import numpy as np
initial_flag=True
#for cancertype in ['BLCA']:
for cancertype in 'ACC BLCA BRCA CESC CHOL COAD COADREAD DLBC ESCA FPPP GBM GBMLGG HNSC KICH KIPAN KIRC KIRP LAML LGG LIHC LUAD LUSC MESO OV PAAD PCPG PRAD READ SARC SKCM STAD STES TGCT THCA THYM UCEC UCS UVM'.split(' '):
    print(cancertype)
    try:
        clin=pd.read_csv('data/gdac.broadinstitute.org_{}.Merge_Clinical.Level_1.2016012800.0.0/{}.clin.merged.txt'.format(cancertype,cancertype),sep='\t')
        clin.index = clin.iloc[:,0]
        clin.columns = clin.iloc[list(clin.index).index('patient.bcr_patient_barcode')]
        rnaseq=pd.read_csv('data/gdac.broadinstitute.org_{}.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0/{}.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt'.format(cancertype,cancertype),sep='\t')
        rnaseq = rnaseq.iloc[1:,:]

        for patient_raw in rnaseq.columns:
            if 'TCGA' in patient_raw:
                print(patient_raw)
                patient = '-'.join(patient_raw.split('-')[0:3]).lower()
                patient_gene_df = rnaseq.loc[:,['Hybridization REF',patient_raw]]
                patient_gene_df.columns = ['features', patient_raw]

                if patient in clin.columns:
                    day2birth = clin.loc['patient.days_to_birth',patient]
                    day2death = clin.loc['patient.days_to_death',patient]
                    print(day2birth,day2death)
                    try:
                        try:
                            dayrecord=np.abs(int(day2birth))+int(day2death)
                        except:
                            dayrecord=np.abs(int(day2birth))
                        new_row = pd.DataFrame({'features':['day2birth','day2death','dayrecord','cancer_type'], patient_raw:[day2birth,day2death,dayrecord, cancertype]})
                        patient_gene_df = pd.concat([new_row, patient_gene_df[:]]).reset_index(drop = True)
                        #print(patient_gene_df)
                        if initial_flag==True:
                            ############# Keeping original sort Step1
                            #all_df = patient_gene_df.assign(source=range(0,len(patient_gene_df)))
                            ###############
                            all_df = patient_gene_df
                            initial_flag=False
                        elif initial_flag==False:
                            all_df = pd.merge(all_df,patient_gene_df, how='outer', on='features')
                            #################### Keeping original sort Step2
                            #all_df = all_df.sort_values(by='source', ascending=False, na_position='last')
                            #################################
                    except:
                        print('error')
                else:
                    print('{} with rnaseq but not in clin'.format(patient_raw))
    except:
        print('error')
#print(all_df)
########################### Keeping original sort Step3
#all_df = all_df.drop(columns='source')
#################################       
all_df.to_csv('data/data.tsv',sep='\t')
#all_df=pd.read_csv('data/data.tsv',sep='\t',index=False)
all_df=pd.read_csv('data/data.tsv',sep='\t',index_col=False)
#print(all_df['features'])
all_df=all_df.set_index('features')
#print(all_df.index)

##### According to all_df output, 'cancer_type''day2birth','day2death','dayrecord' are located in the row 20529 to 20532 (Note: the first row is row 0).
all_df_without_labels=all_df.drop(['cancer_type','day2birth','day2death','dayrecord'], axis=0)
all_data_dict={'features':list(all_df_without_labels.index),'samples':{}}

#all_data_dict={'features':list(all_df.index)[4:],'samples':{}}
for sample in all_df.columns:
    if 'TCGA' in sample:
        tmp=list(all_df.loc[:,sample])
        #day2birth=tmp[0]
        #day2death=tmp[1]
        #dayrecord=tmp[2]
        #cancertype=tmp[3]
        #genes=tmp[4:]
        #################
        day2birth=tmp[20530]
        day2death=tmp[20531]
        dayrecord=tmp[20532]
        cancertype=tmp[20529]
        #print(cancertype, day2birth, day2death, dayrecord)
        #print("\n")
        del tmp[20529:20533]
        genes=tmp
        all_data_dict['samples'][sample]={'day2birth':day2birth,'day2death':day2death,'dayrecord':dayrecord,'cancertype':cancertype,'genes':genes}   #####Note: it is a dictionary including more dictionaries inside.
np.save('data/data.npy', all_data_dict)

#### Note: numpy just save a dictorary with multi-levels. 
####       The fist-level dictionay have keys of 'features' and 'samples'. The value of the key "features" is a list of gene names. The value of the key "samples" is the another dictionary.
####       The second-level dictionary is the value of the key "samples". The key of this dictionary is the sample names. The value is third dictionary.
####       The third-level dictionary include keys of 'day2birth','day2death', 'dayrecord','cacertype','genes'. Among them, the value of the key "genes" is a list of gene expression values.   
data=np.load('data/data.npy',allow_pickle=True)
data=data.item()
for sample in data['samples']:
    tmp=[float(x) for x in data['samples'][sample]['genes']]
    tmp=np.log10(np.asarray(tmp)+0.0001)
    _min=np.min(tmp)
    _max=np.max(tmp)
    data['samples'][sample]['genes']=(np.asarray(tmp)-_min)/(_max-_min)
#np.save('../data/data_norm_log10.npy', data)
np.save('data/data_norm_log10.npy', data)
#print(data)