'''
The script is used to simplify the dictionary structure of the data file.

The structure of the original data (train_data_norm_log10_x.npy):
    {'samples': {
            'TCGA-A2-A0ST-01A-12R-A084-07': {'day2birth': '-22922', 'day2death': nan, 'dayrecord': '22922', 'cancertype': 'BRCA', 'genes': array([0.        , 0.46489265, 0.49866569, ..., 0.74204937, 0.63159535,0.38199453])},
            'TCGA-BR-6706-01A-11R-1884-13_y': {'day2birth': '-23053', 'day2death': nan, 'dayrecord': '23053', 'cancertype': 'STES', 'genes': array([0.42482528, 0.51957173, 0.52230517, ..., 0.68832371, 0.57021237,0.     ])},
            ......
       }    
            
     'feature_names': 'AGFG1|3267', 'AGFG2|3268', ......
    }


The structure of the simplified .npy file:
    {'TCGA-A2-A0ST-01A-12R-A084-07': {
              'type': 'BRCA', 'features': array([0.        , 0.46489265, 0.49866569, ..., 0.74204937, 0.63159535,0.38199453])
             },
     'TCGA-BR-6706-01A-11R-1884-13_y': {
              'type': 'STES', 'features': array([0.42482528, 0.51957173, 0.52230517, ..., 0.68832371, 0.57021237,0.     ])
             },
      .......
        
    }
'''


import numpy as np
x=np.load('train_data_norm_log10.npy', allow_pickle=True).item()
#x=np.load('test_data_norm_log10.npy', allow_pickle=True).item()
x_inner1=x['samples']

for key, value in x_inner1.items():
    #print("cacertype:"+value['cancertype'])
    value['type']=value.pop('cancertype')
    #print("type:"+value['type'])
    value['features']=value.pop('genes')
    value.pop('day2birth')
    value.pop('day2death')
    value.pop('dayrecord')
    
print(x_inner1)
np.save('train_data_norm_log10_simplified.npy', x_inner1, allow_pickle=True)
#np.save('test_data_norm_log10_simplified.npy', x_inner1, allow_pickle=True)

