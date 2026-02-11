import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

############################## input real dataset #########################################
from configparser import ConfigParser
import sys
import os
config = ConfigParser()
config_file_path = './config/config.ini'
config.read(config_file_path)
#types = config.get('Data', 'types')
#types = config.get('Data', 'types30')
#types = config.get('Data', 'typesK')
#types = config.get('Data', 'typesG')
types = config.get('Data', 'typesC')

types_lst = [item.strip() for item in types.split(',')]

for index, item in enumerate(types_lst):
    print(f"{index}: {item}")

datafile = config.get('Data', 'data')
current_directory = os.getcwd()
realdata = np.load(os.path.join(current_directory,"data/{}.npy".format(datafile)),allow_pickle=True).item()
values_list=list(realdata.values())
#print(values_list)
values_len = len(values_list)
features=[]
label_idx=[]
xlabel=[]
for x in range(values_len):
    feature = values_list[x]['features']
    label = values_list[x]['type']
    
    #if label == "KICH" or label == "KIRC" or label == "KIRP" or label == "KIPAN":
    #if label == "GBMLGG" or label == "GBM" or label == "LGG":
    if label == "COADREAD" or label == "COAD" or label == "READ":
       features.append(feature)    
       label_idx.append(types_lst.index(label))
       xlabel.append(label)

#print(label_idx)
#print(features)
features_np = np.array(features)
# label_idx_np = np.array(label_idx)
# xlabel_np = np.array(xlabel)

################## plot ###############################################################
# Instantiate UMAP
reducer = umap.UMAP()
scaled_data = StandardScaler().fit_transform(features_np)
embedding = reducer.fit_transform(scaled_data)

# plt.scatter(embedding[:, 0], embedding[:, 1], c=label_idx)
# plt.colorbar(label='Label')
# plt.savefig("UMAP projection of the K-group data.png")

# color_map = {
    # 'KICH': "darkred",
    # 'KIRC': "darkgreen",
    # 'KIRP': "darkcyan",
    # 'KIPAN': "gold"
# }

# color_map = {
    # 'GBMLGG': "darkred",
    # 'GBM': "darkgreen",
    # 'LGG': "gold"
# }

color_map = {
    'COADREAD': "darkred",
    'COAD': "darkgreen",
    'READ': "gold"
}

colors = [color_map[cat] for cat in xlabel]
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
#create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10, label=label)
           for label in color_map]
plt.legend(handles=handles, title='Cancers')
#plt.title('UMAP projection of the K-group data', fontsize=12);
plt.savefig("UMAP projection of the C-group data.png")



