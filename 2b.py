import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import random

random_state = np.random.RandomState(random.randint(0, 100))

dataframe = pd.read_csv("/media/ghost/Games And Images/INF552/hw4/Anuran Calls (MFCCs)/Frogs_MFCCs.csv")

dataframe.drop('RecordID',axis=1,inplace=True)
dataframe.replace('Bufonidae',int(1),inplace=True)
dataframe.replace('Dendrobatidae',int(2),inplace=True)
dataframe.replace('Hylidae',int(3),inplace=True)
dataframe.replace('Leptodactylidae',int(4),inplace=True)
# Adenomera
# Ameerega
# Dendropsophus
# Hypsiboas
# Leptodactylus
# Osteocephalus
# Rhinella
# Scinax
dataframe.replace('Adenomera',int(10),inplace=True)
dataframe.replace('Ameerega',int(11),inplace=True)
dataframe.replace('Dendropsophus',int(12),inplace=True)
dataframe.replace('Hypsiboas',int(13),inplace=True)
dataframe.replace('Leptodactylus',int(14),inplace=True)
dataframe.replace('Osteocephalus',int(15),inplace=True)
dataframe.replace('Rhinella',int(16),inplace=True)
dataframe.replace('Scinax',int(17),inplace=True)

dataframe.replace('AdenomeraAndre',int(100),inplace=True)
dataframe.replace('AdenomeraHylaedactylus',int(101),inplace=True)
dataframe.replace('Ameeregatrivittata',int(102),inplace=True)
dataframe.replace('HylaMinuta',int(103),inplace=True)
dataframe.replace('HypsiboasCinerascens',int(104),inplace=True)
dataframe.replace('HypsiboasCordobae',int(105),inplace=True)
dataframe.replace('LeptodactylusFuscus',int(106),inplace=True)
dataframe.replace('OsteocephalusOophagus',int(107),inplace=True)
dataframe.replace('Rhinellagranulosa',int(108),inplace=True)
dataframe.replace('ScinaxRuber',int(109),inplace=True)
# AdenomeraAndre
# AdenomeraHylaedactylus
# Ameeregatrivittata
# HylaMinuta
# HypsiboasCinerascens
# HypsiboasCordobae
# LeptodactylusFuscus
# OsteocephalusOophagus
# Rhinellagranulosa
# ScinaxRuber

dataframe.to_csv('/media/ghost/Games And Images/INF552/hw4/Frogs_MFCCs-changed.csv',index=False)

dataset=dataframe.values
shuf_ds = shuffle(dataset, random_state=0)

X=shuf_ds[:,:22]
Y=shuf_ds[:,22:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

Y_family=Y[:,:1]
Y_genus=Y[:,1:2]
Y_species=Y[:,2:]

# k is determined by elbow method
k=2

print("\n#######   k-Means Clustering for Anuran Calls   #######")
kmeans=KMeans(n_clusters=k,n_init=20,random_state=random_state)
print('\nFitting...')
kmeans.fit(X)

centers=kmeans.cluster_centers_
inertia=kmeans.inertia_
labels=kmeans.labels_

print('\nCenters of Centroids:')
for i in range(0,len(centers)):
    print('\n\tCluster-',i,':',centers[i])


#############################################family
print("\nClustering for label-Family...\n")

label_0_counter=0
label_1_counter=0

family_Bufonidae_c0=0
family_Dendrobatidae_c0=0
family_Hylidae_c0=0
family_Leptodactylidae_c0=0

family_Bufonidae_c1=0
family_Dendrobatidae_c1=0
family_Hylidae_c1=0
family_Leptodactylidae_c1=0


for i in range(0,len(labels)):
    temp = int(Y_family[i])
    if int(labels[i])==0:

        label_0_counter+=1

        if temp==1:
            family_Bufonidae_c0+=1
        elif temp==2:
            family_Dendrobatidae_c0+=1
        elif temp==3:
            family_Hylidae_c0+=1
        elif temp==4:
            family_Leptodactylidae_c0+=1
    else:

        label_1_counter += 1

        if temp == 1:
            family_Bufonidae_c1 += 1
        elif temp == 2:
            family_Dendrobatidae_c1 += 1
        elif temp == 3:
            family_Hylidae_c1 += 1
        elif temp == 4:
            family_Leptodactylidae_c1 += 1

family_cluster_0_dict = {'1': family_Bufonidae_c0, '2': family_Dendrobatidae_c0,'3':family_Hylidae_c0,'4':family_Leptodactylidae_c0}
family_cluster_1_dict = {'1': family_Bufonidae_c1, '2': family_Dendrobatidae_c1,'3':family_Hylidae_c1,'4':family_Leptodactylidae_c1}


c_0_family_label=int(max(family_cluster_0_dict.keys(), key=(lambda k: family_cluster_0_dict[k])))
c_1_family_label=int(max(family_cluster_1_dict.keys(), key=(lambda k: family_cluster_1_dict[k])))

temp_preds_family=np.arange(0,len(labels),1)

for i in range(0,len(labels)):
    if labels[i]==0:
        temp_preds_family[i]=c_0_family_label
    else:
        temp_preds_family[i]=c_1_family_label

majority_family={1: 'Bufonidae', 2: 'Dendrobatidae',3:'Hylidae',4:'Leptodactylidae'}

print('\tCluster-0-label by Majority Polling:',majority_family[c_0_family_label])
print('\tCluster-1-label by Majority Polling:',majority_family[c_1_family_label])


############################################genus

print("\nClustering for class-Genus...\n")

label_0_counter=0
label_1_counter=0


genus_Adenomera_c0=0
genus_Ameerega_c0=0
genus_Dendropsophus_c0=0
genus_Hypsiboas_c0=0
genus_Leptodactylus_c0=0
genus_Osteocephalus_c0=0
genus_Rhinella_c0=0
genus_Scinax_c0=0


genus_Adenomera_c1=0
genus_Ameerega_c1=0
genus_Dendropsophus_c1=0
genus_Hypsiboas_c1=0
genus_Leptodactylus_c1=0
genus_Osteocephalus_c1=0
genus_Rhinella_c1=0
genus_Scinax_c1=0

for i in range(0,len(labels)):
    temp = int(Y_genus[i])
    if int(labels[i])==0:

        label_0_counter+=1

        if temp==10:
            genus_Adenomera_c0+=1
        elif temp==11:
            genus_Ameerega_c0+=1
        elif temp==12:
            genus_Dendropsophus_c0+=1
        elif temp==13:
            genus_Hypsiboas_c0+=1
        elif temp==14:
            genus_Leptodactylus_c0+=1
        elif temp==15:
            genus_Osteocephalus_c0+=1
        elif temp==16:
            genus_Rhinella_c0+=1
        elif temp==17:
            genus_Scinax_c0+=1
    else:

        label_1_counter += 1

        if temp==10:
            genus_Adenomera_c1+=1
        elif temp==11:
            genus_Ameerega_c1+=1
        elif temp==12:
            genus_Dendropsophus_c1+=1
        elif temp==13:
            genus_Hypsiboas_c1+=1
        elif temp==14:
            genus_Leptodactylus_c1+=1
        elif temp==15:
            genus_Osteocephalus_c1+=1
        elif temp==16:
            genus_Rhinella_c1+=1
        elif temp==17:
            genus_Scinax_c1+=1

genus_cluster_0_dict = {'10': genus_Adenomera_c0, '11': genus_Ameerega_c0,'12':genus_Dendropsophus_c0,'13':genus_Hypsiboas_c0,'14':genus_Leptodactylus_c0,'15':genus_Osteocephalus_c0,'16':genus_Rhinella_c0,'17':genus_Scinax_c0}
genus_cluster_1_dict = {'10': genus_Adenomera_c1, '11': genus_Ameerega_c1,'12':genus_Dendropsophus_c1,'13':genus_Hypsiboas_c1,'14':genus_Leptodactylus_c1,'15':genus_Osteocephalus_c1,'16':genus_Rhinella_c1,'17':genus_Scinax_c1}

c_0_genus_label=int(max(genus_cluster_0_dict.keys(), key=(lambda k: genus_cluster_0_dict[k])))
c_1_genus_label=int(max(genus_cluster_1_dict.keys(), key=(lambda k: genus_cluster_1_dict[k])))

temp_preds_genus=np.arange(0,len(labels),1)

for i in range(0,len(labels)):
    if labels[i]==0:
        temp_preds_genus[i]=c_0_genus_label
    else:
        temp_preds_genus[i]=c_1_genus_label

majority_genus={10:'Adenomera',11:'Ameerega',12:'Dendropsophus',13:'Hypsiboas',14:'Leptodactylus',15:'Osteocephalus',16:'Rhinella',17:'Scinax'}

print('\tGenus Cluster-0-label by Majority Polling:',majority_genus[c_0_genus_label])
print('\tGenus Cluster-1-label by Majority Polling:',majority_genus[c_1_genus_label])

########################################species

print("\nClustering for class-Species...\n")


label_0_counter=0
label_1_counter=0

species_AdenomeraAndre_c0=0
species_AdenomeraHylaedactylus_c0=0
species_Ameeregatrivittata_c0=0
species_HylaMinuta_c0=0
species_HypsiboasCinerascens_c0=0
species_HypsiboasCordobae_c0=0
species_LeptodactylusFuscus_c0=0
species_OsteocephalusOophagus_c0=0
species_Rhinellagranulosa_c0=0
species_ScinaxRuber_c0=0

species_AdenomeraAndre_c1=0
species_AdenomeraHylaedactylus_c1=0
species_Ameeregatrivittata_c1=0
species_HylaMinuta_c1=0
species_HypsiboasCinerascens_c1=0
species_HypsiboasCordobae_c1=0
species_LeptodactylusFuscus_c1=0
species_OsteocephalusOophagus_c1=0
species_Rhinellagranulosa_c1=0
species_ScinaxRuber_c1=0

# AdenomeraAndre
# AdenomeraHylaedactylus
# Ameeregatrivittata
# HylaMinuta
# HypsiboasCinerascens
# HypsiboasCordobae
# LeptodactylusFuscus
# OsteocephalusOophagus
# Rhinellagranulosa
# ScinaxRuber
for i in range(0,len(labels)):
    temp = int(Y_species[i])
    if int(labels[i])==0:

        label_0_counter+=1

        if temp==100:
            species_AdenomeraAndre_c0+=1
        elif temp==101:
            species_AdenomeraHylaedactylus_c0+=1
        elif temp==102:
            species_Ameeregatrivittata_c0+=1
        elif temp==103:
            species_HylaMinuta_c0+=1
        elif temp==104:
            species_HypsiboasCinerascens_c0+=1
        elif temp==105:
            species_HypsiboasCordobae_c0+=1
        elif temp==106:
            species_LeptodactylusFuscus_c0+=1
        elif temp==107:
            species_OsteocephalusOophagus_c0+=1
        elif temp==108:
            species_Rhinellagranulosa_c0+=1
        elif temp==109:
            species_ScinaxRuber_c0+=1


    else:

        label_1_counter += 1

        if temp == 100:
            species_AdenomeraAndre_c1 += 1
        elif temp == 101:
            species_AdenomeraHylaedactylus_c1 += 1
        elif temp == 102:
            species_Ameeregatrivittata_c1 += 1
        elif temp == 103:
            species_HylaMinuta_c1 += 1
        elif temp == 104:
            species_HypsiboasCinerascens_c1 += 1
        elif temp == 105:
            species_HypsiboasCordobae_c1 += 1
        elif temp == 106:
            species_LeptodactylusFuscus_c1 += 1
        elif temp == 107:
            species_OsteocephalusOophagus_c1 += 1
        elif temp == 108:
            species_Rhinellagranulosa_c1 += 1
        elif temp == 109:
            species_ScinaxRuber_c1 += 1


species_cluster_0_dict = {'100': species_AdenomeraAndre_c0, '101': species_AdenomeraHylaedactylus_c0,'102':species_Ameeregatrivittata_c0,'103':species_HylaMinuta_c0,'104':species_HypsiboasCinerascens_c0,'105': species_HypsiboasCordobae_c0,'106':species_LeptodactylusFuscus_c0,'107':species_OsteocephalusOophagus_c0,'108':species_Rhinellagranulosa_c0,'109':species_ScinaxRuber_c0}
species_cluster_1_dict = {'100': species_AdenomeraAndre_c1, '101': species_AdenomeraHylaedactylus_c1,'102':species_Ameeregatrivittata_c1,'103':species_HylaMinuta_c1,'104':species_HypsiboasCinerascens_c1,'105': species_HypsiboasCordobae_c1,'106':species_LeptodactylusFuscus_c1,'107':species_OsteocephalusOophagus_c1,'108':species_Rhinellagranulosa_c1,'109':species_ScinaxRuber_c1}

c_0_species_label=int(max(species_cluster_0_dict.keys(), key=(lambda k: species_cluster_0_dict[k])))
c_1_species_label=int(max(species_cluster_1_dict.keys(), key=(lambda k: species_cluster_1_dict[k])))

temp_preds_species=np.arange(0,len(labels),1)

for i in range(0,len(labels)):
    if labels[i]==0:
        temp_preds_species[i]=c_0_species_label
    else:
        temp_preds_species[i]=c_1_species_label


majority_species={100:'AdenomeraAndre',101:'AdenomeraHylaedactylus',102:'Ameeregatrivittata',103:'HylaMinuta',104:'HypsiboasCinerascens',105:'HypsiboasCordobae',106:'LeptodactylusFuscus',107:'OsteocephalusOophagus',108:'Rhinellagranulosa',109:'ScinaxRuber'}

print('\tSpecies Cluster-0-label by Majority Polling:',majority_species[c_0_species_label])
print('\tSpecies Cluster-1-label by Majority Polling:',majority_species[c_1_species_label])


print('\nEnd of Program')