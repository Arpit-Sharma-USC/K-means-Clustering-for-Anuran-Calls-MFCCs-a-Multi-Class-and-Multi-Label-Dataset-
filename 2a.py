import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


dataframe = pd.read_csv("/media/ghost/Games And Images/INF552/hw4/Anuran Calls (MFCCs)/Frogs_MFCCs.csv")

dataframe.drop('RecordID',axis=1,inplace=True)
dataframe.replace('Bufonidae',int(1),inplace=True)
dataframe.replace('Dendrobatidae',int(2),inplace=True)
dataframe.replace('Hylidae',int(3),inplace=True)
dataframe.replace('Leptodactylidae',int(4),inplace=True)

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


dataframe.to_csv('/media/ghost/Games And Images/INF552/hw4/Frogs_MFCCs-changed.csv',index=False)

dataset=dataframe.values
shuf_ds = shuffle(dataset, random_state=0)

X=shuf_ds[:,:22]
Y=shuf_ds[:,22:]


# elbow method to determine k(number of clusters) for k-means
distortions = []
K = range(1, 16)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
print('Distortions:',distortions)
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

diff=[]
for i in range(0,len(distortions)-1):
    diff.append(np.abs(distortions[i+1]-distortions[i]))

index=0
temp = diff[0]

for i in range(0,len(diff)):

    if diff[i]>temp:
        temp=diff[i]
        index=i
print('Elbow at k:',index+2)
