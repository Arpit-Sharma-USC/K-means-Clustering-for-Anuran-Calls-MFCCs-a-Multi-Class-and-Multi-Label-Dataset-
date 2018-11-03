# K-means-Clustering for Anuran Calls(MFCCs; a-Multi-Class-and-Multi-Label-Dataset)

Classification of datapoint into the correct family, species and genus using the k_means clustering approach. 

# Dataset

https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29#

# Procedure

•	Determination of the optimal k-value 

o	To determine the number of clusters to be used in the clustering process, I used a technique well-known as Elbow method. 

o	This method looks at the percentage of variance explained as a function of the number of clusters: One should choose several clusters so that adding another cluster doesn't give much better modeling of the data. More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, the first clusters will add much information (explain a lot of variance), but at some point, the marginal gain will drop, giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion". This "elbow" cannot always be unambiguously identified.

•	Performing k-means Clustering 

o	After determining the value of k to be 2 for our problem, I proceeded by running the k-means clustering for the dataset. Post-execution I was able to cluster the data into 2 different clusters, namely Cluster-0 and Cluster-1.

o	I then determined the label-family for each cluster by reading the true labels of the data-points in the corresponding cluster and then took a majority poll.

o	In a similar fashion I was able to perform this task of determining the label-genus and label-species for each cluster.

•	Hamming Loss of the Majority Triplet

o	As part of analysis I decided to calculate the hamming loss of each Cluster's family, genus and species.
