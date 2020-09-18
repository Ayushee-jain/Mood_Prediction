#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r'C:\Users\LENOVO\Documents\Mood-Predictor\spotify.csv')


# In[3]:


data.head()


# In[4]:


data=data.drop(columns=['Unnamed: 0','album'])


# In[5]:


data.shape


# In[6]:


data.corr()


# In[7]:


data.isnull().sum()


# In[8]:


data.dtypes


# In[9]:


data.head()


# In[10]:


data['release_date']=data['release_date'].str.split('-')


# In[11]:


for i in range(len(data['release_date'])):
    if len(data['release_date'][i])==1:
        data['release_date'][i].append(0);
        data['release_date'][i].append(0);
    elif len(data['release_date'][i])==2:
        data['release_date'][i].append(0);


# In[12]:


print(data['release_date'][8][1])


# In[13]:


for i in range(0,len(data['release_date'])):
    data['release_year']=0;
for i in range(0,len(data['release_date'])):
    data['release_year'][i]=(data['release_date'][i][0])


# In[14]:


data.head()


# In[15]:


for i in range(0,len(data['release_date'])):
    data['release_month']=0;
for i in range(0,len(data['release_date'])):
    data['release_month'][i]=data['release_date'][i][1]


# In[16]:


for i in range(0,len(data['release_date'])):
    data['release_day']=0;
for i in range(0,len(data['release_date'])):
    data['release_day'][i]=data['release_date'][i][2]


# In[17]:


data.head()


# In[18]:


data=data.drop(columns=['release_date'])


# In[19]:


data.dtypes


# In[20]:


data.describe()


# In[21]:


columns=data.columns


# In[22]:


print(columns)


# In[23]:


for i in columns:
    print("{} : {}".format(i,len(set(data[i]))))


# In[25]:


data=data.drop(columns=['time_signature','length','tempo','release_month','release_year','release_day'])


# In[26]:


data.head()


# In[27]:


from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()
x=data[['loudness']].values
loudness=mn.fit_transform(x)
data['loudness']=pd.DataFrame(loudness)


# In[28]:


mn=MinMaxScaler()
x=data[['popularity']].values
popularity=mn.fit_transform(x)
data['popularity']=pd.DataFrame(popularity)


# In[29]:


data.shape


# In[31]:


sns.distplot(data['popularity'])


# In[33]:


sns.distplot(data['danceability'])


# In[34]:


sns.distplot(data['acousticness'])


# In[35]:


data=data.drop(columns=['danceability.1'])


# In[36]:


sns.distplot(data['energy'])


# In[37]:


x=data.copy()


# In[38]:


x=x.drop(columns=['name','artist'])


# In[40]:


x.shape


# In[43]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,9):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,9),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')


# In[44]:


clusterer=KMeans(n_clusters=4,random_state=10)
cluster_labels=clusterer.fit_predict(x)
print(cluster_labels)


# In[45]:


from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.cm as cm
range_n_clusters=[2,3,4,5,6]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(7, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(x) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(x)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(x, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(x, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


# In[47]:


from sklearn.decomposition import PCA
kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)


# In[48]:


x.shape


# In[49]:


principal_components.shape


# In[51]:


plt.scatter(principal_components[:,0],principal_components[:,1],c=y_kmeans,cmap='plasma')


# In[53]:


from sklearn.manifold import TSNE 

#T-SNE with two dimensions
tsne = TSNE(n_components=2, perplexity=50)

tsne_components = tsne.fit_transform(x)


# In[54]:


plt.scatter(tsne_components[:,0],tsne_components[:,1],c=y_kmeans,cmap='plasma')


# In[55]:


pca.explained_variance_ratio_


# In[56]:


data['label']=y_kmeans
data=data.sample(frac=1)
data['label'].value_counts()


# In[57]:


data[data['label']==0].tail(10)


# In[58]:


data[data['label'] == 1].tail(10)


# In[59]:


data[data['label'] == 2].tail(10)


# In[61]:


data[data['label']==3].tail(10)


# In[60]:


#Energetic Mood
data[data['label']==0].hist()


# In[62]:


#Cheerfull Mood
data[data['label']==1].hist()


# In[63]:


#Chill mood
data[data['label']==2].hist()


# In[64]:


#Romantic Mood 
data[data['label']==3].hist()


# In[65]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

X = x
y = y_kmeans

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
rfc.fit(X_train,y_train)


# In[66]:


y_pred = rfc.predict(X_test)


# In[67]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[68]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# In[69]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[72]:


from sklearn import svm

#Create a SVM Classifier
svm = svm.SVC(kernel="linear") 

#Train the model using the training sets
svm.fit(X_train, y_train)

#Predict the response for test dataset
svm_pred = svm.predict(X_test)


# In[73]:


print(classification_report(y_test,svm_pred))


# In[ ]:




