# Recommender-by-Python


path ="E:\\gaurav_back up\\Summer 2018\\recommender\\ml-100k\\u.data"


# In[17]:


import pandas as pd
df =pd.read_csv(path, sep='\t')


# In[14]:


type(df)


# In[15]:


df.head()


# In[18]:


df.columns


# In[20]:


df.shape


# Data exploration

# In[26]:


# distribution of rating
import matplotlib.pyplot as plt
#%matplotlib inline
df.columns =['UserId','MovieID','Rating','TimeStamp']
plt.hist(df['Rating'])
plt.show()


# In[35]:


plt.hist(df.groupby(['MovieID'])['MovieID'].count())


# In[47]:


#numpy capabilities of such as arrays and row iterations in matrix in a code
# We are buliding user based cf. So extract all unique user IDs and then we chaeck the length using shape parameter
n_users=df.UserId.unique().shape[0]  # columns
# we aee buliding a matrix of userID and Item ID
n_items=df['MovieID'].unique().shape[0]
print(str(n_users) + ' users')
print(str(n_items) + ' movies')


# In[64]:


# create a zeroes matrix 
import numpy as np
ratings =np.zeros((n_users,n_items))

for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]
#     print(row[1]-1)
#     print(row[2]-1)
#     print(row[3])
type(ratings)
    


# In[65]:


ratings.shape


# In[66]:


ratings
# rating matrix is sparse


# In[67]:


sparsity =float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print('Sparsity: {:4.2f}%'.format(sparsity))


# In[70]:


# now didvide the set using sklearn
from sklearn.cross_validation import train_test_split

ratings_train, ratings_test= train_test_split(ratings, test_size=0.33,random_state=42)
# we always get the same output if we use random_state=42


# In[71]:


ratings_test.shape


# In[72]:


ratings_train.shape


# In[78]:


import numpy as np
import sklearn

dist_out = 1 -sklearn.metrics.pairwise.cosine_distances(ratings_train)
type(dist_out)
dist_out.shape



# In[79]:


dist_out


# As previously mentioned the unknown values can be calculated for all uses by taking the dot product
# 
# between the distance matrix and the rating matrix and then normalizing the data with the number of ratings.
# 
# So we do that now that we've predicted the unknown ratings for use in a training set.

# In[81]:


user_pred = dist_out.dot(ratings_train)/np.array([np.abs(dist_out).sum(axis=1)]).T


# In[83]:


from sklearn.metrics import mean_squared_error
def get_mse(pred,actual):
    pred=pred[actual.nonzero()].flatten()
    actual=actual[actual.nonzero()].flatten()
    return mean_squared_error(pred,actual)
get_mse(user_pred,ratings_train)


# In[84]:


get_mse(user_pred,ratings_test)


# In[116]:


# User based collaborative filtering with the k-nearest Neighbbour
k=5
from sklearn.neighbors import NearestNeighbors
neigh=NearestNeighbors(k,'cosine')
neigh.fit(ratings_train)
top_k_distances,top_k_users = neigh.kneighbors(ratings_train,return_distance =True)


# In[117]:


top_k_distances.shape


# In[90]:


top_k_users[0]


# In[118]:


user_pred_k= np.zeros(ratings_train.shape)
for i in range(ratings_train.shape[0]):
    user_pred_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_users][i])/np.array([np.abs(top_k_distances[i])])
    
user_pred_k    
    


# In[99]:


# whetehr  model is i proved or not
get_mse(user_pred_k,ratings_train)


# In[100]:


get_mse(user_pred_k,ratings_test)


# In[102]:


# item based recommendation system

# Step 1: calculate the similarities between movies
k= ratings_train.shape[1]
neigh = NearestNeighbors(k,'cosine')



# In[103]:


neigh.fit(ratings_train.T)


# In[104]:


top_k_distance,top_k_users = neigh.kneighbors(ratings_train.T, return_distance=True)
top_k_distance.shape


# In[106]:


item_pred =ratings_train.dot(top_k_distance)/ np.array([np.abs(top_k_distance).sum(axis=1)])
item_pred.shape


# In[107]:


item_pred


# In[108]:


get_mse(item_pred, ratings_train)


# In[109]:


get_mse(item_pred, ratings_test)


# In[111]:


k=1
neigh2=NearestNeighbors(k,'cosine')
neigh2.fit(ratings_train.T)
top_k_distance,top_k_movies = neigh2.kneighbors(ratings_train.T,return_distance =True)


# In[119]:


pred= np.zeros(ratings_train.T.shape)
for i in range(ratings_train.T.shape[0]):
    pred[i,:] = top_k_distances[i].dot(ratings_train.T[top_k_users][i])/np.array(top_k_distances[i])


# In[120]:


get_mse(item_pred,ratings_test)

