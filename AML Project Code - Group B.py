#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


ratings = pd.read_csv("D:\Python\BX-Book-Ratings.csv",sep=';',error_bad_lines=False,encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'rating']
books = pd.read_csv("D:\Python\BX-Books.csv",sep=';',error_bad_lines=False,encoding="latin-1",index_col=0)
users = pd.read_csv("D:\Python\BX-Users.csv",sep=';',error_bad_lines=False,encoding="latin-1")
users.columns = ['userID','location','age']
books = books.drop(columns =["Image-URL-S","Image-URL-M","Image-URL-L"])
books.columns = ['title','author','year','publisher']


# In[367]:


books.head()


# In[2]:


print(ratings.shape)
print(list(ratings.columns))
print(ratings.head(),'\n\n')
print(books.head(),'\n\n')
print(users.head(),'\n\n')#show info

plt.rc("font", size=15) 
ratings.rating.value_counts(sort=False).plot(kind='bar',color = 'blue') 
plt.title('Rating Distribution\n') 
plt.xlabel('Rating') 
plt.ylabel('Count') 
plt.show()

users.age.hist(bins=[0, 10, 20, 30, 40, 50, 100],color = 'blue')
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

print(ratings.isnull().any())


# In[5]:


rating_count = pd.DataFrame(ratings.groupby('ISBN')['rating'].count())#show the most popular books
rating_count.sort_values('rating', ascending=False, inplace=False).head(10)


# In[6]:


#the average rating, and the number of ratings each book received
average_rating = pd.DataFrame(ratings.groupby('ISBN')['rating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['rating'].count())
average_rating.sort_values('ratingCount', ascending=False).head()


# In[7]:


#First dataset: exclude users with less than 200 ratings, and books with less than 100 ratings

user_rating_count = ratings['userID'].value_counts()
ratingsCleaned_1 = ratings[ratings['userID'].isin(user_rating_count[user_rating_count >= 200].index)]
book_rating_count = ratings['rating'].value_counts()
ratingsCleaned_1 = ratingsCleaned_1[ratingsCleaned_1['rating'].isin(book_rating_count[book_rating_count >= 100].index)]


print(ratingsCleaned_1.head())
print('Shape of cleaned ratings: ',ratingsCleaned_1.shape)


# PearsonR method

# In[8]:


#convert the ratings table to a 2D matrix
def recByPearsonR(item_id):
    #item_id = '0316666343'
    ratings_pivot = ratingsCleaned_1.pivot(index='userID', columns='ISBN').rating
    userID = ratings_pivot.index
    ISBN = ratings_pivot.columns

    item = ratings_pivot[item_id]
    similar_item = ratings_pivot.corrwith(item)#compare the rating
    corr_item = pd.DataFrame(similar_item, columns=['pearsonR'])
    corr_item.dropna(inplace=True)
    corr_summary = corr_item.join(average_rating['ratingCount'])
    result_Pearson = corr_summary[corr_summary['ratingCount']>=300].sort_values('pearsonR', ascending=False)
    for i in range(1,6):
        print(f"{i}: {books.loc[result_Pearson.index[i],'title']}")


# In[ ]:





# In[9]:


#Second dataset: 

combine_book_rating = pd.merge(ratings, books, on='ISBN') #combine book info with ratings

print(combine_book_rating.head())
print(combine_book_rating.shape)

combine_book_rating.drop(columns = ['year','publisher'],axis=1,inplace = True)#drop useless data
 
print('\n',combine_book_rating.head())

print('\n',combine_book_rating.isnull().any())#the author got some empty data

combine_book_rating.dropna(axis = 0, subset = ['author'])#drop empty data

# group by book titles and create a new column for total rating count.
book_rating_count = (combine_book_rating.groupby(by = ['title'])['rating'].count().reset_index().rename(columns
                    = {'rating':'totalRatingCount'})[['title','totalRatingCount']])

print('\n',book_rating_count.head())

# combine the rating data with the total rating count data
rating_with_totalRatingCount = combine_book_rating.merge(book_rating_count, left_on = 'title', right_on = 'title', how = 'left')

print('\n',rating_with_totalRatingCount.head())

pd.set_option('display.float_format', lambda x: '%.3f' % x)

print('\n',book_rating_count['totalRatingCount'].describe())

print('\n',book_rating_count['totalRatingCount'].quantile(np.arange(.9, 1, .01)))#show the percentage of rating numbers

popularity_standard = 50 #choose the data with 50 or more ratings
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_standard')

print(rating_popular_book.shape)
print(rating_popular_book.head())


# In[10]:


rating_popular_book.drop(columns = ['author'],inplace = True)#drop useless data
rating_popular_book.head(10)


# In[11]:


#build a 2D matrix, and index is ISBN
rating_popular_book = rating_popular_book.drop_duplicates(['userID', 'title'])#drop the same value
rating_popular_book_pivot = rating_popular_book.pivot(index = 'ISBN', columns = 'userID', values = 'rating').fillna(0)#biuld a matrix
userID = rating_popular_book_pivot.columns
ISBN = rating_popular_book_pivot.index


# In[17]:


rating_popular_book_pivot.head()


# In[11]:


model_knn_item = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn_item.fit(rating_popular_book_pivot)# use knn to fit the model


# item-item method

# In[19]:


#item-item method
def recByBook(item_id):
    if(item_id not in rating_popular_book_pivot.index):#if the ISBN not exist, give an erro
        print("Please enter a valid ISBN as shown:\n\n {} ".format(re.sub('[\[\]]', '', np.array_str(rating_popular_book_pivot.index.values))))
    else:
        loc = rating_popular_book_pivot.index.get_loc(item_id)#find the location of input ISBN
        distances, indices = model_knn_item.kneighbors(rating_popular_book_pivot.iloc[loc, :].values.reshape(1, -1), n_neighbors =6)#use the model to predict
        similarities = 1 - distances #change distance to similarity
       
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for "{0}":\n'.format(books.loc[item_id,'title'])) #show the book title we used
            elif(similarities.flatten()[i]==0):
                print(f"{i}. {books.loc[average_rating[average_rating['ratingCount']>=300].sort_values('rating', ascending=False).head(5).index[i],'title']}")
                # some ISBN can't find the similar book. So we recommend books with high ratings.
                break
                
            else:
                print(f"{i}: {books.loc[rating_popular_book_pivot.index[indices.flatten()[i]],'title']}")#show the similar books


# user-user metod

# In[16]:


T_rating_popular_book_pivot = rating_popular_book_pivot.T #use the transpose of matrix
model_knn_user = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn_user.fit(T_rating_popular_book_pivot)# use knn to fit the model


# In[12]:


rating_popular_book_pivot.head()


# In[13]:


rating_popular_book_pivot.shape


# In[14]:


import sklearn
from sklearn.decomposition import TruncatedSVD

SVD = TruncatedSVD(n_components=12, random_state=17)
matrixSVD = SVD.fit_transform(rating_popular_book_pivot)
matrixSVD.shape


# In[17]:


popular_book_title = T_rating_popular_book_pivot.columns
popular_book_list = list(popular_book_title)
coffey_hands = popular_book_list.index("0002558122")
print(coffey_hands)


# In[13]:


corr_coffey_hands  = corr[coffey_hands]
#list(popular_book_title[(corr_coffey_hands0.9)])
corr_coffey_hands


# In[14]:


def recByUser(user_id):

    if(user_id not in T_rating_popular_book_pivot.index):#if the ISBN not exist, give an erro
        print("Please enter a valid user ID as shown:\n\n {} ".format(re.sub('[\[\]]', '', np.array_str(T_rating_popular_book_pivot.index.values))))

    else:
        loc = T_rating_popular_book_pivot.index.get_loc(user_id)#find the location of input ID
        distances, indices = model_knn_user.kneighbors(T_rating_popular_book_pivot.iloc[loc, :].values.reshape(1, -1), n_neighbors =6)#use the model to predict
        similarities = 1 - distances #change distance to similarity

        if (np.all(distances == 1)):
            print('Dont have enough information. So we recommend most popular books.\n')
            for i in range(0,5):
                print(f"{i}. {books.loc[average_rating[average_rating['ratingCount']>=300].sort_values('rating', ascending=False).head(5).index[i],'title']}")
                #recommend popular books

        else:
            similar_user_book = pd.DataFrame()
            for i in range(0, len(distances.flatten())):#create a dataframe by all books the similar user have read
                similar_user_book=similar_user_book.append(ratings[ratings['userID'] == T_rating_popular_book_pivot.index[indices.flatten()[i]]]) 
            average_similar_rating = pd.DataFrame(similar_user_book.groupby('ISBN')['rating'].mean())
            print(f"Recommendations for user {user_id}:")#compare the books and recommend the high rating ones
            for i in range(0,5):
                print(f"{i+1}. {books.loc[average_similar_rating.sort_values('rating', ascending=False).head().index[i],'title']}")

    
    
    


# you can use ISBN:0316666343 and user id: 278854 for test

# In[18]:


#this is the control part. you need to enter the ISBN number to see if there's  a recommendation or not.
flag = True
np.set_printoptions(threshold=np.inf)#make sure it can print full list if input is wrong.

while(flag == True):
    
    item_id = input("\n\ninput the ISBN:(enter q to quit.)")#input the ISBN
    if(item_id != 'q'):
        print("\nitem-item method:\n")
        recByBook(item_id )
        print("\npearsonR method:\n")
        recByPearsonR(item_id)
        user_id = int(input("\n\nnow input the user ID:"))
        print("\nuser-user method:\n")
        recByUser(user_id)
        
    else:
        flag = False

    
    

