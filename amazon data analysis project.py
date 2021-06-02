#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import nltk
import string

The column or features in the dataset:
Id
ProductId — unique identifier for the product
UserId — unqiue identifier for the user
ProfileName
HelpfulnessNumerator — number of users who found the review helpful
HelpfulnessDenominator — number of users who indicated whether they found the review helpful or not
Score — rating between 1 and 5
Time — timestamp for the review
Summary — brief summary of the review
Text — text of the review
# In[4]:


# Create a SQL connection to our SQLite database
con = sqlite3.connect('D:\ISLAM\Projects\Amazon Data Analysis/database.sqlite')


# In[5]:


type(con)


# #### reading data from Sqlite database

# In[6]:


pd.read_sql_query("SELECT * FROM Reviews", con)


# In[7]:


df=pd.read_sql_query("SELECT * FROM Reviews", con)


# #### reading some n number of rows, use LIMIT over ther

# In[8]:


pd.read_sql_query("SELECT * FROM Reviews LIMIT 3", con)


# #### or we can also Load the dataset using pandas

# In[9]:


df = pd.read_csv('D:\ISLAM\Projects\Amazon Data Analysis/Reviews.csv')

print(df.shape)
df.head()


# In[10]:


df.shape


# ### What is sentiment analysis?
#     Sentiment analysis is the computational task of automatically determining what feelings a writer is expressing in text
#     Some examples of applications for sentiment analysis include:
# 
#     1.Analyzing the social media discussion around a certain topic
#     2.Evaluating survey responses
#     3.Determining whether product reviews are positive or negative
# 
#     Sentiment analysis is not perfect.It also cannot tell you why a writer is feeling a certain way. However, it can be useful to quickly summarize some qualities of text, especially if you have so much text that a human reader cannot analyze it.For this project,the goal is to to classify Food reviews based on customers' text.

# In[11]:


from textblob import TextBlob


# In[12]:


TextBlob(df['Summary'][0]).sentiment.polarity


# In[13]:


## takes 3 mins 
polarity=[] # list which will contain the polarity of the comments

for i in df['Summary']:
    try:
        polarity.append(TextBlob(i).sentiment.polarity)   
    except:
        polarity.append(0)


# In[14]:


len(polarity)


# In[15]:


data=df.copy()


# In[16]:


data['polarity']=polarity


# In[17]:


data.head()


# In[18]:


data['polarity'].nunique()


# ### Lets perform EDA for the Positve sentences¶

# In[19]:


data_positive = data[data['polarity']>0]
data_positive.shape


# In[20]:


from wordcloud import WordCloud, STOPWORDS


# In[21]:


stopwords=set(STOPWORDS)


# In[22]:


positive=data_positive[0:200000]


# In[23]:


total_text= (' '.join(data_positive['Summary']))


# In[24]:


len(total_text)


# In[25]:


total_text[0:10000]


# In[26]:


import re
total_text=re.sub('[^a-zA-Z]',' ',total_text)


# In[27]:


total_text


# In[28]:


## remove extra spaces
total_text=re.sub(' +',' ',total_text)


# In[29]:


total_text[0:10000]


# In[30]:


len(total_text)


# In[31]:


wordcloud = WordCloud(width = 1000, height = 500,stopwords=stopwords).generate(total_text)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# ## Lets perform EDA for the Neagtive sentences

# In[32]:


data_negative = data[data['polarity']<0]
data_negative.shape


# In[33]:


data_negative.head()


# In[34]:


total_negative= (' '.join(data_negative['Summary']))


# In[35]:


total_negative


# In[36]:


import re
total_negative=re.sub('[^a-zA-Z]',' ',total_negative)


# In[37]:


len(total_negative)


# In[38]:


total_negative


# In[39]:


total_negative=re.sub(' +',' ',total_negative)


# In[40]:


len(total_negative)


# In[41]:


wordcloud = WordCloud(width = 1000, height = 500,stopwords=stopwords).generate(total_negative)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# ## Analyse to what User Amazon Can recommend more product

# #### Amazon can recommend more products to only those who are going to buy more or to one who has a better conversion rate,so lets ready data according to this problem statement

# In[42]:


df['UserId'].shape


# In[43]:


df['UserId'].nunique()


# In[44]:


df.head()


# In[45]:


raw=df.groupby(['UserId']).agg({'Summary':'count', 'Text':'count','Score':'mean','ProductId':'count'}).sort_values(by='Text',ascending=False)
raw


# In[46]:


raw.columns=['Number_of_summaries','num_text','Avg_score','Number_of_products_purchased']
raw


# In[47]:


user_10=raw.index[0:10]
number_10=raw['Number_of_products_purchased'][0:10]

plt.bar(user_10, number_10, label='java developer')
plt.xlabel('User_Id')
plt.ylabel('Number of Products Purchased')
plt.xticks(rotation='vertical')


# #### These are the Top 10 Users so we can recommend more & more Prodcuts to these Usser Id as there will be a high probability that these person are going to be buy more

# #### as data is so huge,so if your system takes a lot for the execution , u can considered some sample of data from entire data,
#     as may be some of you have not that much good specifications in terms of processor ,RAM & HArd Disk..
#     so according to system specifications,u can considered some sample of data,if u have not issue with your specifications,
#     u can go ahead with this bulky data

# In[48]:


df.head()


# In[49]:


## picking a random sample
final=df.sample(n=2000)


# In[50]:


final=df[0:2000]


# check missing values in dataset

# In[51]:


final.isna().sum()


# #### Removing the Duplicates if any

# In[52]:




final.duplicated().sum()


# ### Analyse Length of Comments whether Customers are going to give Lengthy comments or short one

# In[53]:


final.head()


# In[54]:


final['Text'][0].split(' ')


# In[55]:


len(final['Text'][0].split(' '))


# In[56]:


final['Text'][0]


# In[57]:


def calc_len(text):
    return(len(text.split(' ')))
    


# In[58]:


final['text_lenght']=final['Text'].apply(calc_len)


# In[59]:


pip install plotly


# # box plot to show customer feeback

# In[60]:


import plotly.express as px
px.box(final, y="ProductId")


# #### Analyze Score

# In[63]:


sns.countplot(final['Score'],palette='plasma')


# In[ ]:





# # analyse behaviour of customer

# In[64]:


final.head()


# In[65]:


final['Text']=final['Text'].str.lower()


# In[66]:


final['Text']


# In[67]:


from wordcloud import WordCloud, STOPWORDS 


# In[68]:


comment_words=' '.join(final['Text'])


# In[69]:


stopwords = set(STOPWORDS) 


# In[70]:


wordcloud=WordCloud(width=800 ,height=800 ,stopwords=stopwords).generate(comment_words)
plt.figure(figsize = (8, 8)) 
plt.imshow(wordcloud)
plt.axis("off") 


# In[71]:


final['Text'][34].replace('br','')


# In[72]:


for i in range(len(final['Text'])):
    final['Text'][i]=final['Text'][i].replace('br','')


# In[73]:


comment_words=' '.join(final['Text'])


# In[74]:


stopwords = set(STOPWORDS) 


# In[75]:


wordcloud=WordCloud(width=800 ,height=800 ,stopwords=stopwords).generate(comment_words)
plt.figure(figsize = (8, 8)) 
plt.imshow(wordcloud)

plt.axis("off") 


# In[ ]:





# In[ ]:




