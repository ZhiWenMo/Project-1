
# coding: utf-8

# In[1]:


import re
from zhon.hanzi import punctuation
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
import editdistance
import pickle
from tqdm import tqdm


# In[2]:


with open('/Users/mozhiwen/Documents/Data_source/chinese_stopwords.txt','r') as file:
    stopwords=[i[:-1] for i in file.readlines()]


# In[3]:


news = pd.read_csv('sqlResult_1558435.csv',encoding='gb18030')


# In[4]:


news.shape


# In[5]:


news.head(5)


# In[6]:


#nas in the dataset
news[news.content.isna()].head(5)


# In[7]:


#drop the nas
news=news.dropna(subset=['content'])


# In[8]:


news.shape


# In[9]:


news[news.source=='新华社'].shape


# In[10]:


78661/87054


# In[11]:


def split_text(text):return ' '.join([w for w in list(jieba.cut(re.sub('\s|[%s]' % (punctuation),'',text))) if w not in stopwords])


# In[12]:


split_text(news.iloc[1].content)


# In[13]:


jieba.enable_parallel(4)


# In[14]:


#prepare data for machine learning
#corpus=list(map(split_text,[str(i) for i in news.content]))
with open('corpus.pkl','rb') as file:
    corpus = pickle.load(file)


# In[15]:


jieba.disable_parallel()


# with open('corpus.pkl','wb') as file:
#     pickle.dump(corpus,file)

# In[16]:


countvectorizer = CountVectorizer(encoding='gb18030',min_df=0.015)
tfidftransformer = TfidfTransformer()
clf = MultinomialNB()


# In[17]:


countvector = countvectorizer.fit_transform(corpus)


# In[18]:


countvector.shape


# In[19]:


tfidf = tfidftransformer.fit_transform(countvector)


# In[20]:


tfidf.shape


# In[135]:


label=list(map(lambda source: 1 if source == '新华社' or source == '新华网' else 0,news.source))


# In[136]:


#split the data
X_train, X_test, y_train, y_test = train_test_split(tfidf,label,test_size = 0.3, random_state=42)


# In[137]:


clf.fit(X=X_train.toarray(),y=y_train)


# In[138]:


scores=cross_validate(clf,X_train.toarray(),y_train,scoring=('precision','recall','accuracy','f1'),cv=3,return_train_score=True)


# In[139]:


scores


# In[140]:


y_predict = clf.predict(X_test.toarray())


# In[141]:


def show_test_reslt(y_true,y_pred):
    print('accuracy:',accuracy_score(y_true,y_pred))
    print('precison:',precision_score(y_true,y_pred))
    print('recall:',recall_score(y_true,y_pred))
    print('f1_score:',f1_score(y_true,y_pred))


# In[142]:


show_test_reslt(y_test,y_predict)


# ## Use the model to detect copy news
# 
# ------------------

# In[143]:


prediction = clf.predict(tfidf.toarray())

labels = np.array(label)


# In[144]:


compare_news_index = pd.DataFrame({'prediction':prediction,'labels':labels})


# In[145]:


copy_news_index=compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index


# In[245]:


xinhuashe_news_index=compare_news_index[(compare_news_index['labels'] == 1)].index


# In[146]:


len(copy_news_index)


# - In sklearn, in order to use cosine similarity in kmeans, we should scale the data to unit norm then use the kmeans class

# In[195]:


from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from collections import defaultdict


# In[196]:


normalizer = Normalizer()


# In[197]:


scaled_array = normalizer.fit_transform(tfidf.toarray())


# In[342]:


kmeans = KMeans(n_clusters=25,random_state=42,n_jobs=-1)
k_labels = kmeans.fit_predict(scaled_array)


# In[344]:


id_class = {index:class_ for index,class_ in enumerate(k_labels)}


# In[345]:


class_id = defaultdict(set)
for index,class_ in id_class.items():
    if index in xinhuashe_news_index.tolist():
        class_id[class_].add(index)


# In[346]:


count=0
for k in class_id:
    print(len(class_id[k]),count)
    count +=1


# In[370]:


def print_element(ids,n=10):
    count=0
    for i in ids:
        if count > 10: break
        count+=1
        print(news.iloc[i].content,'\n')


# In[371]:


for k in class_id:
    print(k,'\n')
    print_element(class_id[k],5)
    


# In[347]:


def find_similar_text(cpindex,top=10):
    dist_dict={i:editdistance.eval(corpus[cpindex],corpus[i]) for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.items(),key=lambda x:x[1])[:top]


# In[348]:


copy_news_index.tolist()


# In[357]:


fst=find_similar_text(3352)


# In[358]:


print('怀疑抄袭:\n')

print(news.iloc[3352].content)

print('相似原文:\n')

print(news.iloc[3134].content)

print('editdistince:',editdistance.eval(corpus[3352],corpus[3134]))


# In[360]:


#for i in fst:print(news.iloc[i[0]].content,i) 


# In[289]:


from random import sample


# In[291]:


dis_sort_dict={}
for i in tqdm(sample(copy_news_index.tolist(),300)):
    dis_sort_dict[i] = find_similar_text(i)

