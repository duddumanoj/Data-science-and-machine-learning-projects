import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
df=pd.read_csv("train.csv")
df1=pd.read_csv("test.csv")
df.loc[df["topic"]=="Allergic","topic"]=1
df.loc[df["topic"]=="Bad Taste/Flavor","topic"]=0
df.loc[df["topic"]=="Smells Bad","topic"]=2
df.loc[df["topic"]=="Packaging","topic"]=3
df.loc[df["topic"]=="Not Effective","topic"]=4
df.loc[df["topic"]=="Pricing","topic"]=15
df.loc[df["topic"]=="False Advertisement","topic"]=16
df.loc[df["topic"]=="Inferior to competitors","topic"]=17
df.loc[df["topic"]=="Didn't Like","topic"]=18
df.loc[df["topic"]=="Ingredients","topic"]=5
df.loc[df["topic"]=="Customer Service","topic"]=6
df.loc[df["topic"]=="Texture","topic"]=7
df.loc[df["topic"]=="Too Sweet","topic"]=8
df.loc[df["topic"]=="Quality/Contaminated","topic"]=9
df.loc[df["topic"]=="Too big to swallow","topic"]=10
df.loc[df["topic"]=="Shipment and delivery","topic"]=11
df.loc[df["topic"]=="Wrong Product received","topic"]=12
df.loc[df["topic"]=="Expiry","topic"]=13
df.loc[df["topic"]=="Color and texture","topic"]=14
df.loc[df["topic"]=="Customer Issues","topic"]=19
df.loc[df["topic"]=="Hard to Chew","topic"]=20
#print(df[df.columns[0:2]])
#df["topic"]=df["topic"].astype(int)
#cd={'topic':int}
#df=df.astype(cd)
#print(df.dtypes)
#df['topic']=df['topic'].apply(pd.to_numeric)

#df["topic"]=pd.to_numeric(df["topic"])
#df_y=df_y.astype({'topic':"int"})
#print(df_y.features)
df=df.infer_objects()
df1_x=df1["Review Text"]
#df_x=df[df.columns[0:1]]
df_y=df["topic"]
#print(df_x)
#print(df_x)
#print(df.dtypes)
#print(df_y.info())
#print(df.topic.unique())
#mnb: MultinomialNB=MultinomialNB()
#mnb.fit(df_x,df_y)

#df_y=df_y.astype({"topic":'int64'})
#print(df_y)
#cv = TfidfVectorizer(min_df=1,stop_words='english')
#print(t)
df_x=df["Review Text"]
xn,xt,yn,yt=train_test_split(df_x,df_y,test_size=0,random_state=4)
cv=CountVectorizer()
a=cv.fit_transform(xn)
b=cv.transform(xt)
c1=cv.transform(df1_x)
mnb=MultinomialNB()

mnb.fit(a,yn)
pred=mnb.predict(b)
#yt1=np.array(yt)
#print(pred)
c=0
pr=pd.DataFrame(pred,columns=['pred']).to_csv('pred.csv')
for i in range(len(pred)):
    if pred[i]==yt1[i]:
        c=c+1
print("no.of correct predictions")
print(c)
pri=mnb.predict(c1)
pri=pd.DataFrame(pri,columns=["pri"]).to_csv('pri.csv')