
#source code python file
import numpy as np
import seaborn as sns
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
df=df.infer_objects()
df1_x=df1["Review Text"]
df_x=df["Review Text"]
cv=CountVectorizer()
a=cv.fit_transform(df_x)
c1=cv.transform(df1_x)
mnb=MultinomialNB()
df_y=df["topic"]
mnb.fit(a,df_y)
pri=mnb.predict(c1)
pri=pd.DataFrame(pri,columns=["pri"]).to_csv('pri.csv')
g=df.groupby(['Review Text'])
print(g.first())