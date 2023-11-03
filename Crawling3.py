import pandas as pd
from konlpy.tag import Okt
import re

df=pd.read_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/reviews_fantasy_1_9.csv')

df.info()

okt=Okt()

df_stopwords=pd.read_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/stopwords (3).csv')

stopwords=list(df_stopwords['stopword'])

count=0

cleaned_sentences=[]

for review in df.reviews:

    count+=1

    if count % 10==0:
        print('.',end=' ')

    if count % 100==0:
        print()


    if count % 1000==0:
        print(count/1000,end='')





    review=re.sub('[^가-힣]',' ',review)
    tokened_review=okt.pos(review,stem=True)


    df_token=pd.DataFrame(tokened_review,columns=['word','class'])

    df_token=df_token[(df_token['class']=='Noun') | (df_token['class']=='Verb') | (df_token['class']=='Adjective')]

    #print(df_token.head())


    words=[]

    for word in df_token.word:
        if 1<len(word):
            if word not in stopwords:
                words.append(word)

    cleaned_sentence=' '.join(words)

    cleaned_sentences.append(cleaned_sentence)

#df_test=df.iloc[:10,:]

df['cleaned_sentences']=cleaned_sentences

df=df[['title','cleaned_sentences']]

print(df.head(10))

df.info()

df.to_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/cleaned_review.csv',index=False)




    #exit()



    #print(tokened_review)



    #exit()



