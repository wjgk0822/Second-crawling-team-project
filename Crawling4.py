import pandas as pd

df=pd.read_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/cleaned_review.csv')

df.dropna(inplace=True)

df.info()

print(df.head(10))


#print(df['title'].unique())

#exit()

#()

one_sentences=[]

for title in df['title'].unique():

    temp=df[df['title']==title]

    one_sentence=' '.join(temp['cleaned_sentences'])

    one_sentences.append(one_sentence)

    #print(one_sentence)

    #exit()

df_one=pd.DataFrame({'titles':df['title'].unique(),'reviews':one_sentences})

print(df_one.head())

df_one.info()

df_one.to_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/cleaned_one_review.csv',index=False)



