import pandas as pd

from sklearn.metrics.pairwise import linear_kernel

from scipy.io import mmread

import pickle

from konlpy.tag import Okt

import re

from gensim.models import Word2Vec

def getRecommendation(cosine_sim):
    simScore=list(enumerate(cosine_sim[-1]))

    simScore=sorted(simScore,key=lambda s:s[1],reverse=True)


    simScore=simScore[:11]

    moviIdx=[i[0] for i in simScore]

    recMoveList=df_reviews.iloc[moviIdx,0]

    return recMoveList

df_reviews=pd.read_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/cleaned_one_review.csv')

Tfidf_matrix=mmread('D:/AI_exam/SecondCrawlingProject/models/Tfidf_movie_review.mtx').tocsr()

with open('D:/AI_exam/SecondCrawlingProject/models/tfidf.pickle','rb') as f:
    Tfidf=pickle.load(f)


#sentence='화려한 액션과 소름 돋는 반전이 있는 영화'






#keyword base movie recommend

#sentence base recommend

#embedding_model=Word2Vec.load('D:/AI_exam/intel_second_crawling/models/word2vec_movie_review.model')

okt=Okt()



sentence='액션 판타지'


sentence = re.sub('[^가-힣]',' ',sentence)
tokened_sentence = okt.pos(sentence, stem=True)

df_token = pd.DataFrame(tokened_sentence, columns = ['word','class'])
df_token = df_token[(df_token['class']=='Noun') |
                    (df_token['class']=='Verb') |
                    (df_token['class']=='Adjective')]

df_stopwords = pd.read_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/stopwords (3).csv')
stopwords = list(df_stopwords['stopword'])

keywords = []
for word in df_token.word:
    if len(word) > 1 and word not in stopwords:
        keywords.append(word)

embedding_model = Word2Vec.load('D:/AI_exam/SecondCrawlingProject/models/word2vec_movie_review.model')

sim_words = []
for keyword in keywords:
    try:
        sim_word = embedding_model.wv.most_similar(keyword, topn=10)
        for word, _ in sim_word:
            sim_words.append(word)
    except:
        continue
print(sim_words)
sentence = ' '.join(sim_words)
print(sentence)
sentence_vec = Tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation)




# sentence_vec=Tfidf.transform([sentence])
# #
# cosine_sim=linear_kernel(sentence_vec,Tfidf_matrix)
# #
# recommendation=getRecommendation(cosine_sim)
# #
# print(recommendation)

#sim_sen=embedding_model.wv.most_similar(sentence,topn=)

# keyword = '마블'
#
# sim_word = embedding_model.wv.most_similar(keyword, topn=10)
#
# try:
#
#
#
#     #keyword='마블'
#
#     #sim_word=embedding_model.wv.most_similar(keyword,topn=10)
#
#     print(sim_word)
#
#     words=[keyword]
#
#     for word,_ in sim_word:
#         words.append(word)
#
#     print(words)
#
#     sentence=[]
#     count=10
#     for word in words:
#         sentence=sentence+[word]*count
#         count-=1
#
#     sentence=' '.join(sentence)
#     print(sentence)
#     sentence_vec=Tfidf.transform([sentence])
#
#     cosine_sim=linear_kernel(sentence_vec,Tfidf_matrix)
#
#     recommendation=getRecommendation(cosine_sim)
#
#     print(recommendation)
#
# except:
#
#     print('enter another keyword')













#movie title base movie recommend
# print(df_reviews.iloc[120,0])
#
#
# cosine_sim=linear_kernel(Tfidf_matrix[120],Tfidf_matrix)
#
# print(cosine_sim[0])
#
# print(len(cosine_sim[0]))
#
# recommendation=getRecommendation(cosine_sim)
#
# print(recommendation)





