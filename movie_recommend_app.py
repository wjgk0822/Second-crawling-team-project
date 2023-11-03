import sys
#from PIL import Image
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import numpy as np
#from tensorflow.keras.models import load_model
from PyQt5 import uic

import pandas as pd

from sklearn.metrics.pairwise import linear_kernel

from gensim.models import Word2Vec

from scipy.io import mmread

import pickle

from PyQt5.QtCore import QStringListModel



form_window = uic.loadUiType('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/LegoTeam.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.Tfidf_matrix=mmread('D:/AI_exam/SecondCrawlingProject/models/Tfidf_movie_review.mtx').tocsr()
        with open('D:/AI_exam/SecondCrawlingProject/models/tfidf.pickle','rb') as f:
            self.Tfidf=pickle.load(f)

        self.embedding_model=Word2Vec.load('D:/AI_exam/SecondCrawlingProject/models/word2vec_movie_review.model')




        self.df_reviews=pd.read_csv('D:/AI_exam/SecondCrawlingProject/Second-crawling-team-project/cleaned_one_review.csv')

        self.titles=list(self.df_reviews['titles'])

        self.titles.sort()

        for title in self.titles:

            self.comboBox.addItem(title)

        model=QStringListModel()

        model.setStringList(self.titles)

        completer=QCompleter()

        completer.setModel(model)

        self.le_keyword.setCompleter(completer)



        self.comboBox.currentIndexChanged.connect(self.combobox_slot)

        self.btn_recommend.clicked.connect(self.btn_slot)


    def btn_slot(self):

        # title=self.le_keyword.text()
        #
        # recommendation = self.recommendation_by_movie_title(title)
        #
        # self.lbl_recommend.setText(recommendation)


        keyword=self.le_keyword.text()

        self.le_keyword.setText('')

        print(keyword)

        if keyword:

            if keyword in self.titles:
                recommendation=self.recommendation_by_movie_title(keyword)
                #recommendation=self.recommendation_by_keyword(keyword)
                self.lbl_recommendation.setText(recommendation)

            else:

                print('debug01')
                #recommendation=self.recommendation_by_movie_title(keyword)
                recommendation = self.recommendation_by_keyword(keyword)
                print(recommendation)

                self.lbl_recommendation.setText(recommendation)


    def recommendation_by_keyword(self,keyword):
        #sim_word=self.embedding_model.wv.most_similar(keyword)

        #for keyword in keywords:
        try:
            sim_word = self.embedding_model.wv.most_similar(keyword, topn=10)

            words=[keyword]
            for words, _ in sim_word:
                sim_word.append(words)

            sentence=[]
            count=10

            for word in words:
                sentence=sentence+[word]*count
                count-=1

            sentence=' '.join(sentence)
            print(sentence)

            sentence_vec=self.Tfidf.transform([sentence])

            cosine_sim=linear_kernel(sentence_vec,self.Tfidf_matrix)
            recommendation=self.getRecommendation(cosine_sim)

            return recommendation

        except:

            return 'type different keywords'


                #continue





       # except:








    def combobox_slot(self):

        title=self.comboBox.currentText()

        recommendation=self.recommendation_by_movie_title(title)

        self.lbl_recommend.setText(recommendation)




    def recommendation_by_movie_title(self,title):

        movie_idx=self.df_reviews[self.df_reviews['titles']==title].index[0]
        cosine_sim=linear_kernel(self.Tfidf_matrix[movie_idx],self.Tfidf_matrix)

        recommendation=self.getRecommendation(cosine_sim)

        #self.lbl_recommend.setText(recommendation)

        return recommendation



    def getRecommendation(self,cosine_sim):
        simScore = list(enumerate(cosine_sim[-1]))

        simScore = sorted(simScore, key=lambda s: s[1], reverse=True)

        simScore = simScore[:11]

        moviIdx = [i[0] for i in simScore]

        recMoveList = self.df_reviews.iloc[moviIdx, 0]

        recMoveList='\n'.join(recMoveList)



        print(recMoveList)

        return recMoveList














        # self.setFixedWidth(600)
        # self.setFixedHeight(600)
        # self.setupUi(self)
        # self.btn_open.clicked.connect(self.btn_clicked_slot)
        # model_path = './cat_and_dog_0.833.h5'
        # self.model = load_model(model_path)
        # self.path = ('../datasets/cat_dog/test/cat_test01.jpg', '')
        # pixmap = QPixmap(self.path[0])
        #self.lbl_image.setPixmap(pixmap)

    #def btn_clicked_slot(self):
        # old_path = self.path
        # self.path = QFileDialog.getOpenFileName(self, 'Open file',
        #         '../datasets/cat_dog', 'Image Files(*.jpg;*.png);;All Files(*.*)')
        # if self.path[0] == '':
        #     self.path = old_path
        # print(self.path)
        # pixmap = QPixmap(self.path[0])
        # self.lbl_image.setPixmap(pixmap)
        # try:
        #     img = Image.open(self.path[0])
        #     img = img.convert('RGB')
        #     img = img.resize((64, 64))
        #     data = np.asarray(img)
        #     data = data / 255
        #     data = data.reshape(1, 64, 64, 3)
        #
        #     pred = self.model.predict(data)
        #     print(pred)
        #     if pred < 0.5:
        #         self.lbl_result.setText('고양이입니다.')
        #     else :
        #         self.lbl_result.setText('강아지입니다.')
        # except:
        #     print('error : {}'.format(self.path[0]))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())