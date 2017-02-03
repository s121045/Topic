#!/usr/bin/env python
# -*- coding:utf-8 -*-

from file import line_list #(file.pyから文章を読み込む)
import MeCab
import gensim
from gensim import corpora, models, similarities

golo = []
##文章を名詞に分割する
mecab = MeCab.Tagger('-Ochasen')
def extractKeyword(text):
    """textを形態素解析して、名詞のみのリストを返す"""
    mecab.parse('')
    node = mecab.parseToNode(text)
    keywords = []
  #  texts = []
    while node:
        if node.feature.split(",")[0] == "名詞":
             keywords.append(node.surface)
        node = node.next
    golo.append(keywords)     
    return keywords

if __name__ == "__main__":
    for i in line_list:
       keywords = extractKeyword(i)

#stop_word_list = ["ため","これ","それ","的","(",")","０","１","２","３","４","５","６","７","８","９","1","2","3","4","5","6","7","8","9","日","私","たち","こと","自分","自身","さん"]
#golo = [[word for word in keywords if word not in stop_word_list] for keywords in golo]

#特徴語辞書の作成
dictionary = corpora.Dictionary(golo)

#低頻度語や二割以上の単語を削除
dictionary.filter_extremes(no_below=2, no_above = 0.2)

##text全体に対する特徴ベクトルの集合= corpusを作成する。
corpus = [dictionary.doc2bow(keywords) for keywords in golo]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
#print(corpus)

# LDAのモデルの呼出と学習 ここでtopicの数(ユーザー層の数）を設定出来る
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)

#トピックの出力
for topic in lda.show_topics(-1,8):
   print(topic)
   print() 

#各文書ごとの推定トピックを出力
for topics_per_document in lda[corpus]: 
   print (topics_per_document)
   print()
 
