# download ptt file and put it in 2018台大電機系AI體驗營/
# https://www.dropbox.com/s/9cpk2ymdiq4nnnt/ptt100M.txt?dl=0
# install gensim usage:
#      conda install gensim
#-------------------------------------------------------------
# demo website: http://140.112.21.35:2880/~tlkagk/Word2Vec/


from gensim.models import word2vec
from gensim import models

sentences = word2vec.LineSentence("ptt100M.txt")
model = word2vec.Word2Vec(sentences, size=250)
model.save("ptt.model")

print('Loading models')
model = models.Word2Vec.load('ptt.model')


print("相似詞前 100 排序")
res = model.most_similar(u'    Todo    ',topn = 20)
for item in res:
	print(item[0]+","+str(item[1]))


print("計算 Cosine 相似度：")
res = model.similarity(u'    Todo    ',u'    Todo    ')
print(res)

x,y,z = u'    Todo    ',u'    Todo    ',u'    Todo    '
a = u'    Todo    '
print(a)
print("{}之於{}，如{}之於".format(x,y,z))
res = model.most_similar([x,y], [z], topn= 20)
for item in res:
    print(item[0]+","+str(item[1]))
