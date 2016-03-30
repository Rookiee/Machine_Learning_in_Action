# -*- coding: utf-8 -*-
"""


@author: 
"""
# coding: utf-8
# -------------------------词表到向量的转换函数---------------------------------
# 返回的第一个变量是词条切分后的文档集合
# 返回的第二个变量是一个类别标签的集合，用于训练程序
def loadDataSet():
    postingList = [
                ['my','dog','has','flea','problem','help','please'],
                ['maybe','not','take','him','to','dog','park','stupid'],
                ['my','dalmation','is','so','cute','I','love','him'],
                ['stop','posting','stupid','worthless','garbage'],
                ['mr','licks','ate','my','steak','how','to','stop','him'],
                ['quit','buying','worthless','dog','stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    # print '\n', type(postingList), '\n', len(postingList)
    return postingList, classVec

# 创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
	vocabSet = set([])	# 创建一个空集合，集合中的每一个元素都是列表
	# 遍历数据集（List），取List中的每一个元素（List），并把每一个元素设为集合（去除重复）
	# 再和空集合做并操作，取出dataSet中所有的字母（无重复项）
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	# 把集合转换为列表，返回(词汇表)
	return list(vocabSet)


# 词集模型 Set-of-words-model
# 函数输入为词汇表， 和某个文档inputSet
# 返回文档向量，向量每个元素为0或1，分表表示词汇表中的单词在inputSet中是否出现
def setOfWords2Vec(vocabList, inputSet):
	# 首先，创建一个和词汇表等长的向量
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			# list.index(element)获取某个元素的下标
			returnVec[vocabList.index(word)] = 1
		else:
			print 'The word: "%s" is not in the vocabulary' %(word)
	return returnVec


# 朴素贝叶斯词袋模型
# 如果一个词在文档中不止出现一次,可能意味着"包含该词是否出现在文档中"所不能表达的某种信息
# 这种方法称为"词袋模型"
# 在词袋中,每个单词可以出现多次
# 在词集中,每个单词只能出现一次
# 词袋与词集的唯一不同:每当遇到一个单词,它会增加词向量中的对应值,而不只是将对应的数字设为1
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec


from numpy import *
# 朴素贝叶斯分类器训练函数
# trainMatrix: 文档矩阵
# trainCategory: 每篇文档类别标签所构成的向量,
def trainNB0(trainMatrix, trainCategory):	#112行调用
	# print len(trainMatrix), len(trainCategory) # equal
	# 传入矩阵中文档的总数目
	numTrainDocs = len(trainMatrix)
	# ?
	numWords = len(trainMatrix[0])
	# print numWords	# 31
	# print sum(trainCategory)	# 3, 说明有3个文档含有侮辱性词汇
	pAbusive = sum(trainCategory)/float(numTrainDocs)	# 侮辱性词的概率
	print "pAbusive: %f " %(pAbusive)
	p0Num = ones(numWords)	# 若一个为0,p(w1|1)*p(w2|1)...最后结果为0,所以初始化为1, Denom=2.0
	# print p0Num, '\n', len(p0Num)	# 由1组成的ndarray
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:	# 第i个文档包含侮辱词汇
			p1Num += trainMatrix[i] 
			p1Denom += sum(trainMatrix[i])
			# print "%d: p1Num" %i;	print p1Num;	print p1Denom
		else:
			p0Num += trainMatrix[i]	
			p0Denom += sum(trainMatrix[i])
			# print "%d: p0Num" %i; 	print p0Num;	print p0Denom
	p1Vect = log(p1Num/p1Denom)	# 很多很小的数相乘,最后结果为0,所以这里用ones
	p0Vect = log(p0Num/p0Denom)
	return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPost, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPost)
	trainMat = []

	for postinDoc in listOPost:
		# print postinDoc
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses)) # 传参之前转为numpy数组

	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)

	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


#---------------------------------------------------------------------------------------#
# 过滤垃圾邮件
# 框架:
# 1. 收集数据: 提供文本文件
# 2. 准备数据: 将文本文件解析成词条向量
# 3. 分析数据: 检查词条确保解析的正确性
# 4. 训练算法: 使用前面的trainNB0()函数
# 5. 测试算法: 使用classifyNB(), 并且构建一个新的测试函数来计算文档集的错误率
# 6. 使用算法: 构建一个完整的程序对一组文档进行分类,将错误的文档输出到屏幕上


# 文件解析及完整的垃圾邮件测试函数
def textParse(bigString):	# 接收一个字符串,并转换为字符串列表
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]
# 
def spamTest():
	docList = []
	classList = []
	fullText = []
	for i in range(1, 26):
		wordList = textParse(open('email/spam/%d.txt' %(i) ).read() )
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' %(i) ).read() )
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = range(50)
	testSet = []
	# 随机构建测试集
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])	# 将随机选出的10个从训练集剔除
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print 'The error rate is: ', float(errorCount)/len(testSet)


#---------------------------------------------------------------------------------------#
# 从个人广告中获取区域倾向
# 框架:
# 1. 收集数据: 从RSS源收集内容,需要对RSS源构建一个接口
# 2. 准备数据: 将文本文件解析成词条向量
# 3. 分析数据: 检查词条, 确保解析的正确性
# 4. 训练算法: 使用之前建立的trainNB0() 函数
# 5. 测试算法: 观察错误率,确保分类器可用.可以修改切分程序,以降低错误率,提高分类结果
# 6. 使用算法: 构建一个完整的程序, 封装所有内容.给定两个RSS源,该程序会显示最常用的公共词











if __name__ == '__main__':
	# testingNB()
	#--------------------------------------------
	# 要使用正则表达式
	# import re
	# regEx = re.compile('\\W*')
	# mySent = 'This book is hte best book on Python or M.L. I have ever laid eyes upon.'
	# listOfTokens = regEx.split(mySent)
	# print listOfTokens
	# print [tok.lower() for tok in listOfTokens if len(tok)>0]	# 变为小写, 消除长度为0的元素
	print spamTest()
	print spamTest()
	print spamTest()
	print spamTest()
	print spamTest()
	print spamTest()
	#--------------------------------------------

	
	





	
