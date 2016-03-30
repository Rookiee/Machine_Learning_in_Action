# coding: utf-8
from numpy import *
import operator
import os

# 创建测试数据
def createDataSet():
	# group = array([[1.0, 1.1], [1.0, 1.0],	[0, 0],	[0, 0.1]])
	group = array([
		[1.0, 1.1],
		[1.0, 1.0],
		[0.0, 0.0],
		[0.0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

# ---------------------k-近邻算法----------------------------
# 
# inX：用于分类的输入向量
# dataSet：输入的训练样本集
# labels：标签向量
# k：用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
	# dataSetSize： 输入训练样本的行数，dataSet是一个矩阵
	dataSetSize = dataSet.shape[0]
	# 下面都是向量化的计算，完成了输入向量(inX),到训练集中每一个点的欧氏距离，
	# 其结果distances还是一个矩阵
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5

	# argsort() 返回的是索引
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		# get(), 字典中的方法，get(voteIlabel,0), 如果键值voteIlabel存在，返回其对应的value值，否则返回0
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), 
		reverse = True)
	return sortedClassCount[0][0]

# -----------------------------从文件读取数据并进行处理-------------------------
# -----------------------------将文本记录转换Numpy的解析函数--------------------
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros( (numberOfLines, 3) )
	classLabelVector = []
	fr = open(filename)
	index  = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

# --------------------------特征归一化------------------------------
def autoNorm(dataSet):
	# 这里的dataSet就是file2matrix返回的returnMat，
	# minVals， maxVals 是每一列的最小值和最大值组成的数组
	# 其大小都是(3,), 就是minVals和maxVals各有3各数, 行向量
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	# ranges还是行向量，1x3的矩阵
	# 函数创建可能的取值范围，并创建新的返回矩阵
	ranges = maxVals - minVals
	# 先创建一个全为0，大小和returnMat相等的矩阵normDataSet,
	# 最后返回的就是normDataSet
	normDataSet = zeros(dataSet.shape) 
	# normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	# 下面两行是归一化的步骤
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals

# -------------------------计算错误率-------------------------------
# 
def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * hoRatio)
	errorCount = 0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], \
			datingLabels[numTestVecs:m],3)
		print "the classifier came back with: %d, the real answer is: %d" \
			% (classifierResult, datingLabels[i])
		if (classifierResult != datingLabels[i]):
			errorCount += 1
	print 'the total error rate is: %f' % (errorCount/float(numTestVecs))

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	# 输入待分类的特征
	percentTats = float(raw_input(\
				'percentage of time spent playing video games?'))
	ffMiles = float(raw_input("frequent filier miles earned per year? "))
	iceCream = float(raw_input("liters of iceCream consumed per year? "))
	# 导入训练数据
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	# 归一化
	normMat, ranges, minVals = autoNorm(datingDataMat)
	# 待测试向量inArr，由输入数据组成
	inArr = array([ffMiles, percentTats, iceCream])
	# 应用分类器，返回结果，整型
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels,3)
	# 将返回的结果与文字化的列表对应，索引从0开始，需要减1
	print 'You will probably like this person: ', \
		  resultList[classifierResult-1]


# -------------------------手写识别------------------------------- 
# -------------------------将32x32的二进制图像转换为1x1024向量-------
def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			# returnVect 只有一行，第一个下固定为0
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

# ------------------手写数字识别系统的测试代码----------------------
# 将img2vector获得的数据输入分类器，检测分类器的执行效果
# 执行下面代码，需要 from os import listdir (从os模块中导入函数listdir, 
#	可以列出给定目录的文件名)
def handwritingClassTest():
	hwLabels = []
	# 每个文件名都是列表中的一个元素
	trainingFileList = os.listdir('trainingDigits')
	m = len(trainingFileList) # 获取文件数量
	# m 行，1024列，每一行代表一个数据的1024个特征，m是文件的总数，即训练数据的总数
	trainingMat = zeros((m, 1024))
	for i in range(m): # 对每一行
		fileNameStr = trainingFileList[i]
		# 每一个元素表示为 假如 trainingFileList[8]='9_86.txt'
		# 需要对 trainFileList[i] 这个字符串 进行分解，分解出 9_86
		# fileNameStr.split('.')[0] 对每一个文件名进行分解，[0]只取分解后的前部分
		fileStr = fileNameStr.split('.')[0]
		# 再从9_86中，取出9， 这就是lable了吧，（解析出了分类数字）
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' %(fileNameStr))

	testFileList = os.listdir('testDigits')
	errorCount = 0.0
	# 测试集的大小
	mTest = len(testFileList)
	for i in range(mTest): # 对测试集中的每一个数据
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' %(fileNameStr))
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print ('The classsifier came back with: %d, the real answer is : %d'
			% (classifierResult, classNumStr))
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print "\nthe total number of error is: %d" % errorCount
	print "\nthe total rate is: %f" % (errorCount/float(mTest))


if __name__ == '__main__':
# 	datingDataMat, datingLabels = file2matrix('/Users/Haoyang/Documents/Test_Codes/机器学习实战/k-近邻算法/datingDataSet2.txt')
# 	print datingDataMat

	# ------------------测试img2vector函数----------------------
	testVector = img2vector('testDigits/0_13.txt')
	print testVector[0,0:31]
	print testVector[0,32:63]
	print type(testVector)
	print testVector.shape

	# ------------------测试handwritingClassTest()函数----------
	# handwritingClassTest()
