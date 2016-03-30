# coding: utf-8
# 导入log函数，在计算给定数据集熵的时候用
from math import log
# 在majorityCnt()中用operator.itemgetter()
import operator

# ----------------------------计算给定数据集的熵----------------------------
def calcShannonEnt(dataSet):
	# dataSet 的类型应该是list
	# 计算数据集中实例的总数，也可以在需要的时候再计算
	numEntries = len(dataSet)
	# 为所有可能分类创建字典
	labelCounts = {}	# dict，键值是最后一列的数值
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob, 2) #  以2为底求对数
	return shannonEnt

# 简单测试，
def createDataSet():
	dataSet = [[1,1,'yes'],
			   [1,1,'yes'],
			   [1,0,'no'],
			   [0,1,'no'],
			   [0,1,'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


# ----------------------------按照给定特征划分数据集----------------------------
def splitDataSet(dataSet, axis, value):
	retDataSet = []	# 创建新的list对象
	# 遍历数据集中的每个元素，一旦发现符合要求的，就添加到retDataSet中
	# featVec 是data中的每一个元素，也是list
	for featVec in dataSet:
		# 假如现在featVec(dataSet中的一个元素) 是[1，‘ok’，‘yes']
		# axis： 索引，假如是1，即值为ok
		# featVec[:axis] 取出ok前的1
		# featVec[axis+1:]  取出ok后的yes
		# 循环内的两个 reducedFeatVec将 前、后 合并为 [1,'yes']
		# 最后在加入到retDataSet中
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

# ----------------------------选择最好的数据集划分方式----------------------------
# 有了上面的划分函数，还需要选择最好的划分方式，（考虑上面函数传入什么参数最合适）
# 可遍历真个数组，循环计算香浓熵和splitDataSet()函数，找到最好的划分方式
# 熵会得出如何划分数据集最好
# 该函数实现选取特征，划分数据集，计算出最好的划分数据集的特征
def chooseBestFeatureToSplit(dataSet):
	# 要求：1， 数据必须是由列表元素组成的列表，元素个数必须相等
	# 	   2， 最后一个元素必须是标签
	numFeatures = len(dataSet[0])-1 #(dataSet[i]中包含的元素，其中最后一个是分类，-1后得到特征数量)
	# 计算整个数据集的原始香农熵，保存最初的无序度量值，用来和划分之后的数据集计算的香农熵进行比较
	baseEntropy = calcShannonEnt(dataSet)	# 计算给定数据集的熵
	bestInfoGain = 0.0
	bestFeature = -1
	# 遍历数据集中所有的特征
	for i in range(numFeatures):
		# 使用列表推导创建新列表
		featList = [example[i] for example in dataSet]
		# print "The feature [%d]: " % i , featList
		
		uniqueVals = set(featList)	# 利用集合特性，去除重复
		# print "The uniqueVals [%d]: " %i, uniqueVals, '\n'
		newEntropy = 0.0
		# 遍历当前特征中所有唯一属性值，对每个特征划分一次数据集
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			# 对每个特征划分出的数据集求熵，并对所有唯一特征值得到的熵求和得 newEntropy
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		# 计算信息增益
		infoGain = baseEntropy - newEntropy
		# 获得最好的信息增益
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature  # 得出按 第bestFeature 个特征 划分数据集最好

# ----------------------------决定子叶节点----------------------------
# 采用多数表决的方法决定该子叶节点的分类
# 使用分类名称列表做为参数
def majorityCnt(classList):
	# 创建 键值为classList中唯一值的数据字典，字典对象存储classList中每个标签出现的频率
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	# 利用operator操作键值排序字典，从大打小(reverse = True)
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	# 返回出现次数最多的
	return sortedClassCount[0][0]


# ----------------------------创建树----------------------------
# labels: 列表，包含了数据集中所有的特征标签
def createTree(dataSet, labels):
	# classList: 包含了数据集的所有类标签 （取出dataSet每一行的最后一个元素组成新列表）
	classList = [example[-1] for example in dataSet]
	# print classList
	# 类别完全相同，停止接续划分
	if classList.count(classList[0]) == len(dataSet):
		return classList[0]
	# 遍历完所有特征时，返回出现次数最多的
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]

	myTree = {bestFeatLabel: {} }
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree( splitDataSet(dataSet,bestFeat,value), subLabels)
	return myTree


# ----------------------------使用决策树----------------------------
def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	# 将标签字符串转换为索引
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key],featLabels, testVec)
			else:
				classLabel = secondDict[key]

	return classLabel

# ------------------------使用pickle模块存储决策树------------------------------
# 为了在每次执行分类时，能够调用已经构造好的决策树，
# 使用pickle模块序列化对象，序列化对象可以在磁盘上保存，并在需要的时候读取
# 任何对象都可以序列化，包括字典
def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, 'w')
	pickle.dump(inputTree, fw)
	fw.close()

def grabTree(filename):
	import pickle
	fr = open(filename,'r')
	return pickle.load(fr)


if __name__ == '__main__':
	# ----------------------------测试calaShannonEnt()----------------------------
	# myDat, labels = createDataSet()
	# print myDat
	# # print type(myDat)
	# # print myDat[0]	# 每一个元素也是list
	# # print type(myDat[0])
	# print calcShannonEnt(myDat)
	# # 熵越高，则混合的数据越多，下面在myDat中添加一个分类
	# myDat.append([1,1,'maybe'])
	# print myDat
	# print calcShannonEnt(myDat)

	# ----------------------------测试splitDataSet()----------------------------
	# myDat, labels = createDataSet()
	# print splitDataSet(myDat, 0,1)
	# print splitDataSet(myDat, 2,'no')
	# print myDat[0]
	# print len(myDat[0])-1

	# ----------------------------测试chooseBestFeatureToSplit()------------------
	# myDat, labels = createDataSet()
	# print chooseBestFeatureToSplit(myDat)
	
	# ----------------------------测试chooseBestFeatureToSplit()------------------
	# myDat, labels = createDataSet()
	# myTree = createTree(myDat, labels)
	# print myTree


	# ----------------------测试calssify(inputTree, featLabels, testVec)------------
	# import treePlotter as tr
	# myDat, labels = createDataSet()
	# print labels
	# print myDat
	# myTree = tr.retrieveTree(0)
	# print myTree
	# print classify(myTree, labels, [1,0])
	# print classify(myTree, labels, [1,1])

	# ----------------------测试pickle 写入和读取------------
	myDat, labels = createDataSet()
	myTree = createTree(myDat, labels)
	print "Old Tree: ", myTree
	# 写入文件
	storeTree(myTree, 'Storage.txt')	# 执行后，在目录下有了Storage.txt
	# 重新读取决策树
	myTreeNew = grabTree('Storage.txt')
	print "New Tree: ", myTreeNew

	