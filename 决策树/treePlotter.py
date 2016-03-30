# coding: utf-8
import matplotlib.pyplot as plt 

# 使用文本注解回执树节点

# 定义文本框和箭头格式， 树节点格式的常量
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
# decisionNode = {boxstyle:'sawtooth', fc:'0.8'}	# error
leafNode = dict(boxstyle= "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")






# 首先创建一个新图形并清空区域，然后绘制两个代表不同类型的树节点 
# def createPlot():
# 	fig = plt.figure(1, facecolor = 'white')
# 	fig.clf()
# 	createPlot.ax1 = plt.subplot(111, frameon = False)
# 	plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
# 	plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
# 	plt.show()








# 为了获取完整的树形图，需要知道有多少个叶节点，以便知道x轴的长度
# 需要知道有多少层，一遍知道y轴的高度
# 下面函数获取节点和层的数目
def getNumLeafs(myTree):
	numLeafs = 0
	# print "The tree is: ", myTree
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	# print "secondDict: ", secondDict

	for key in secondDict.keys():
		# 测试节点的数据类型是否是字典
		# 如果该节点是字典类型，则该节点也是判断节点，需要递归调用getNumLeafs()
		if type(secondDict[key]).__name__ == 'dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__ == 'dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1
		if thisDepth > maxDepth:
			maxDepth = thisDepth
	return maxDepth




# 绘制带箭头的注解
# 该函数执行实际绘图功能，该函数需要一个绘图区域，该区域由全局变量createPlot.ax1定义
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = "axes fraction",
		xytext = centerPt, textcoords = "axes fraction", va = "center", 
		ha = "center", bbox = nodeType, arrowprops = arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)




def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]




if __name__ == '__main__':
	myTree = retrieveTree(0)
	firstStr = myTree.keys()[0]
	print myTree
	print myTree[firstStr]

	

	