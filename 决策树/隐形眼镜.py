# coding: utf-8
# 应用trees.py 和 treePlotter.py
# 2016年01月30日

if __name__ == '__main__':

	import trees
	import treePlotter 


	fr = open('lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	# print lenses
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = trees.createTree(lenses, lensesLabels)
	print lensesTree, '\n'
	treePlotter.createPlot(lensesTree)

else:
	import trees
	import treePlotter 


	fr = open('lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = trees.createTree(lenses, lensesLabels)
	print lensesTree
	treePlotter.createPlot(lensesTree)
