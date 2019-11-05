"""
HW06_720_DecTrees_2191
Author: Aseem Mehta
email: am1435@rit.edu
Program contains a classifier produced by another program and uses the classifier
to predict the values for the testing dataset
"""

# Importing the libraries
import pandas as pd # libtrary used to read CSV file

"""
Imports the data set using pandas library into a dataframe
return: the testing dataset in dataframe format
"""
def read(fileName):
	dataset = pd.read_csv(fileName)
	return dataset

"""
Compares test Value with Split Criteria and returns the next Side of the tree
"""
def compare(testValue, splitCriteria):
	if testValue < splitCriteria:
		nextSide = "LEFT"
	else:
		nextSide = "RIGHT"
	return nextSide

"""
Based on the classifier assigns values to dProgram ata in testing File
"""
def AssignValue(dataset, treeStructure):

	values = [] # stores all the predicted values

	# traverses through all the indexes in the test set
	for index in range(0,len(dataset)):

		node = 0 # Starting point of treeStructure (pointer)
		if treeStructure[node][1] is None:# None specifies it is a Leaf Node
											# add Class Value in the values list
			values.append(treeStructure[node][0])
		else:
			# Checks criteria for the node if it is not a leaf node, comparision
			# informs which side of tree we have to traverse
			nextSide = compare(dataset.iloc[index][treeStructure[node][0]],treeStructure[node][1])
			"""
			nextDepth informs which nodes to check next, depth 10 can have
			maximum of subsequent nodes at depth 9
			"""
			nextDepth = treeStructure[node][2]-1

			node = 1
			# traverses through the rest of tree structure
			while(node < len(treeStructure)):

				# Checks depth of nodes in treeStructure with the nextDepth to
				# only check the subsequent nodes
				if nextDepth == treeStructure[node][2]:
					if nextSide == "LEFT":
						if treeStructure[node][1] is None: # Leaf Node
							values.append(treeStructure[node][0])
							break
						else:
							"""
							Checks criteria for the node if it is not a leaf
							node, comparision informs which side of tree
							program has to traverse
							"""
							nextSide = compare(dataset.iloc[index][treeStructure[node][0]],treeStructure[node][1])
							nextDepth = treeStructure[node][2]-1

					else:
						# nextSide is Right
						onlyRight = False # states if only right side is available
						tempNode = node # pointer to probe for the right Node

						# Skipping left side of tree as In preorder left side
						# will always come first
						tempNode  += 1
						# traverses through the rest of the tree to find
						while(tempNode < len(treeStructure) and treeStructure[tempNode][2]!=nextDepth):
							tempNode += 1
							"""
							If loop doesn't find another node with same depth
							then node had only one child node as right node"""
							if tempNode >= len(treeStructure):
								onlyRight = True
						# if node had 2 child nodes, using the second one
						if not onlyRight:
							node = tempNode
						if treeStructure[node][1] is None: # Leaf Node
							values.append(treeStructure[node][0])
							break
						else:
							# Checks depth of nodes in treeStructure with the
							# nextDepth to only check the subsequent nodes
							nextSide = compare(dataset.iloc[index][treeStructure[node][0]],treeStructure[node][1])
							nextDepth = treeStructure[node][2]-1
				node+=1
	return values

# creates a csv file where results are stored
def writeprediction(values, classValues ):
	f = open("HW06_Mehta_Aseem_MyClassifications.csv","w+")
	for currentValue in values:
		if classValues[1] == currentValue:
			f.write('1\n')
		else:
			f.write('0\n')
	f.close()

"""
Start point of program, runs the program in a defined sequence
"""
def main():
	testingFileName = 'DT_Data_CakeVsMuffin_v012_TEST.csv'

	"""
	defines the tree order traversal and values in preorder
	TreeStructure consists of a list of nodes, these nodes define every node of
	the program's classifier tree. 
	Structure of node
	e.g. 1 ['Flour', 5, 10], Flour is the Attribute to check, 5 is the
	splitting criteria and 10 is the tree level in desecnding order, starting 
	at 10 (see figure in writeup)
	e.g. 2 ['CupCake', None, 7], None specifies it is a leaf node, CupCake is 
	the return class and 7 is the level of the tree.
	"""

	treeStructure = [['Flour', 5, 10], ['Oils', 3, 9], ['Flour', 3, 8], ['CupCake', None, 7], ['Muffin', None, 7], ['CupCake', None, 8], ['Oils', 7, 9], ['Muffin', None, 8], ['Flour', 7, 8], ['CupCake', None, 7], ['Muffin', None, 7]]
	dataset = read(testingFileName)
	values = AssignValue(dataset, treeStructure)
	classValues = ['Muffin', 'CupCake']

	writeprediction(values, classValues)

if __name__ == '__main__':
	main()