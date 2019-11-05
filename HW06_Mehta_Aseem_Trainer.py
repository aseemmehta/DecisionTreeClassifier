"""
HW06_720_DecTrees_2191
Author: Aseem Mehta
email: am1435@rit.edu
Program uses a tarining dataset to train a decision tree classifier, and write
another program to use the classifier on testing data
"""
# Importing the libraries
import math as m
import pandas as pd
import numpy as np


"""
Imports the data set using pandas library into a dataframe
return: the training dataset in dataframe format
"""
def read(fileName):
    dataset = pd.read_csv(fileName)
    return dataset

"""
checks if the Stopping criteria is met
"""
def stoppingCriteria(classCount, totalClasses,currentDepth):
    criteriaMet = False #initialize criteria met value to false
    returnClass = None #initialize return class to None 
    """
    criteriaMet and returnClass identifies if criteria is not met
    """
    for classKey in classCount:
        if classCount[classKey] > (totalClasses-22): #Returns class 0 if class1
                                                # has less than 23 data points
            criteriaMet = True
            returnClass = classKey
        elif classCount[classKey]/totalClasses >= .9:# returns a class if class
                                                    # is more than 90% of data
            criteriaMet = True
            returnClass = classKey
        elif currentDepth==0:
            criteriaMet = True
            returnClass = classKey
    return criteriaMet, returnClass

"""
Finds and returns the count of every class for the data
"""
def findClassCount(dataset,target):
    uniqueTargetValues = list(set(target)) # Identifies the unique values in 
                                            #target and returns in list
    
    targetCountDict = {} # Dictionary stores Class and corresponding count
    for targetValue in uniqueTargetValues:
        targetCountDict[targetValue] = target.count(targetValue)
    
    return targetCountDict, uniqueTargetValues


"""
Traverses through all attribute values to find counts of classes for both less 
than, greater than and equal to values
"""
def findCount(splitValue, attribute, target):
    leftDict = {} # contains count of class values for less than
    totalLeft = 0 # total number of left values
    rightDict = {} # contains count of class values for greater than, equal to
    totalRight = 0 # total number of right values
    """
    traverses through all attribute values to find count
    """
    for index in range(len(attribute)):
        if attribute[index] >= splitValue:
            """
            right Node
            If key not in dicitonary, add 1 for key otherwise add one to the 
            last number, Keeps count of the dictionary for right and left.
            """
            if target[index] not in rightDict:
                rightDict[target[index]] = 1
            else:
                rightDict[target[index]] += 1
            totalRight += 1
        else:
            #left Node
            if target[index] not in leftDict:
                leftDict[target[index]] = 1
            else:
                leftDict[target[index]] += 1
            totalLeft +=1
    return leftDict, totalLeft, rightDict, totalRight

"""
Function calulates and returns entropy of a node
"""
def calculateEntropy(classCount, total, classValues):
    if len(classCount) <= 1:
        Entropy = 0 # if all values are part of no or one class
    else:
        p0 = classCount[classValues[0]]/total # percent of class 0
        p1 = classCount[classValues[1]]/total # percent of class 1
        Entropy = -(p0*m.log2(p0)+p1*m.log2(p1))
    return Entropy

"""
Function calculates the weighted entropy for the split value
return 
weightedEntropy: has to be minimum for best split
index: provides
"""
def calculateWeightedEntropy(splitValue, attribute, target, classValues):
    """
    Finds count for splitValue to be used for entropy calculation
    """
    leftDict, totalLeft, rightDict, totalRight = findCount(splitValue, 
                                                           attribute, target)
    total = totalLeft + totalRight # total number of values
    
    # calculates entropy for the left node
    leftEntropy = calculateEntropy(leftDict, totalLeft, classValues)
    # calculates entropy for the right node
    rightEntropy = calculateEntropy(rightDict, totalRight, classValues)
    
    weightedEntropy = (totalLeft/total*leftEntropy)+(totalRight/total*\
                      rightEntropy)
    return weightedEntropy


"""
Traverses through all values to find the best split of a attribute
"""
def findBestSplit(attribute, target, classValues):
    binSize = 1 # identify the range of values to be considered
    """
    minimum and maximum values define the range in which splits should
    be checked, Taking both floor and ceiling so that no values should slip
    """
    minAttributeValue = m.floor(min(attribute))
    maxAttributeValue  = m.ceil(max(attribute))
    minWeightedEntropy = m.inf # helps compute the minimum entropy of the attibute
    
    """
    split value of data to find count of classess 
    """
    for splitValue in np.arange(minAttributeValue,maxAttributeValue+binSize,
                                binSize):
        weightedEntropy = calculateWeightedEntropy(splitValue, attribute, 
                                                   target, classValues)
        if weightedEntropy < minWeightedEntropy:
            minWeightedEntropy = weightedEntropy
            minSplitValue = splitValue # provides the split value for which 
                                                    # minimum entropy was found
    return minWeightedEntropy, minSplitValue
    

"""
Traverses all attributes to find the attribute producing next best split
returns
splitAttributeName: attribute on which dataset has to be split
minSplitValue: 
"""
def findBestSplitAttribute(dataset,target,targetName,classValues):    
    minWeightedEntropy = m.inf # helps compute the minimum entropy of the attibute
    
    for attributeName in dataset:
        if attributeName != targetName:
            attribute = dataset[attributeName].values.tolist()
            weightedEntropy, splitValue = findBestSplit(attribute,target,
                                                        classValues)
            if weightedEntropy < minWeightedEntropy:
                minWeightedEntropy = weightedEntropy
                minSplitValue = splitValue # provides the split value for which 
                                                    # minimum entropy was found
                
                splitAttributeName = attributeName # provides the split attribute 
                                    # corresponding to the minWeightedEntropy
        
    return splitAttributeName, minSplitValue
   
    
"""
Creates a decision Tree based on the dataframe provided
"""
def decisionTree(dataset,targetName,currentDepth,treeStructure):
    target = dataset[targetName].values.tolist()#target column in a list format
    classCount, classValues = findClassCount(dataset,target)
    
    criteriaMet, returnClass = stoppingCriteria(classCount, len(target),
                                                currentDepth)
    if criteriaMet:
        # Yes
        """
        If stopping criteria is met returns the value of majority class
        """
        treeStructure.append([returnClass,None,currentDepth])
        
    else:
        # No
        splitAttributeName, minSplitValue = findBestSplitAttribute(dataset,
                                                target,targetName,classValues)
        
        """
        Using dataframe.loc dividing the dataset into 2 parts
        leftDataset Contains less than values of minSplitValue
        rightDataset Contains greater than and equal to values of minSplitValue
        """
        treeStructure.append([splitAttributeName, minSplitValue,currentDepth])
        leftDataset = dataset.loc[dataset[splitAttributeName]<minSplitValue]
        rightDataset = dataset.loc[dataset[splitAttributeName]>=minSplitValue]
        
        # recursion on leftDataset, dropClassValues is not required
        treeStructure, dropClassValues = decisionTree(leftDataset,targetName,
                                                currentDepth-1,treeStructure)
        # recursion on rightDataset 
        treeStructure, dropClassValues = decisionTree(rightDataset,targetName,
                                                currentDepth-1,treeStructure)
    return treeStructure, classValues

# generates classifier program
def writeFile(treeStructure,classValues):
    f = open("HW06_Mehta_Aseem_Classifier.py","w+")
    f.write("\"\"\"\nHW06_720_DecTrees_2191\nAuthor: Aseem Mehta\nemail: "\
            "am1435@rit.edu\nProgram contains a classifier produced by another"\
            " program and uses the classifier\nto predict the values for the"\
            " testing dataset\n\"\"\"\n")
    
    f.write("\n# Importing the libraries\nimport pandas as pd # libtrary used"\
            " to read CSV file\n\n\"\"\"\nImports the data set using pandas"\
            " library into a dataframe\nreturn: the testing dataset in"\
            " dataframe format\n\"\"\"\n")
    f.write("def read(fileName):\n\tdataset = pd.read_csv(fileName)\n\t"\
    "return dataset\n\n\"\"\"\nCompares test Value with Split Criteria and"\
    " returns the next Side of the tree\n\"\"\"\ndef compare(testValue, "\
    "splitCriteria):\n\tif testValue < splitCriteria:\n\t\tnextSide = \"LEFT\""\
    "\n\telse:\n\t\tnextSide = \"RIGHT\"\n\treturn nextSide\n\n")
    f.write("\"\"\"\nBased on the classifier assigns values to dProgram ata in testing"\
            " File\n\"\"\"\ndef AssignValue(dataset, treeStructure):\n\n\t"\
    "values = [] # stores all the predicted values\n\n\t# traverses through"\
    " all the indexes in the test set\n\tfor index in range(0,len(dataset)):\n")
    
    f.write("\n\t\tnode = 0 # Starting point of treeStructure (pointer)\n\t\t"\
            "if treeStructure[node][1] is None:# None specifies it is a Leaf Node"\
                "\n\t\t\t\t\t\t\t\t\t\t\t# add Class Value in the values list\n\t\t\t"\
                "values.append(treeStructure[node][0])\n\t\telse:\n\t\t\t"\
                "# Checks criteria for the node if it is not a leaf node,"\
                " comparision\n\t\t\t# informs which side of tree we have"\
                " to traverse\n\t\t\tnextSide = compare(dataset.iloc[index]"\
                "[treeStructure[node][0]],treeStructure[node][1])\n\t\t\t\"\"\""\
                "\n\t\t\tnextDepth informs which nodes to check next, depth"\
                " 10 can have\n\t\t\tmaximum of subsequent nodes at depth 9"\
                "\n\t\t\t\"\"\"\n\t\t\tnextDepth = treeStructure[node][2]-1\n\n"\
                "\t\t\tnode = 1\n\t\t\t# traverses through the rest of tree"\
                " structure\n\t\t\twhile(node < len(treeStructure)):\n")
    
    f.write("\n\t\t\t\t# Checks depth of nodes in treeStructure with the"\
            " nextDepth to\n\t\t\t\t# only check the subsequent nodes\n\t\t\t\t"\
            "if nextDepth == treeStructure[node][2]:\n\t\t\t\t\tif nextSide "\
            "== \"LEFT\":\n\t\t\t\t\t\tif treeStructure[node][1] is None:"\
            " # Leaf Node\n\t\t\t\t\t\t\tvalues.append(treeStructure[node][0])"\
            "\n\t\t\t\t\t\t\tbreak\n\t\t\t\t\t\telse:\n\t\t\t\t\t\t\t\"\"\""\
            "\n\t\t\t\t\t\t\tChecks criteria for the node if it is not a leaf"\
            "\n\t\t\t\t\t\t\tnode, comparision informs which side of tree"\
            "\n\t\t\t\t\t\t\tprogram has to traverse\n\t\t\t\t\t\t\t\"\"\""\
            "\n\t\t\t\t\t\t\tnextSide = compare(dataset.iloc[index]"\
            "[treeStructure[node][0]],treeStructure[node][1])\n\t\t\t\t\t\t\t"\
            "nextDepth = treeStructure[node][2]-1\n")            
    f.write("\n\t\t\t\t\telse:\n\t\t\t\t\t\t# nextSide is Right\n\t\t\t\t\t\t"\
            "onlyRight = False # states if only right side is available"\
            "\n\t\t\t\t\t\ttempNode = node # pointer to probe for the right Node\n"\
            "\n\t\t\t\t\t\t# Skipping left side of tree as In preorder left side"\
            "\n\t\t\t\t\t\t# will always come first\n\t\t\t\t\t\ttempNode  += 1"\
            "\n\t\t\t\t\t\t# traverses through the rest of the tree to find"\
            "\n\t\t\t\t\t\twhile(tempNode < len(treeStructure) and treeStructur"\
            "e[tempNode][2]!=nextDepth):\n\t\t\t\t\t\t\ttempNode += 1\n\t\t\t\t\t\t\t"\
            "\"\"\"\n\t\t\t\t\t\t\tIf loop doesn't find another node with same"\
            " depth\n\t\t\t\t\t\t\tthen node had only one child node as right"\
            " node\"\"\"\n\t\t\t\t\t\t\tif tempNode >= len(treeStructure):"\
            "\n\t\t\t\t\t\t\t\tonlyRight = True\n\t\t\t\t\t\t# if node had 2 "\
            "child nodes, using the second one\n")
    f.write("\t\t\t\t\t\tif not onlyRight:\n\t\t\t\t\t\t\tnode = tempNode"\
            "\n\t\t\t\t\t\tif treeStructure[node][1] is None: # Leaf Node"\
                "\n\t\t\t\t\t\t\tvalues.append(treeStructure[node][0])"\
                "\n\t\t\t\t\t\t\tbreak\n\t\t\t\t\t\telse:\n\t\t\t\t\t\t\t"\
                "# Checks depth of nodes in treeStructure"\
                " with the\n\t\t\t\t\t\t\t# nextDepth to only check the"\
                " subsequent nodes\n\t\t\t\t\t\t\tnextSide = "\
                "compare(dataset.iloc[index][treeStructure[node][0]],treeStr"\
                "ucture[node][1])\n\t\t\t\t\t\t\tnextDepth = treeStructu"\
                "re[node][2]-1\n\t\t\t\tnode+=1\n\treturn values\n\n")
                
    f.write("# creates a csv file where results are stored\ndef writepredict"\
            "ion(values, classValues ):\n\tf = open(\"HW06_Mehta_Aseem_My"\
            "Classifications.csv\",\"w+\")\n\tfor currentValue in values:\n\t\t"\
            "if classValues[1] == currentValue:\n\t\t\tf.write('1\\n')\n\t\t"\
            "else:\n\t\t\tf.write('0\\n')\n\tf.close()\n")
    f.write("\n\"\"\"\nStart point of program, runs the program in a defined"\
            " sequence\n\"\"\"\ndef main():\n\ttestingFileName = "\
            "'DT_Data_CakeVsMuffin_v012_TEST.csv'\n\n\t\"\"\"\n\tdefines the"\
            " tree order traversal and values in preorder\n\tTreeStructure"\
            " consists of a list of nodes, these nodes define every node of"\
            "\n\tthe program's classifier tree. \n\tStructure of node\n\t"\
            "e.g. 1 ['Flour', 5, 10], Flour is the Attribute to check, 5 is"\
            " the\n\tsplitting criteria and 10 is the tree level in "\
            "desecnding order, starting \n\tat 10 (see figure in"\
            " writeup)\n\te.g. 2 ['CupCake', None, 7], None specifies it is"\
            " a leaf node, CupCake is \n\tthe return class and 7 is the "\
            "level of the tree.\n\t\"\"\"\n")
    f.write("\n\ttreeStructure = "+str(treeStructure))
    
    f.write("\n\tdataset = read(testingFileName)\n\tvalues = AssignValu"\
            "e(dataset, treeStructure)\n\tclassValues = "+str(classValues)+"\n"\
            "\n\twriteprediction(values, classValues)\n\n")
    f.write("if __name__ == '__main__':\n\tmain()")
    f.close()
    pass
        

"""
Start point of program, runs the program in a defined sequence
"""
def main():
    # Given Information: training Dataset and target Name
    fileName = 'DT_Data_CakeVsMuffin_v012_TRAIN.csv'
    targetName = 'RecipeType'
    
    maxDepth = 10 # maximum depth a tree can traverse to
    treeStructure = [] # defines the tree order traversal and values in 
                                                                    #preorder
    
    dataset = read(fileName)
    treeStructure, classValues = decisionTree(dataset,targetName,maxDepth,
                                              treeStructure)
    writeFile(treeStructure, classValues)

if __name__ == "__main__":
    main()