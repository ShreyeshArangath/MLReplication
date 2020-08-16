#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:43:51 2020

@author: shreyesh
"""
import os
import pandas as pd
import random
import unittest
import RockYouDatasetParser as parser

dataPath = os.path.join(os.getcwd(), "Data")
def _getDataFilePath(fileName):
    return os.path.join(dataPath + os.sep, fileName)

rockYouDataframe = pd.read_csv(_getDataFilePath("rockyou8subset.csv"))
relevantDigraphDataframe = pd.read_csv(_getDataFilePath("uniqueDigraphs.csv"))
uniqueDigraphs= set(relevantDigraphDataframe['digraph'])
del rockYouDataframe['Unnamed: 0']


def getDigraphs(password):
    digraphSet = set()
    i = 0
    while (i<len(password)-2):
        digraphSet.add(password[i:i+2])
        i+=1
    return digraphSet

def getPasswordDigraphsFromExtractedPasswords(rockYouDataframe, relevantDigraphDataframe):
    listToInspect = parser.extractAllRelevantPasswords(rockYouDataframe, relevantDigraphDataframe)
    randomIndex = random.randint(0, len(listToInspect)-1)
    digraphArray = getDigraphs(listToInspect[randomIndex])
    return digraphArray
    
    

class RockYouDatasetParserTests(unittest.TestCase):
    def testAllDigraphsExistInTheDataframe(self):
        digraphArray = getPasswordDigraphsFromExtractedPasswords(rockYouDataframe, relevantDigraphDataframe)
        testArray = []
        for element in digraphArray: 
            if element in uniqueDigraphs: 
                testArray.append(True)
            else: 
                testArray.append(False)
        self.assertEqual(testArray, [True for x in range(len(digraphArray))])
        

if __name__ == "__main__":
    unittest.main()