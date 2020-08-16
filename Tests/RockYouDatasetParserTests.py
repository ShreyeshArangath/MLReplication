#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:43:51 2020

@author: shreyesh
"""

import pandas as pd
import random
import unittest
from parser import RockYouDatasetParser
from path import Path 

parser = RockYouDatasetParser()
path = Path()

rockYouDataframe = pd.read_csv(path.getDataFilePath("rockyou8subset.csv"))
relevantDigraphDataframe = pd.read_csv(path.getDataFilePath("uniqueDigraphs.csv"))
uniqueDigraphs= set(relevantDigraphDataframe['digraph'])
del rockYouDataframe['Unnamed: 0']

def getPasswordDigraphsFromExtractedPasswords(rockYouDataframe, relevantDigraphDataframe):
    listToInspect = parser.extractAllRelevantPasswords(rockYouDataframe, relevantDigraphDataframe)
    randomIndex = random.randint(0, len(listToInspect)-1)
    digraphArray = parser.getDigraphs(listToInspect[randomIndex])
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