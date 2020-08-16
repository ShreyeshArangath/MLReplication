#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 23:55:17 2020

@author: shreyesh
"""

import numpy as np
import os
import pandas as pd

dataPath = os.path.join(os.getcwd(), "Data")


def _getDataFilePath(fileName):
    return os.path.join(dataPath + os.sep, fileName)


def extractAllRelevantPasswords(rockYouDataframe, relevantDigraphDataframe):
    uniqueDigraphs= set(relevantDigraphDataframe['digraph'])
    relevantRockYouPasswords = []
    
    for password in rockYouDataframe['password']: 
        i = 0
        foundAllDigraphsInThePassword = True
        while (i<len(password)-1):
            currentDigraph = password[i:i+2]
            if currentDigraph not in uniqueDigraphs: 
                foundAllDigraphsInThePassword = False
                break
            i+=1
        if foundAllDigraphsInThePassword: 
            relevantRockYouPasswords.append(password) 
        break
    return relevantRockYouPasswords
    

rockyou = pd.read_csv(_getDataFilePath("rockyou8subset.csv"))
digraphs = pd.read_csv(_getDataFilePath("uniqueDigraphs.csv"))
del rockyou['Unnamed: 0']

relevantRockYouPasswords = extractAllRelevantPasswords(rockyou, digraphs)
