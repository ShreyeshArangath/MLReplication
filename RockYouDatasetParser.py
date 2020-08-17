#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 23:55:17 2020

@author: shreyesh
"""

import pandas as pd
from path import Path

class RockYouDatasetParser:

    def getDigraphs(self, password):
        digraphSet = list()
        i = 0
        while (i<len(password)-1):
            digraphSet.append(password[i:i+2])
            i+=1
        return digraphSet
    
    def extractAllRelevantPasswords(self, rockYouDataframe, relevantDigraphDataframe):
        uniqueDigraphs= set(relevantDigraphDataframe['digraph'])
        relevantRockYouPasswords = []
        
        for password in rockYouDataframe['password']: 
            i = 0
            digraphArray = self.getDigraphs(password)
            foundAllDigraphsInThePassword = True
            for currentDigraph in digraphArray: 
                if currentDigraph not in uniqueDigraphs: 
                    foundAllDigraphsInThePassword = False
                    break
                i+=1
            if foundAllDigraphsInThePassword: 
                relevantRockYouPasswords.append(password) 
                
        return relevantRockYouPasswords
        


if __name__=="__main__":
    parser = RockYouDatasetParser()
    path = Path()
    rockyou = pd.read_csv(path.getDataFilePath("rockyou8subset.csv"))
    digraphs = pd.read_csv(path.getDataFilePath("uniqueDigraphs.csv"))
    del rockyou['Unnamed: 0']
    
    relevantRockYouPasswords = parser.extractAllRelevantPasswords(rockyou, digraphs)
    

