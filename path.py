#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 19:47:12 2020

@author: shreyesh
"""
import os

class Path: 
    def __init__(self): 
        self.dataPath = os.path.join(os.getcwd(), "Data")
        
    def getDataFilePath(self,fileName):
        return os.path.join(self.dataPath + os.sep, fileName)