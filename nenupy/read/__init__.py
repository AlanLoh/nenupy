#! /usr/bin/python3.5
# -*- coding: utf-8 -*-

from .SST import SST
from .BST import BST
from .XST import XST

__all__ = ['SST', 'BST', 'XST', 'FakeObs']

class FakeObs():
    def __init__(self):
        self.freq  = 50
        self.azana = 180 
        self.elana = 90
        self.azdig = 180 
        self.eldig = 90
        self.polar = 'NW'
        # self.time  = 
