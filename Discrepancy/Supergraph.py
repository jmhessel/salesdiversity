import subprocess,glob,math,sys,os
from itertools import *
from scipy.stats import entropy
from math import ceil,log
from random import *
from time import time

import numpy as np
from numpy.random import choice

class SuperGraph:
  def __init__(self, table, l, r, m):
    self.table, self.l, self.r, self.m = table, l, r, m
    self.cdist = {}
    self.adist = {}

  def filter(self,threshold=0,top_k=None):
    '''filters the supergraph to include the either the top k recommendations or the 
       recommendations which have relevance above threshold'''
    new_table = {}
    m = 0
    if top_k is not None:
      for u in xrange(1,self.l+1):
        new_table[u] = self.table[u][:top_k]
        m += top_k
    else:
      for u in xrange(1,self.l+1):
        new_table[u] = self.table[u][:10]
        v = 11
        while v<len(self.table[u]) and self.table[u][v][1]>=threshold:
          new_table[u].append(self.table[u][v])
          v += 1
        m += v-1
    return SuperGraph(new_table,self.l,self.r,m)

  def printOut(self, fn_out):
    fn_out = open(fn_out,'w')
    for u in xrange(1,self.l+1):
      print >>fn_out, '\n'.join('%d\t%d\t%.4f'%(u,v,rating) for v,rating in self.table[u])

  def uniformC(self,c):
    '''produces the uniform display constraints'''
    if c not in self.cdist:
     self.cdist[c] = [0] + [c]*self.l
    return self.cdist[c]

  def uniformA(self,c):
    '''produces the uniform target degree distribution'''
    aa = 1.0*c*self.l/self.r
    A = [0] + [aa]*(self.r)
    return A

  def proportionalA(self,c):
    '''produces the degree distribution that is proportional to the supergraph's distribution'''
    A = [0]*(self.r+1)
    for u in xrange(1,self.l+1):
      for v,rating in self.table[u]:
        A[v] += 1

    sumA = sum(A)
    for v in xrange(1,self.r+1):
      A[v] *= (1.0 * (c*self.l)) / sumA
    return A

  def proportionalAtoTop(self,c):
    '''produces a target distribution that is proportional to the top c recommendations'''
    A = [0]*(self.r+1)
    for u in xrange(1,self.l+1):
      for v,rating in islice(self.table[u],0,c):
        A[v] += 1

    sumA = sum(A)
    for v in xrange(1,self.r+1):
      A[v] *= (1.0 * (c*self.l)) / sumA
    return A

  def allOnesA(self):
    '''produces the aggregate diversity target distribution'''
    A = [1]*(self.r+1)
    A[0] = 0
    return A

  def convexCombinationA(self,c,alpha=0):
    '''produces a blend of the uniform and proportional target distributions'''
    if (c,alpha) not in self.adist:
      A1 = self.proportionalA(c)
      A2 = self.uniformA(c)
      A3 = [a1*alpha+a2*(1-alpha) for a1,a2 in izip(A1,A2)]
      self.adist[c,alpha] = self.roundDistribution(A=A3,target=self.l*c)

    return self.adist[c,alpha]

  def convexCombinationAtoTop(self,c,alpha=0):
    '''produces a blend of the uniform and proportional target distributions'''
    if (c,alpha) not in self.adist:
      A1 = self.proportionalAtoTop(c)
      A2 = self.uniformA(c)
      A3 = [a1*alpha+a2*(1-alpha) for a1,a2 in izip(A1,A2)]
      self.adist[c,alpha] = self.roundDistribution(A=A3,target=self.l*c)
    return self.adist[c,alpha]

  def roundDistribution(self,A,target):
    '''round the degree distribution to integers'''
    def findCap(A,target,lo=0,hi=1000):
      if hi==lo+1: return hi
      cap = (lo+hi)/2
      newsum = sum(min(A[v],cap) for v in xrange(1,self.r+1))
      if newsum<=target:
        return findCap(A,target,cap,hi)
      else:
        return findCap(A,target,lo,cap)

    for v in xrange(1,self.r+1):
      A[v] = max(1,int(math.ceil(A[v])))
    
    total = sum(A)
    v = 1
    while 1:
      if v == self.r: v = 1
      if total == target: break
      if A[v] > 1:
        total -= 1
        A[v] -=1
      v += 1

    return A

  def export(self):
    return self.table, self.l, self.r, self.m

  @staticmethod
  def readRecommendationTable(fn_in):
    '''reads a list of candidate recommendations from fn_in. assumes one recommendation per line
       and that recommendations are space separated triples of the form (user, item, rating)'''
    f_in = open(fn_in)
    table = {}
    l = r = 0

    for line in f_in:
      u,i,rating = line.split()
      u,i,rating = int(u), int(i), float(rating)
      l = max(u,l)
      r = max(i,r)
      if u in table:
        table[u].append((i,rating))
      else:
        table[u] = [(i,rating)]

    for u in xrange(1,l+1):
      if u not in table: table[u]=[]
      deficit = 10-len(table[u])
      for _ in xrange(deficit):
        table[u].append((randint(1,r),2.5))

    m = sum(len(table[u]) for u in xrange(1,l+1))
    for u,v in table.iteritems():
      v.sort(key=lambda x: -x[1])

    return SuperGraph(table,l,r,m)