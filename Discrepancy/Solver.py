import subprocess,glob,math,sys,os
from itertools import *
from scipy.stats import entropy
from math import ceil,log
from random import *
from Supergraph import *
from time import time

import numpy as np
from numpy.random import choice


class Solver:
  def __init__(self,superg):
    self.superg = superg

  def writeDMXProblem(self, C, A, lambd, mu, fn_out, stages=1, stage_amount=5):
    '''writes out the dmx problem for the bicriteria discrepancy reduction problem with
       C            as the display constraints
       A            as the target degree distribution
       lambd        as the relative weight of the discrepancy term
       mu           as the relative weight of the relevance term
       test_fn      as the a list of held out ratings which are known to be relevant 
       stages       as the number of slopes of the degree overrun penalty
       stage_amount as the length of each leg of the slopes
       fn_out       as the name of the output file'''

    INFINITY = 2*sum(C)
    f_out = open(fn_out, 'w')
    table,l,r,m = self.superg.export()

    # sanity checks
    assert A[0] == C[0] == 0
    assert all(len(table[u]) >= C[u] for u in xrange(1,l+1))
    assert len(C) == l+1
    assert len(A) == r+1
    assert sum(A) <= sum(C)

    # print 'c Problem line (nodes, links)'
    print >>f_out, 'p min %d %d'%(l+r+stages+1,m+(stages+1)*r+stages)
    
    # print 'c Node descriptor lines (supply+ or demand-)'
    for i in xrange(1,l+1):
      print >>f_out, 'n %d %d'%(i,C[i])
    print >>f_out, 'n %d %d'%(l+r+2,-sum(C))

    # print 'c Arc descriptor lines (from, to, minflow, maxflow, cost)'
    for u in xrange(1,l+1):
      for v,rating in table[u]:
        print >>f_out, 'a %d %d %d %d %.2f'%(u,l+v,0,1,-rating*mu)

    for v in xrange(1,r+1):
      print >>f_out, 'a %d %d %d %d %d'%(l+v,l+r+1,0,A[v],0)
      for stage in xrange(1,stages):
        print >>f_out, 'a %d %d %d %d %d'%(l+v,l+r+stage+1,0,stage_amount,2*stage*lambd)    
      print >>f_out,   'a %d %d %d %d %d'%(l+v,l+r+stages+1,0,INFINITY,2*stages*lambd) 

    for stage in xrange(1,stages+1):
      print >>f_out, 'a %d %d %d %d %d'%(l+r+stage,l+r+stage+1,0,INFINITY,0)

  def writeDMXProblemGoal(self, C, A, lambd, mu, fn_out, goal=None, stages=1, stage_amount=5):
    '''writes out the dmx problem for the goal programming discrepancy reduction problem with
       C            as the display constraints
       A            as the target degree distribution
       lambd        as the relative weight of the discrepancy term
       mu           as the relative weight of the relevance term
       test_fn      as the a list of held out ratings which are known to be relevant 
       stages       as the number of slopes of the degree overrun penalty
       stage_amount as the length of each leg of the slopes
       fn_out       as the name of the output file'''
    INFINITY = 2*sum(C)
    f_out = open(fn_out, 'w')
    table,l,r,m = self.superg.export()

    # sanity checks
    assert A[0] == C[0] == 0
    assert len(C) == l+1
    assert len(A) == r+1
    assert sum(A) <= sum(C)

    # print 'c Problem line (nodes, links)'
    print >>f_out, 'p min %d %d'%(l+r+stages+1,m+(stages+1)*r+stages)
    
    # print 'c Node descriptor lines (supply+ or demand-)'
    for i in xrange(1,l+1):
      print >>f_out, 'n %d %d'%(i,C[i])
    print >>f_out, 'n %d %d'%(l+r+2,-sum(C))

    # print 'c Arc descriptor lines (from, to, minflow, maxflow, cost)'
    for u in xrange(1,l+1):
      for v,rating in table[u]:
        print >>f_out, 'a %d %d %d %d %.5f'%(u,l+v,0,1,-rating)

    for v in xrange(1,r+1):
      print >>f_out, 'a %d %d %d %d %d'%(l+v,l+r+1,0,A[v],0)
      print >>f_out, 'a %d %d %d %d %d'%(l+v,l+r+2,0,INFINITY,0) 

    for stage in xrange(1,stages+1):
      print >>f_out, 'a %d %d %d %d %d'%(l+r+stage,l+r+stage+1,goal,INFINITY,0)

  def writeDMXProblemCategory(self, C, A, lambd, mu, fn_out, categories, categoryTargets):
    '''category is a dictionary mapping vertices to their category id [1..k]'''
    '''category targets is a list denoting the target for each category'''
    if not goal: goal = sum(C)
    INFINITY = sum(C)
    f_out = open(fn_out, 'w')
    table,l,r,m = self.superg.export()

    num_categories = len(categories)
    collector = l+r+3*num_categories+1
    supersink = l+r+3*num_categories+2
    assert A[0] == C[0] == 0
    assert len(C) == l+1
    assert len(A) == r+1
    assert sum(A) <= sum(C)

    # print 'c Problem line (nodes, links)'
    print >>f_out, 'p min %d %d'%(l+r+3*num_categories+2,m+2*r+4*num_categories+2)
    
    # print 'c Node descriptor lines (supply+ or demand-)'
    for i in xrange(1,l+1):
      print >>f_out, 'n %d %d'%(i,C[i])
    for category in xrange(1,num_categories+1):
      print >>f_out, 'n %d %d'%(l+r+3*category+2,categoryTargets[category])
    print >>f_out, 'n %d %d'%(supersink,l*c-sum(categoryTargets))

    # print 'c Arc descriptor lines (from, to, minflow, maxflow, cost)'
    for u in xrange(1,l+1):
      for v,rating in table[u]:
        print >>f_out, 'a %d %d %d %d %.2f'%(u,l+v,0,1,rating)

    for v in xrange(1,r+1):
      print >>f_out, 'a %d %d %d %d %d'%(l+v,l+r+3*category[v]  ,0,INFINITY,0)
      print >>f_out, 'a %d %d %d %d %d'%(l+v,l+r+3*category[v]+1,0,INFINITY,1)

    for category in xrange(1,num_categories+1):
      print >>f_out, 'a %d %d %d %d %d'%(l+r+3*category+1,l+r+3*category+2,categoryTargets[category],0)
      print >>f_out, 'a %d %d %d %d %d'%(l+r+3*category+1,collector       ,INFINITY                 ,0)
      print >>f_out, 'a %d %d %d %d %d'%(collector       ,l+r+3*category+2,INFINITY                 ,1)
      print >>f_out, 'a %d %d %d %d %d'%(l+r+3*category+2,supersink       ,INFINITY                 ,0)

    print >>f_out, 'a %d %d %d %d %d'%(collector,supersink,0,INFINITY,0)

  def degree_distribution(self, edges):
    '''calculates the degree distribution of the edge list edges'''
    table,l,r,m = self.superg.export()
    AA = [0]*(r+1)
    for u,v in edges:
      AA[v-l] += 1
    return np.array(AA)

  def entropy(self, arr):
    '''calculates the entropy of the normalized degree distribution'''
    return entropy(arr)

  def gini(self, array):
    '''calculates the gini index of the degree distribution'''
    sorted_list = sorted(array)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(array) / 2.
    return (fair_area - area) / fair_area

  def readOutput(self, dmx_fn):
    '''calls MCFSolve to solve the dmx problem and reads the set of edges'''
    start = time()
    cmd  = ['./MyMCFSolve', dmx_fn]
    g1 = subprocess.Popen( cmd, stdout=subprocess.PIPE ).communicate()[0].split('\n')
    duration = time()-start
    g2 = [g1[x].split() for x in xrange(2,len(g1)-2) if g1[x]]
    g3 = {(int(u),int(i)) for u,i in g2 if u.isdigit() and i.isdigit() and int(i)<=self.superg.l+self.superg.r}
    return g3,duration

  def solutionMetrics(self,C,A,edges,test_fn):
    '''calculates the aggregate diversity, precision and discrepancy of the solutions described by edges'''
    table,l,r,m = self.superg.export()
    ratings  = 0.0
    AA = [0]*(r+1)
    
    tests = set()
    for line in open(test_fn):
      u,v = map(int, line.split()[:2])
      tests.add((u,v+l))

    for u,v in edges:
      AA[v-l] += 1

    ctr = 0
    for u in xrange(1,l+1):
      for v,rating in table[u]:
        if (u,l+v) in edges:
          ratings += rating

    PRECISION = len(tests & edges)*1. / (len(set(u for u,v in tests)) * C[1])
    AGGDIV = sum(a>0 for a in AA) * 1. / r
    return sum(abs(A[i]-AA[i]) for i in xrange(1,r+1)), ratings/sum(C), AGGDIV, PRECISION

  def lowerBound(self,A):
    '''calculates the trivial lower bound on discrepancy given the edges of the supergraph'''
    table,l,r,m = self.superg.export()
    AA = [0]*(r+1)
    for u in table:
      for v,rating in table[u]:
        AA[v] += 1
    return 2*sum(max(0,A[i]-AA[i]) for i in xrange(1,r+1))

  def solveGreedy(self,C,A,test_fn):
    '''implements the greedy algorithm'''
    start = time()
    table,l,r,m = self.superg.export()
    edges = set()
    deg_dist = np.zeros(r+10)
    AA = self.superg.proportionalA(C[1])

    order = range(1,l+1)
    shuffle(order)
    for u in order:
      discrepancy_reducers = [v      for v,rating in table[u] if deg_dist[v] <A[v]]
      d_weights            = [rating for v,rating in table[u] if deg_dist[v] <A[v]]
      sum_d_weights        = sum(d_weights)
      d_weights            = [rating*1./sum_d_weights for rating in d_weights]
      
      rest                 = [v      for v,rating in table[u] if deg_dist[v] >=A[v]]
      r_weights            = [rating for v,rating in table[u] if deg_dist[v] >=A[v]]
      sum_r_weights        = sum(r_weights)
      r_weights            = [rating*1./sum_r_weights for rating in r_weights]


      neighbors = []
      if len(discrepancy_reducers) >= C[u]:
        neighbors  = list(choice(discrepancy_reducers, replace=False, size=C[u]        , p=d_weights))
      else:
        left_over = C[u] - len(discrepancy_reducers)
        neighbors  = discrepancy_reducers
        neighbors += list(choice(rest                , replace=False, size=left_over   , p=r_weights))

      for v in neighbors:
        edges.add((u,v+l))
        deg_dist[v] += 1

    duration = time()-start
    GINI = self.gini(deg_dist)
    ENT = self.entropy(deg_dist/(1.*sum(C)))
    SOLD, SOLR, SOLA, P = self.solutionMetrics(C, A, edges, test_fn)
    self.edges = edges
    return (SOLD/2./sum(C), SOLR, SOLA, P, GINI, ENT, deg_dist, duration)

  def solveExternal(self, C, A, train_in, rec_in, test_fn, script_name="FD"):
    '''calls an external script to diversify recommendations'''
    '''the external script must take as input a single file of candidate recommendations'''
    '''the cmd variable below must be changed to the path of the external solver'''

    rec_in = rec_in[:-4]
    dummy_rec_in = rec_in + '_.txt'
    self.superg.printOut(dummy_rec_in)

    start = time()


    cmd = "CHANGE THIS TO THE PATH OF THE SCRIPT YOU WOULD LIKE TO USE"
    cmd = ' '.join([cmd, train_in, rec_in + '_', script_name])

    subprocess.call(cmd, shell=True)
    duration = time()-start
    fn_in  = '%s__%s.txt'% (rec_in, script_name)

    table,l,r,m = self.superg.export()

    user_deg = [0]*(l+1)
    deg_dist = np.zeros(r+1)
    edges = set()
    for line in open(fn_in):
      u,v,_ = line.strip().split()
      u = int(u)
      v = int(v)
      if user_deg[u] < C[u]:
        edges.add((u,v+l))
        user_deg[u] += 1
        deg_dist[v] += 1

    GINI = self.gini(deg_dist)
    ENT = self.entropy(deg_dist/(1.*sum(C)))
    SOLD, SOLR, SOLA, P = self.solutionMetrics(C, A, edges, test_fn)
    return (SOLD/2./sum(C), SOLR, SOLA, P, GINI, ENT, deg_dist, duration)

  def solveStandard(self, C, A, test_fn):
    table,l,r,m = self.superg.export()
    
    edges = set()
    for u in xrange(1,l+1):
      for i in xrange(C[u]):
        edges.add((u,l+table[u][i][0]))

    SOLD, SOLR, SOLA, P = self.solutionMetrics(C, A, edges, test_fn)
    deg_dist         = self.degree_distribution(edges)
    GINI             = self.gini(deg_dist)
    ENT              = self.entropy(deg_dist/(1.*sum(C)))
    self.edges = edges
    return (SOLD/2./sum(C), SOLR, SOLA, P, GINI, ENT, deg_dist, 0)

  def solve(self, C, A, dmx_fn, test_fn, lambd=1, mu=0, stages=1, stage_amount=5):
    '''solves the recommendation problem encoded by dmx_fn using the bicriteria method
       C            as the display constraints
       A            as the target degree distribution
       lambd        as the relative weight of the discrepancy term
       mu           as the relative weight of the relevance term
       test_fn      as the a list of held out ratings which are known to be relevant 
       stages       as the number of slopes of the degree overrun penalty
       stage_amount as the length of each leg of the slopes
       fn_out       as the name of the output file'''

    l,r = self.superg.l, self.superg.r
    self.writeDMXProblem(C, A, lambd, mu, dmx_fn, stages, stage_amount)
    edges,duration   = self.readOutput     (dmx_fn)
    deg_dist         = self.degree_distribution(edges)
    LOWERBOUND       = self.lowerBound     (  A)
    SOLD, SOLR, SOLA, P = self.solutionMetrics(C,A,edges,test_fn)
    GINI             = self.gini(deg_dist)
    ENT              = self.entropy(deg_dist/(1.*sum(C)))
    self.edges = edges
    return (SOLD/2./sum(C), SOLR, SOLA, P, GINI, ENT, deg_dist, duration)

  def solveWithGoal(self, C, A, dmx_fn, test_fn, lambd=1, mu=0, stages=1, stage_amount=5):
    '''solves the recommendation problem encoded by dmx_fn using the goal programming method
       C            as the display constraints
       A            as the target degree distribution
       lambd        as the relative weight of the discrepancy term
       mu           as the relative weight of the relevance term
       test_fn      as the a list of held out ratings which are known to be relevant 
       stages       as the number of slopes of the degree overrun penalty
       stage_amount as the length of each leg of the slopes
       fn_out       as the name of the output file'''

    l,r = self.superg.l, self.superg.r
    self.writeDMXProblem(C, A, 1, 0, dmx_fn, stages, stage_amount)
    edges,duration1  = self.readOutput     (dmx_fn)
    LOWERBOUND       = self.lowerBound     (  A)
    SOLD, SOLR, SOLA, P = self.solutionMetrics(C,A,edges,test_fn)
    OLDSOLD = int(SOLD * 2*sum(C) + 1)

    self.writeDMXProblemGoal(C, A, 0, 1, dmx_fn, sum(C)-SOLD/2-1, stages, stage_amount)
    edges,duration2  = self.readOutput     (dmx_fn)
    deg_dist         = self.degree_distribution(edges)
    SOLD, SOLR, SOLA, P  = self.solutionMetrics(C,A,edges,test_fn)
    GINI             = self.gini(deg_dist)
    ENT              = self.entropy(deg_dist/(1.*sum(C)))
    self.edges = edges
    return (SOLD/2./sum(C), SOLR, SOLA, P, GINI, ENT, deg_dist, duration1+duration2)
