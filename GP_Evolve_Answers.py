#!/usr/bin/env python3

import argparse
import random
import operator
import csv
import itertools
import numpy as np
import tkinter
import matplotlib
import matplotlib.pylab as pl
import gensim
import pylab
import scipy.spatial
import re
import warnings
warnings.filterwarnings("ignore")


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.metrics.pairwise import cosine_similarity
import time

# List of the the 13 groups of questions in which the original file questions-words.txt was split
# Available from https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt
list_fnames = ['capitals-common-countries_0.txt','capital-world_1.txt','currency_2.txt','city-in-state_3.txt','family_4.txt','gram1-adjective-to-adverb_5.txt','gram2-opposite_6.txt','gram3-comparative_7.txt','gram4-superlative_8.txt','gram5-present-participle_9.txt','gram6-nationality-adjective_10.txt','gram7-past-tense_11.txt','gram8-plural_12.txt','gram9-plural-verbs_13.txt']


####################################################################################################
# Definition of auxiliary functions for computation of fitness functions F1 and F2

def mycorrelation(x1, x2):    
    ans = np.corrcoef(x1,x2)    
    return ans[0][1]

def mycosine(x1, x2):
    x1 = x1.reshape(1,-1)
    ans = cosine_similarity(x1, x2)   
    return ans[0][0]


####################################################################################################
# Read the file with the questions according to the type_problem (between 1 and 13)

def Read_Questions(type_problem):
 file_name = list_fnames[type_problem]

 with open(file_name) as f:
    text = f.read()
 words = text.split()
 return words


####################################################################################################
# Definition of the fitness functions
####################################################################################################

#  Objective function that interrogates the model to determine the closest word  to a given vector
# produced by the GP program. That word is taken as the answer to the question. Function F0 in the paper

def Compute_Portion_Question_Accuracy(gp_program):
    aux_range =  range(0,int(n_word_questions_train),4)   
    rand_perm = np.random.permutation(number_questions_train)
    count = 0
    func = toolbox.compile(expr=gp_program)   
    for j in range(0,portion):
        l = aux_range[rand_perm[j]]
        i = train_set[l]
        new_vect = func(all_vectors[i], all_vectors[i+1],all_vectors[i+2])                
        top_words = model.similar_by_vector(new_vect,topn=4,restrict_vocab=Limit)        
        
        if j==0:
          if(np.array_equal(all_vectors[i],new_vect) or np.array_equal(all_vectors[i+1],new_vect) or np.array_equal(all_vectors[i+2],new_vect)):                   return(-2,)   
          old_top_words = top_words
        elif j==1:
          if top_words[0][0]==old_top_words[0][0] and top_words[0][1]==old_top_words[0][1]:
             return(-2.0,)
        elif j>max(10,portion/10) and (count/j<0.05):
             return (count/j),        
        
        for k in range(4):
          if (top_words[k][0]==new_words[i] or top_words[k][0]==new_words[i+1] or top_words[k][0]==new_words[i+2]):
            continue                              
          if top_words[k][0]==new_words[i+3]:
            count = count + 1
          break          
    #print(gp_program,count,count/portion)      
    return (count/portion),


#  Objective function based on the normalized cosine similarity measure 
#  Cosine similarity, or the cosine kernel, computes similarity as the normalized dot product of X and Y
#  Corresponds to function F1 in the paper
def word_fitness4(individual):
    sum_results = 0  
     # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)  
    
    aux_range =  range(0,int(n_word_questions_train),4)   
    rand_perm = np.random.permutation(number_questions_train)
    
    for j in range(0,portion):
        l = aux_range[rand_perm[j]]
        i = train_set[l]
        new_vect = func(all_vectors[i], all_vectors[i+1],all_vectors[i+2])  
               
        if(np.array_equal(all_vectors[i],new_vect) or np.array_equal(all_vectors[i+1],new_vect) or np.array_equal(all_vectors[i+2],new_vect)):        
           return(-2,)
        else:               
           rep_new_vect = new_vect.reshape(1,-1)  
           result = mycosine(all_vectors[i+3], rep_new_vect)
           if np.isnan(result):               
               return(-2,)                     
           sum_results += result  
           if (j==10) and (sum_results==0):             
               return(-2,)          
    return (sum_results/portion,)



#  Objective function based on the linear correlation
#  Corresponds to function F2 in the paper
def word_fitness1(individual):
   
    sum_results = 0  
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)  
    aux_range =  range(0,int(n_word_questions_train),4)   
    rand_perm = np.random.permutation(number_questions_train)    
    for j in range(0,portion):
        l = aux_range[rand_perm[j]]
        i = train_set[l]
        new_vect = func(all_vectors[i], all_vectors[i+1],all_vectors[i+2])        
       
        if(np.array_equal(all_vectors[i],new_vect) or np.array_equal(all_vectors[i+1],new_vect) or np.array_equal(all_vectors[i+2],new_vect)):        
           return(-2,)
        else:               
           result = mycorrelation(all_vectors[i+3], new_vect)
           if np.isnan(result):               
               return(-2,)                     
           sum_results += result  
           if (j==10) and (sum_results==0):             
               return(-2,)    
    return (sum_results/portion,)


##################################################################################################
# DEFINITION AND IMPLEMENTATION OF THE GP PROGRAMS
##################################################################################################


def GP_Definitions(nfeatures,nwords):
    # defined a new primitive set for strongly typed GP
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(np.ndarray, nwords), np.ndarray)

    # Corresponds to the arithmetic rule
    # Not used to evolve programs in the paper
    def KER(left, center, right):
        result = (center - left) + right       
        return result   


    # floating point operators
    # Define a safe division function
    def safeDiv(left, right): 
            while True: 
              try:
                  with np.errstate(divide='ignore'):
                       result = left / right
                       result[right == 0] = 0     
                       break                  
              except RuntimeWarning:
                   break   
            return result   

    def Diff(x):
        return 1-x 

    def ABS(x):
        return np.abs(x)  

    def Half(x):
        return 0.5*x  


    def Norm(x):   
        tmax = abs(x).max()
        if tmax==0:
          result = x
        else:
           result = x / tmax              
        return result  
  
    def Log1p(x): 
        #print(1+Norm(x))
        result = np.log1p(1+Norm(x))        
        return result  

    def Cos(x):
        return np.cos(x)  

    def Sin(x):
        return np.sin(x)  

    def Roll(x):
        return np.roll(x,1)  
  
    def Rint(x):
        return np.rint(x)

    def Rand(x):
        return np.random.rand(nfeatures)  

    def Ones(x):
        return np.ones((nfeatures))  

    def Zeros(x):
        return np.zeros((nfeatures))  

    pset.addPrimitive(operator.add, [np.ndarray,np.ndarray], np.ndarray)
    pset.addPrimitive(operator.sub,  [np.ndarray,np.ndarray], np.ndarray)
    pset.addPrimitive(operator.mul,  [np.ndarray,np.ndarray], np.ndarray)
    pset.addPrimitive(operator.neg,  [np.ndarray], np.ndarray)
 
    #pset.addPrimitive(KER,  [np.ndarray,np.ndarray,np.ndarray], np.ndarray)
    pset.addPrimitive(safeDiv,  [np.ndarray,np.ndarray], np.ndarray)
    pset.addPrimitive(Diff,  [np.ndarray], np.ndarray)
    pset.addPrimitive(ABS,  [np.ndarray], np.ndarray)
    pset.addPrimitive(Cos,  [np.ndarray], np.ndarray)
    pset.addPrimitive(Sin,  [np.ndarray], np.ndarray)
    pset.addPrimitive(Roll,  [np.ndarray], np.ndarray)
    pset.addPrimitive(Rint,  [np.ndarray], np.ndarray)
    pset.addPrimitive(Half,  [np.ndarray], np.ndarray)
    pset.addPrimitive(Norm,  [np.ndarray], np.ndarray)
    pset.addPrimitive(Log1p,  [np.ndarray], np.ndarray)
     
    return pset


#####################################################################################################
# Initialization of GP algorithm
####################################################################################################

def Init_GP_MOP(nwords):
    nfeat = number_features
    pset = GP_Definitions(nfeat,nwords)
    maxDepthLimit = 10
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    #toolbox.register("expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=1, max_=2) # IT MIGHT BE A BUG WITH THIS
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


  
    if FUNC==0:
      toolbox.register("evaluate", Compute_Portion_Question_Accuracy) # Direct query on of the closest vector F0
    elif FUNC==1:
      toolbox.register("evaluate", word_fitness4)                     # Cosine similarity distance  F1
    elif FUNC==2:
      toolbox.register("evaluate", word_fitness1)                     # Correlation as distance   F2
   
    if SEL==0:
      toolbox.register("select", tools.selBest)
    elif SEL==1:
      toolbox.register("select", tools.selTournament, tournsize=tournsel_size)
    elif SEL==2:
      toolbox.register("select", tools.selNSGA2)
    
    
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

    if MUT==0:
         toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    elif MUT==1:
         toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
    elif MUT==2:
         toolbox.register("mutate", gp.mutShrink)

    toolbox.decorate('mutate',gp.staticLimit(key=operator.attrgetter('height'),max_value=maxDepthLimit))
    toolbox.decorate('mate',gp.staticLimit(key=operator.attrgetter('height'),max_value=maxDepthLimit))

    return toolbox


#####################################################################################################
# Application of the GP algorithm
#####################################################################################################


def Apply_GP_MOP(toolbox,pop_size,gen_number,therun):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
   
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    #res, logbook = algorithms.eaSimple(pop, toolbox, 0.5, 0.5, gen_number, stats, halloffame=hof,verbose=1)

    res, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=nselpop, 
                                     lambda_=pop_size, 
                                     cxpb= CXp,
                                     mutpb=1-CXp, 
                                     stats=stats, 
                                     halloffame=hof,
                                     ngen=gen_number, 
                                     verbose=1)         

    return res, logbook, hof

#####################################################################################################
# Auxiliary function to compute accuracy in a given set (trainin, test, or whole set of questions)
#####################################################################################################

def Compute_Question_Accuracy(gp_program,nwords,theset):
    count = 0
    func = toolbox.compile(expr=gp_program)  
   
    for i in range(0,nwords,4):
        j = theset[i]                 
        new_vect = func(all_vectors[j], all_vectors[j+1],all_vectors[j+2])  
        top_words = model.similar_by_vector(new_vect,topn=4,restrict_vocab=Limit)             
        for k in range(4):       
          if (top_words[k][0]==new_words[j] or top_words[k][0]==new_words[j+1] or top_words[k][0]==new_words[j+2]):
            continue          
          if top_words[k][0]==new_words[j+3]:
            count = count + 1           
          break          
      
    return (4.0*count)/(nwords)



   

##################################################################################################
#  MAIN PROGRAM
##################################################################################################

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(3000),
         nargs='+', help='an integer in the range 0..3000')
    parser.add_argument(
        '--sum', dest='accumulate', action='store_const', const=sum,
        default=max, help='sum the integers (default: find the max)')


    # The parameters of the program are set or read from command line

    global Gen                  # Current generation 
    number_runs = 1             # Number of times that the GP program is executed

    args = parser.parse_args()
    seed = args.integers[0]              # Seed: Used to set different outcomes of the stochastic program
    corpus  =  args.integers[1]          # Which of the dictionaries of vectors (0,1,2)
    Limit  =  args.integers[2]*1000      # Number of variables to consider from the dictionary totalwords/1000
    number_features =  args.integers[3]  # Number of features. Size of the word vector  (50,100,200,300)
    ninput_words = args.integers[4]      # It coincides with the number of terminals of the program. (3)
    type_problem = args.integers[5]      # Which group of questions is addressed   (1...13)
    npop = args.integers[6]              # Population size of the GP programs      (500 and 1000 in the experiments)
    ngen = args.integers[7]              # Number of generations                   (250 and 500 in the experiments)
    FUNC = args.integers[8]              # Fitness function used by the algorithm  (0,1,2) corresponding to F0,F1,F2
    SEL  = args.integers[9]              # Selection method                        (0,1,2) See explanation above
    MUT  = args.integers[10]             # Mutation method                          (0,1,2) See explanation above
    CXp   =  args.integers[11]*0.01      # Crossover probability (Mutation is 1-CXp) (0.5 in the experiments)
    nselpop = args.integers[12]          # Selected population size                 (100 individuals in all experiment)
    tournsel_size = args.integers[13]    # Tournament value                        (5, when tested) 
      
    np.random.seed(seed)
    random.seed(seed)
    

    # The corpus tested. Corpus GoogleNews only used for transferability, not for evolving programs
    if corpus==0:
       fname = 'mymodel_text8_'+str(number_features)+'.mod'    
       model =  gensim.models.Word2Vec.load(fname)   
    elif corpus==1:      
       model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True,limit=Limit)  # C binary format
    elif corpus==2:      
       model = gensim.models.Word2Vec.load_word2vec_format('~/Dropbox/ExperimentsAndCode/word2vec-master/GoogleNews-vectors-negative300.bin.gz', binary=True)  # C binary format
   

    # The following code read the group of questions, checks if all the words are in the corpus
    # and transform all the words to their corresponding word vectors   
 
    words = Read_Questions(type_problem)
    new_words = words
    n_word_questions = len(words)
    n_questions = int(n_word_questions/4)  
    all_vectors = np.zeros((n_word_questions,number_features))
    word_count = 0
    
    for j in range(0,n_word_questions,4):
     A = 1   
     while True:
       try:
            w = words[j]           
            all_vectors[word_count,:] = model[w]
            new_words[word_count] = w
            w = words[j+1]
            all_vectors[word_count+1,:] = model[w]
            new_words[word_count+1] = w
            w = words[j+2]
            all_vectors[word_count+2,:] = model[w]
            new_words[word_count+2] = w
            w = words[j+3]
            all_vectors[word_count+3,:] = model[w]
            new_words[word_count+3] = w
            break
       except KeyError:
            A = 0           
            #word_count = max(word_count-4,0)
            #A = 1
            break        
     if A==0:              
        continue
     word_count = word_count + 4
          

    # The set of questions is divided into train and test sets with the same number of words

    if word_count>4:
      n_word_questions  = word_count-4    
      number_questions = int(n_word_questions/4)
      all_vectors = all_vectors[:n_word_questions,:]
      new_words = new_words[:n_word_questions]
      portion = int(number_questions/5)
   
      number_questions_train = int(number_questions/2)
      number_questions_test =  number_questions - number_questions_train
      n_word_questions_train = 4*number_questions_train
      n_word_questions_test =  4*number_questions_test
     
      train_set = np.array(range(0,n_word_questions_train))
      test_set =  np.array(range(n_word_questions_train,4*number_questions))
      
      # Prints all the parameters of the algorithm 
      print(list_fnames[type_problem],seed,corpus,Limit,number_features,ninput_words,type_problem,npop,ngen,FUNC,SEL,MUT,CXp,nselpop,tournsel_size,len(words),n_word_questions,n_questions,number_questions,all_vectors.shape[1])
      

      # The GP program is initialized and runs
        
      toolbox = Init_GP_MOP(ninput_words)
      for run in range(number_runs):        
        start_time = time.time()
        pop, stats, hof = Apply_GP_MOP(toolbox,npop,ngen,run)
        elapsed_time = time.time() - start_time

        # After the algorithm finishes the evolution, we print the accuracies in the train and test sets
        # for all programs in the last selected population (100 programs).
        # We also print the tree-based programs

        for i in range(0,nselpop):
           val = pop[i].fitness.values[0]         
           Acc_Train = Compute_Question_Accuracy(pop[i],n_word_questions_train,train_set)
           Acc_Test = Compute_Question_Accuracy(pop[i],n_word_questions_test,test_set)
           print("VALS ",run, i, 0.0, pop[i].fitness.values[0], Acc_Train, Acc_Test, elapsed_time)   
           print("PROG ",pop[i])
    else:
        print("The words of the questions are not in the dictionary")
        
        

# EXAMPLE OF HOW TO CALL THE PROGRAM
# ./GP_Evolve_Answers.py 111 0 30 50 3 5 500 250 0 0 0 5 100 5

  
