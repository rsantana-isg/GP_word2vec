# GP_word2vec
Contains the main implementation of programs for the paper: Reproducing and learning new algebraic operations on word embeddings using genetic programming (https://arxiv.org/abs/1702.05624v1).

This project implements the automatic learning of compositions of word vector representations using genetic programming. The project is based on the DEAP library that implements evolutionary algorithms (https://github.com/deap/deap) and on  the gensim library that implements natural language processing (NLP) routines (https://radimrehurek.com/gensim/). 

GP_Evolve_Answers.py programs tries to find a suitable transformation of word vectors for solving a word analogy task.  (Mikolov et al:2013,Pennington et al:2014). This task consists of answering a question as: "a is to b  as c is to ?". A correct answer is the exact word that would fit the analogy.  

The problem  to be solved by GP is, given the vector representations of the three known words, produce a vector whose closest word in the corpus is the one that correctly answers the question. 

To work, GP_Evolve_Answers.py requires a file with the questions. Each question is a list of four words like:
 boy      girl       sons    daughters
 
 Currently, GP_Evolve_Answers.py works on 13 possible files:
 
 ['capitals-common-countries_0.txt','capital-world_1.txt','currency_2.txt','city-in-state_3.txt','family_4.txt','gram1-adjective-to-adverb_5.txt','gram2-opposite_6.txt','gram3-comparative_7.txt','gram4-superlative_8.txt','gram5-present-participle_9.txt','gram6-nationality-adjective_10.txt','gram7-past-tense_11.txt','gram8-plural_12.txt','gram9-plural-verbs_13.txt']
 
 Each of these files includes one group of questions of those proposed by Mikolov and available from 
 https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt
 
 For example, to create the 'family_4.txt' file, all the questions within the 'family' group of questions in the questions-words.txt file as copied as an independent file.
 
 GP_Evolve_Answers.py also needs a word-vector embedding. The one used to generate the GP programs was created from the  text8.zip corpus, http://mattmahoney.net/dc/text8.zip. It was created with the gensim instructions:
 
sentences = gensim.models.word2vec.LineSentence('text8') 
model = gensim.models.Word2Vec(sentences, cbow_mean=1, size=300, window=8, negative=25, hs=0, sample=1e-4, iter=15, min_count=5, workers=6)
model.save(fname)

Examples to call GP_Evolve_Answers.py are given at the end of the file. 
For questions, email to Roberto Santana (roberto DOT santana AT ehu DOT es)


References

T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.

J. Pennington, R. Socher, and C. D. Manning. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), volume 14, pages 1532–1543, 2014.



