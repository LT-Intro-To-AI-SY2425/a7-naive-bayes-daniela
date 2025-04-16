import math, os, pickle, re
from typing import Tuple, List, Dict
 
 
class BayesClassifier:
     """A simple BayesClassifier implementation
 
     Attributes:
         pos_freqs - dictionary of frequencies of positive words
         neg_freqs - dictionary of frequencies of negative words
         pos_filename - name of positive dictionary cache file
         neg_filename - name of positive dictionary cache file
         training_data_directory - relative path to training directory
         neg_file_prefix - prefix of negative reviews
         pos_file_prefix - prefix of positive reviews
     """
 
     def __init__(self):
         """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
         cache of a trained classifier is stored in the current folder it is loaded,
         otherwise the system will proceed through training.  Once constructed the
         classifier is ready to classify input text."""
         # initialize attributes
         self.pos_freqs: Dict[str, int] = {}
         self.neg_freqs: Dict[str, int] = {}
         self.pos_filename: str = "pos.dat"
         self.neg_filename: str = "neg.dat"
         self.training_data_directory: str = "movie_reviews/"
         self.neg_file_prefix: str = "movies-1"
         self.pos_file_prefix: str = "movies-5"
 
         # check if both cached classifiers exist within the current directory
         if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
             print("Data files found - loading to use cached values...")
             self.pos_freqs = self.load_dict(self.pos_filename)
             self.neg_freqs = self.load_dict(self.neg_filename)
         else:
             print("Data files not found - running training...")
             self.train()
 
     def train(self) -> None:
         """Trains the Naive Bayes Sentiment Classifier
 
         Train here means generates `pos_freq/neg_freq` dictionaries with frequencies of
         words in corresponding positive/negative reviews
         """
         # get the list of file names from the training data directory
         # os.walk returns a generator (feel free to Google "python generators" if you're
         # curious to learn more, next gets the first value from this generator or the
         # provided default `(None, None, [])` if the generator has no values)
         _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
         if not files:
             raise RuntimeError(f"Couldn't find path {self.training_data_directory}")
 
         # files now holds a list of the filenames
         # self.training_data_directory holds the folder name where these files are
 
 
         # stored below is how you would load a file with filename given by `filename`
         # `text` here will be the literal text of the file (i.e. what you would see
         # if you opened the file in a text editor
         # text = self.load_file(os.path.join(self.training_data_directory, files))
 
 
         # *Tip:* training can take a while, to make it more transparent, we can use the
         # enumerate function, which loops over something and has an automatic counter.
         # write something like this to track progress (note the `# type: ignore` comment
         # which tells mypy we know better and it shouldn't complain at us on this line):
 
         file  = self.load_file("sorted_stoplist.txt")
         stopwords = self.tokenize(file)
 
         for index, filename in enumerate(files, 1): # type: ignore
             print(f"Training on file {index} of {len(files)}")
         #     <the rest of your code for updating frequencies here>
             text = self.load_file(os.path.join(self.training_data_directory, filename))
             token = self.tokenize(text)
             tokens = self.tokenize(text)
 
             filter_tokens = [token for token in tokens if token not in stopwords]
 
             if filename.startswith(self.pos_file_prefix):
                 self.update_dict(token, self.pos_freqs)
                 self.update_dict(tokens, self.pos_freqs)
             elif filename.startswith(self.neg_file_prefix):
                 self.update_dict(token, self.neg_freqs)
                 self.update_dict(tokens, self.neg_freqs)
 
         # we want to fill pos_freqs and neg_freqs with the correct counts of words from
         # their respective reviews
               
         tokens = self.tokenize(text)
 
 
         file = self.load_file("sorted_stoplist.txt")
         stopwords = self.tokenize(file)
 
         # create some variables to store the positive and negative probability. since
         # we will be adding logs of probabilities, the initial values for the positive
         # and negative probabilities are set to 0

         pos_total = sum(self.pos_freqs.values())
         neg_total = sum(self.neg_freqs.values())
 
         vocab = set(self.pos_freqs.keys()).union(self.neg_freqs.keys())
         vocab_size = len(vocab)
 
         # for each token in the text, calculate the probability of it occurring in a
         # postive document and in a negative document and add the logs of those to the
         # running sums. when calculating the probabilities, always add 1 to the numerator
         # of each probability for add one smoothing (so that we never have a probability
         # of 0)
         for token in tokens:
             pos_freqs = self.pos_freqs.get(token, 0) + 1
             neg_freqs = self.neg_freqs.get(token, 0) + 1
 
             pos_score += math.log(pos_freqs / pos_total)
             neg_score += math.log(neg_freqs / neg_total)
             if token not in stopwords:
 
                 pos_freqs = self.pos_freqs.get(token, 0) + 1
                 neg_freqs = self.neg_freqs.get(token, 0) + 1
 
                 pos_score += math.log(pos_freqs / pos_total)
                 neg_score += math.log(neg_freqs / neg_total)
 
         # for debugging purposes, it may help to print the overall positive and negative
         # probabilities
        
    #  print(f"count for the word 'computer' in negative dictionary {b.neg_freqs['computer']}")
    #  print(f"count for the word 'science' in positive dictionary {b.pos_freqs['science']}")
    #  print(f"count for the word 'science' in negative dictionary {b.neg_freqs['science']}")
    #  print(f"count for the word 'i' in positive dictionary {b.pos_freqs['i']}")
    #  print(f"count for the word 'i' in negative dictionary {b.neg_freqs['i']}")
    #  print(f"count for the word 'is' in positive dictionary {b.pos_freqs['is']}")
    #  print(f"count for the word 'is' in negative dictionary {b.neg_freqs['is']}")
    #  print(f"count for the word 'the' in positive dictionary {b.pos_freqs['the']}")
    #  print(f"count for the word 'the' in negative dictionary {b.neg_freqs['the']}")
    #  print(f"count for the word 'i' in positive dictionary {b.pos_freqs['i']}")
     #print(f"count for the word 'i' in negative dictionary {b.neg_freqs['i']}")
     #print(f"count for the word 'is' in positive dictionary {b.pos_freqs['is']}")
     #print(f"count for the word 'is' in negative dictionary {b.neg_freqs['is']}")
     #print(f"count for the word 'the' in positive dictionary {b.pos_freqs['the']}")
     #print(f"count for the word 'the' in negative dictionary {b.neg_freqs['the']}")
 
    #  print("\nHere are some sample probabilities.")
    #  print(f"P('love'| pos) {(b.pos_freqs['love']+1)/pos_denominator}")
    #  print(f"P('love'| neg) {(b.neg_freqs['love']+1)/neg_denominator}")
    #  print(f"P('terrible'| pos) {(b.pos_freqs['terrible']+1)/pos_denominator}")
    #  print(f"P('terrible'| neg) {(b.neg_freqs['terrible']+1)/neg_denominator}")
 
     # # uncomment the below lines once you've implemented `classify`
     print("\nThe following should all be positive.")
    #  print(b.classify('I love computer science'))
    #  print(b.classify('this movie is fantastic'))
    #  print("\nThe following should all be negative.")
    #  print(b.classify('rainy days are the worst'))
    #  print(b.classify('computer science is terrible'))