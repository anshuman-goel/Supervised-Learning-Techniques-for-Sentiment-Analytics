import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#import numpy as np
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def removeStopwords(data,stopwords):
    i=0
    for each_list in data:
        data[i]=list(set(each_list)-stopwords)
        i=i+1
    return data


def gettingWordList(data):
    word_list={}
    for para in data:
        for word in set(para):
            if word in word_list:
                word_list[word]+=1
            else:
                word_list[word]=1
    return word_list


def vectorConstruction(text,word_list):
    vector=[[0]*len(word_list) for i in range(len(text))]
    for i,line in enumerate(text):
        for word in line:
            if word in word_list:
                vector[i][word_list.index(word)]=1
    return vector
    

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    #print(train_pos[len(train_pos)-1])
    #train_pos=list(set(train_pos)-stopwords))
    #train_pos_new=[]
    train_pos=removeStopwords(train_pos,stopwords)
    train_neg=removeStopwords(train_neg,stopwords)
    test_pos=removeStopwords(test_pos,stopwords)
    test_neg=removeStopwords(test_neg,stopwords)

    word_list_pos=gettingWordList(train_pos)
    word_list_neg=gettingWordList(train_neg)

    word_list=[]

    for word,count in word_list_pos.items():
        if count>=len(train_pos)*.01 or count>=len(train_neg)*.01:
            word_list.append(word)

    for word,count in word_list_neg.items():
        if count>=len(train_pos)*.01 or count>=len(train_neg)*.01:
            word_list.append(word)

    temp_list=list(set(word_list))
    word_list=[]
    for word in temp_list:
        if word_list_pos[word]>=word_list_neg[word]*2 or word_list_neg[word]>=word_list_pos[word]*2:
            word_list.append(word)

    #print(train_pos[len(train_pos)-1])

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    train_pos_vec=vectorConstruction(train_pos,word_list)
    train_neg_vec=vectorConstruction(train_neg,word_list)
    test_pos_vec=vectorConstruction(test_pos,word_list)
    test_neg_vec=vectorConstruction(test_neg,word_list)

    """print(len(train_pos_vec))
    print(len(train_neg_vec))
    print(len(test_pos_vec))
    print(len(test_neg_vec))"""
    #print(train_pos_vec[1])

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def constructFeatureVecDoc(data,label):

    object_list=[]
    for i,sentence in enumerate(data):
        object_list.append(LabeledSentence(sentence,[label+str(i)]))

    return object_list
        

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos=constructFeatureVecDoc(train_pos,"TRAIN_POS_")
    labeled_train_neg=constructFeatureVecDoc(train_neg,"TRAIN_NEG_")
    labeled_test_pos=constructFeatureVecDoc(test_pos,"TEST_POS_")
    labeled_test_neg=constructFeatureVecDoc(test_neg,"TEST_NEG_")

    #print(train_pos_obj)
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    #print(model.docvecs)

    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = [],[],[],[]
    for each_tag in model.docvecs.doctags.keys():
        if "TRAIN_POS_" in each_tag:
            train_pos_vec.append(model.docvecs[each_tag])
        elif "TRAIN_NEG_" in each_tag:
            train_neg_vec.append(model.docvecs[each_tag])
        elif "TEST_POS_" in each_tag:
            test_pos_vec.append(model.docvecs[each_tag])
        elif "TEST_NEG_" in each_tag:
            test_neg_vec.append(model.docvecs[each_tag])

    #print(train_pos_vec)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    NB=sklearn.naive_bayes.BernoulliNB(alpha=1.0 , binarize=None)
    nb_model=NB.fit(train_pos_vec+train_neg_vec,Y)
    LR=sklearn.linear_model.LogisticRegression()
    lr_model=LR.fit(train_pos_vec+train_neg_vec,Y)
    #a=train_pos_vec+train_neg_vec
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    GB=sklearn.naive_bayes.GaussianNB()
    nb_model=GB.fit(train_pos_vec+train_neg_vec,Y)
    LR=sklearn.linear_model.LogisticRegression()
    lr_model=LR.fit(train_pos_vec+train_neg_vec,Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE

    class_label=model.predict(test_pos_vec+test_neg_vec)
    test_pos_len=len(test_pos_vec)
    #test_neg_len=len(test_neg_vec)

    tp,tn,fp,fn=0,0,0,0
    for i in range(len(class_label)):
        if i<test_pos_len and class_label[i]=='pos':
            tp+=1
        elif i<test_pos_len and class_label[i]=='neg':
            fn+=1
        elif class_label[i]=='pos':
            fp+=1
        else:
            tn+=1

    accuracy=float((tp+tn))/float((tp+tn+fp+fn))
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
