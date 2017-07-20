import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

import re
import os
import sys
import codecs
#import pandas
import argparse
import yaml
import importlib

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def parse_arg(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('-m', '--mapping', default='corpus.yaml', help='mapping yaml file to map corpus name to file')
    return parser.parse_args(argv[1:])

def combine_files(output_file_path, neg, pos):
    dirs = output_file_path.split('/')

    with open(output_file_path, 'w') as outfile:
        root = dirs[0]+'/'+dirs[1]

        for fname in os.listdir(root):
            if fname == neg :
                with open(root+"/"+fname) as infile:
                    for line in infile:
                        outfile.write("0 "+line)
            elif fname == pos:
                with open(root+"/"+fname) as infile:
                    for line in infile:
                        outfile.write("1 "+line)
            else:
                pass
    return

def build_data(data_files, cv=1, clean_string=True, TREC= False):
    """
    Loads data
    split = 1-> Test data
    split = 0 -> Train data
    split = 2 -> dev Data
    """

    revs = []
    vocab = defaultdict(float)

    for file in range(len(data_files)):
        with open(data_files[file], "rb") as f:
            for line in f:
                div = line.index(' ')
                if TREC:
                    sentence = clean_str(line[div+1:], TREC)
                else:
                    sentence = clean_str_sst(line[div+1:])

                label = line[:div]
                rev = []
                rev.append(sentence.strip())
                if clean_string:
                    if TREC:
                        orig_rev = clean_str(" ".join(rev),TREC)
                    else:
                        orig_rev = clean_str_sst(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":label, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 2}
                revs.append(datum)

    return revs, vocab

def build_data_cv(dat_file, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    all_file = data_file
    vocab = defaultdict(float)
    with open(all_file, "rb") as f:
        for line in f:
            div = line.index(' ')
            sentence = clean_str(line[div+1:])
            label = line[:div]

            rev = []
            rev.append(sentence.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":label, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    print revs[0]
    return revs, vocab

if __name__=="__main__":
    #python preprocess_data.py -m data.yam MR
    args = parse_arg(sys.argv)

    with open(args.mapping) as f:
        corpus = yaml.load(f)
        assert 'dir' in corpus, "yaml file should contain 'dir: path/to/data' line"
        assert 'w2v_file' in corpus, "yaml file should contain 'word2vec:' line"

    dataset = corpus[args.dataset]
    corpus_dir = corpus['dir']
    w2v_file = corpus['w2v_file']

    #### Doesn't need to CV (TREC, SST1, SST2)
    if 'test' in dataset:
        test_file = os.path.join(corpus_dir, (dataset['test']))
        train_file = os.path.join(corpus_dir, (dataset['train']))

        if 'dev' in dataset:
            ### SST1, SST2 (Special def clean_str_sst)
            dev_file = os.path.join(corpus_dir, (dataset['dev']))
            data_files = [train_file, test_file, dev_file]

            #data_files = os.path.join(corpus_dir, (dataset))
            #print data_files
            print "loading data...",
            revs, vocab = build_data(data_files, cv=10, clean_string=True, TREC= False)
            
        else:
            ### TREC (TREC=True)
            data_files = [train_file, test_file]
            print "loading data...",
            revs, vocab = build_data(data_files, cv=10, clean_string=True, TREC= True)
        
        max_l = np.max(pd.DataFrame(revs)["num_words"])

    #### Need to CV (MR, SUBJ, CR, MPQA dataset)
    else:
        if args.dataset == 'MR':
            ouput_file = os.path.join(corpus_dir, (dataset['train']))
            combine_files(ouput_file, 'rt-polarity.neg', 'rt-polarity.pos')
        elif args.dataset == 'SUBJ':
            ouput_file = os.path.join(corpus_dir, (dataset['train']))
            combine_files(ouput_file, 'quote.tok.gt9.5000', 'plot.tok.gt9.5000')

        data_file = os.path.join(corpus_dir, (dataset['train']))
        print "loading data...",
        revs, vocab = build_data_cv(data_file, cv=10, clean_string=True)
        max_l = np.max(pd.DataFrame(revs)["num_words"])
        avg_l = np.mean(pd.DataFrame(revs)["num_words"])

    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "average sentence length: " + str(np.mean(pd.DataFrame(revs)["num_words"]))


    #v_l = sorted(vocab.keys())
    #print (v_l)

    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("data_preprocess/"+args.dataset+".pkl", "wb"))
    print "dataset created!"
