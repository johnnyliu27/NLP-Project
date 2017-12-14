import sys
import gzip
import random
import torch
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# pieces of code taken from https://github.com/taolei87/rcnn/blob/master/code/qa/myio.py

def read_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.decode("utf-8").split("\t")
            if len(title) == 0:
                empty_cnt += 1
                continue
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = (title, body)
    return raw_corpus

def read_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = [ ]
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            if K_neg != -1:
                random.shuffle(neg)
                neg = neg[:K_neg + 2] # decreases chance of bad luck
            s = set()
            qids = [ ]
            qlabels = [ ]
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            lst.append((pid, qids, qlabels))

    return lst

def getEmbeddingTensor(embedding_path):
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.decode("utf-8").split()[0], l.decode("utf-8").split()[1:]
        vector = [float(x) for x in emb ]
        vector.append(0) # coord 201 is unk word
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) ) # padding
            unk = np.zeros(len(vector))
            unk[-1] = 1
            embedding_tensor.append( unk ) # for unk words
            
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+2
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)

    return embedding_tensor, word_to_indx

def getIndicesTensor(text_arr, word_to_indx, max_length, kernel_width = 3):
    nil_indx = 1
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    pad_indx = []
    if len(text_arr) >= kernel_width:
        pad_indx = ([1] * (len(text_arr) - kernel_width + 1))[:max_length - kernel_width + 1]
    if len(text_indx) < max_length:
        pad_indx.extend([0 for _ in range(max_length - kernel_width + 1 - len(pad_indx))])
        text_indx.extend( [0 for _ in range(max_length - len(text_indx))])

    x = torch.LongTensor(text_indx)
    y = torch.FloatTensor(pad_indx)

    return (x,y)

def map_corpus(corpus, word_to_indx, kernel_width = 3):
    mapped_corpus = {}
    for id in corpus:
        (title, body) = corpus[id]
        (titleTensor, titlePad) = getIndicesTensor(title, word_to_indx, 60, kernel_width = kernel_width)
        (bodyTensor, bodyPad) = getIndicesTensor(body, word_to_indx, 100, kernel_width = kernel_width)
        mapped_corpus[id] = (titleTensor, bodyTensor, titlePad, bodyPad)
    return mapped_corpus

def create_train_set(ids_corpus, data, K_neg = 20):
    N = len(data)
    triples = [ ]
    for u in range(N):
        pid, qids, qlabels = data[u]
        if pid not in ids_corpus: continue
        pos = [ q for q, l in zip(qids, qlabels) if l == 1 and q in ids_corpus ]
        neg = [ q for q, l in zip(qids, qlabels) if l == 0 and q in ids_corpus ][:K_neg]
        triples += [ [pid,x]+neg for x in pos ]

    train_set = []
    
    for triple in triples:
        pid = triple[0]
        pos = triple[1]
        neg = triple[2:]
        pid_tensor_title = ids_corpus[pid][0]
        pid_tensor_body = ids_corpus[pid][1]
        rest_title = torch.cat([torch.unsqueeze(ids_corpus[x][0],0) for x in [pos] + neg])
        rest_body = torch.cat([torch.unsqueeze(ids_corpus[x][1],0) for x in [pos] + neg])

        pid_pad_title = ids_corpus[pid][2]
        pid_pad_body = ids_corpus[pid][3]
        rest_pad_title = torch.cat([torch.unsqueeze(ids_corpus[x][2],0) for x in [pos] + neg])
        rest_pad_body = torch.cat([torch.unsqueeze(ids_corpus[x][3],0) for x in [pos] + neg])
        
        train_set.append({"pid_title" : pid_tensor_title, "rest_title" : rest_title,
                          "pid_body" : pid_tensor_body, "rest_body" : rest_body,
                          "pid_title_pad" : pid_pad_title, "pid_body_pad" : pid_pad_body,
                          "rest_title_pad" : rest_pad_title, "rest_body_pad" : rest_pad_body})
        
    return train_set


def create_dev_set(ids_corpus, data):
    N = len(data)
    dev_set = []
    for u in range(N):
        pid, qids, qlabels = data[u]
        if pid not in ids_corpus: continue 
    
        pid_tensor_title = ids_corpus[pid][0]
        pid_tensor_body = ids_corpus[pid][1]
        rest_title = torch.cat([torch.unsqueeze(ids_corpus[x][0],0) for x in qids])
        rest_body = torch.cat([torch.unsqueeze(ids_corpus[x][1],0) for x in qids])
        
        pid_pad_title = ids_corpus[pid][2]
        pid_pad_body = ids_corpus[pid][3]
        rest_pad_title = torch.cat([torch.unsqueeze(ids_corpus[x][2],0) for x in qids])
        rest_pad_body = torch.cat([torch.unsqueeze(ids_corpus[x][3],0) for x in qids])
        
        dev_set.append({"pid_title" : pid_tensor_title, "rest_title" : rest_title,
                        "pid_body" : pid_tensor_body, "rest_body" : rest_body,
                        "pid_title_pad" : pid_pad_title, "pid_body_pad" : pid_pad_body,
                        "rest_title_pad" : rest_pad_title, "rest_body_pad" : rest_pad_body,
                        "labels" : torch.LongTensor(qlabels)})

    return dev_set

def convert(sim, labels):
    (a, b) = sim.shape
    output = []
    for i in range(a):
        ranks = (-sim[i]).argsort()
        ranked_labels = labels[i][ranks]
        output.append(ranked_labels)
    return output

def build_tfidf_features(corpus_path):
    features = {}
    qid2rowid = {}
    corpus = []
    
    f = open(corpus_path)
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        s = re.split(r'\t', line)
        qID = s[0]
        text = s[1] + ' ' + s[2]
        qid2rowid[qID] = i
        corpus.append(text)
    f.close()
    
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    arrays = tfidf.toarray()
    for key in qid2rowid:
        features[key] = torch.FloatTensor(arrays[qid2rowid[key]])
    return features

def build_android_qsets(path_pos, path_neg):
    qIDs = []
    qCandidates = {}
    labels = {}

    f = open(path_pos)
    lines = f.readlines()
    for line in lines:
        qs = line.split()
        qID = qs[0]
        if qID not in qIDs:
            qIDs.append(qID)
            qCandidates[qID] = [qs[1]]
            labels[qID] = [1]
        else:
            if qs[1] not in qCandidates[qID]:
                qCandidates[qID].append(qs[1])
                labels[qID].append(1)

    f = open(path_neg)
    lines = f.readlines()
    for line in lines:
        qs = line.split()
        qID = qs[0]
        if qID not in qIDs:
            print('Error! No positive questions.')
            break
        else:
            if qs[1] not in qCandidates[qID]:
                qCandidates[qID].append(qs[1])
                labels[qID].append(0)
    return qIDs, qCandidates, labels
