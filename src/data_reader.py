import sys
import gzip
import random
import torch
import numpy as np

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
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)

    return embedding_tensor, word_to_indx

def getIndicesTensor(text_arr, word_to_indx, max_length):
    nil_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])

    x =  torch.LongTensor(text_indx)

    return x

def map_corpus(corpus, word_to_indx):
    mapped_corpus = {}
    for id in corpus:
        (title, body) = corpus[id]
        titleIds = getIndicesTensor(title, word_to_indx, 60)
        bodyIds = getIndicesTensor(body, word_to_indx, 100)
        mapped_corpus[id] = (titleIds, bodyIds)
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
        train_set.append({"pid_title" : pid_tensor_title, "rest_title" : rest_title,
                         "pid_body" : pid_tensor_body, "rest_body" : rest_body})
        # TODO : also add body/title lengths for normalization in cnn
    return train_set


def create_dev_set(ids_corpus, data):
    N = len(data)
    dev_set = []
    for u in range(N):
        pid, qids, qlabels = data[u]
        if pid not in ids_corpus: continue 
        pos = [ q for q, l in zip(qids, qlabels) if l == 1 and q in ids_corpus ]
        neg = [ q for q, l in zip(qids, qlabels) if l == 0 and q in ids_corpus ]
    
        pid_tensor_title = ids_corpus[pid][0]
        pid_tensor_body = ids_corpus[pid][1]
        rest_title = torch.cat([torch.unsqueeze(ids_corpus[x][0],0) for x in qids])
        rest_body = torch.cat([torch.unsqueeze(ids_corpus[x][1],0) for x in qids])
        dev_set.append({"pid_title" : pid_tensor_title, "rest_title" : rest_title,
                        "pid_body" : pid_tensor_body, "rest_body" : rest_body,
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
