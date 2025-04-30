import numpy as np
import tensorflow as tf

def calc_F1(traj_act, traj_rec, noloop=False):
    assert isinstance(noloop, bool)
    assert traj_act and traj_rec
    if noloop:
        inter = len(set(traj_act)&set(traj_rec))
    else:
        match = np.zeros(len(traj_act), dtype=bool)
        for poi in traj_rec:
            for j in range(len(traj_act)):
                if not match[j] and poi==traj_act[j]:
                    match[j]=True
                    break
        inter = np.count_nonzero(match)
    rec = inter/len(traj_act)
    prec= inter/len(traj_rec)
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

def calc_pairsF1(y, y_hat):
    n, nr = len(y), len(y_hat)
    if n<2 or nr<2: return 0.0
    order = {v:i for i,v in enumerate(y)}
    nc=0
    for i in range(nr):
        for j in range(i+1,nr):
            if y_hat[i] in order and y_hat[j] in order and order[y_hat[i]]<order[y_hat[j]]:
                nc+=1
    n0 = n*(n-1)/2; n0r = nr*(nr-1)/2
    prec=nc/n0r if n0r>0 else 0.0
    rec =nc/n0  if n0>0  else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
