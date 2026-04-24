import argparse
import math
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
def grouped_split(ids,train_ratio=0.7,val_ratio=0.15,seed=42):
    set_seed(seed)
    ids=np.array(ids)
    uniq=np.unique(ids)
    uniq=uniq[np.random.permutation(len(uniq))]
    n_train=int(len(uniq)*train_ratio)
    n_val=int(len(uniq)*val_ratio)
    train_ids=set(uniq[:n_train])
    val_ids=set(uniq[n_train:n_train+n_val])
    test_ids=set(uniq[n_train+n_val:])
    all_idx=np.arange(len(ids))
    train_idx=[i for i in all_idx if ids[i] in train_ids]
    val_idx=[i for i in all_idx if ids[i] in val_ids]
    test_idx=[i for i in all_idx if ids[i] in test_ids]
    return train_idx,val_idx,test_idx
def compute_class_weights(counts,method="effective",beta=0.9999,normalize=True,eps=1e-6):
    counts=np.asarray(counts,dtype=np.float32)
    if method=="none":
        w=np.ones_like(counts,dtype=np.float32)
    elif method=="inv":
        w=1.0/(counts+eps)
    elif method=="sqrt_inv":
        w=1.0/np.sqrt(counts+eps)
    elif method=="effective":
        w=(1.0-beta)/(1.0-np.power(beta,counts)+eps)
    else:
        raise ValueError("Unsupported weight method")
    w=np.where(counts>0,w,0.0).astype(np.float32)
    if normalize and float(w.sum())>0:
        w=w*(len(w)/(float(w.sum())+eps))
    return w
def crop_and_pad(x,length,seq_len):
    max_len=x.shape[0]
    L=max(1,min(int(length),max_len))
    if L>=seq_len:
        start=(L-seq_len)//2
        seg=x[start:start+seq_len]
    else:
        seg=np.zeros((seq_len,4,6),dtype=np.float32)
        seg[:L]=x[:L]
    return seg.reshape(seq_len,-1).transpose(1,0).astype(np.float32)
def temporal_bin_mean(signal,out_len):
    t=len(signal)
    bounds=np.linspace(0,t,out_len+1)
    pooled=np.zeros(out_len,dtype=np.float32)
    for i in range(out_len):
        left=int(math.floor(bounds[i]))
        right=int(math.floor(bounds[i+1]))
        if right<=left:
            right=min(left+1,t)
        pooled[i]=float(np.mean(signal[left:right]))
    return pooled
def build_features(X,lengths,indices,seq_len=2048,downsample_len=128):
    feats=[]
    for i in indices:
        x=crop_and_pad(X[i],lengths[i],seq_len)
        if downsample_len<seq_len:
            x=np.stack([temporal_bin_mean(ch,downsample_len) for ch in x],axis=0)
        feats.append(x.reshape(-1))
    return np.asarray(feats,dtype=np.float32)
def main(args):
    set_seed(args.seed)
    data=np.load(args.npz_path,allow_pickle=True)
    X=data["X"]
    lengths=data["lengths"]
    pathology=data["pathology"]
    id_key="sub"+"ject_id"
    group_ids=data[id_key]
    classes=sorted(list(set(pathology.tolist())))
    class_to_id={c:i for i,c in enumerate(classes)}
    labels=np.array([class_to_id[p] for p in pathology],dtype=np.int64)
    num_classes=len(classes)
    train_idx,val_idx,test_idx=grouped_split(group_ids,train_ratio=0.7,val_ratio=0.15,seed=args.seed)
    train_all_idx=np.array(sorted(list(train_idx)+list(val_idx)),dtype=np.int64)
    train_labels=labels[train_all_idx]
    class_counts=np.bincount(train_labels,minlength=num_classes)
    class_weights=compute_class_weights(class_counts,method=args.weight_method,beta=args.weight_beta,normalize=bool(args.normalize_weights))
    sample_weights=class_weights[train_labels] if args.use_sample_weight else None
    X_train=build_features(X,lengths,train_all_idx,seq_len=args.seq_len,downsample_len=args.downsample_len)
    y_train=labels[train_all_idx]
    X_test=build_features(X,lengths,test_idx,seq_len=args.seq_len,downsample_len=args.downsample_len)
    y_test=labels[test_idx]
    model=XGBClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth,learning_rate=args.lr,subsample=args.subsample,colsample_bytree=args.colsample_bytree,objective="multi:softprob",num_class=num_classes,eval_metric="mlogloss",random_state=args.seed,n_jobs=args.n_jobs,tree_method=args.tree_method)
    model.fit(X_train,y_train,sample_weight=sample_weights)
    probs=model.predict_proba(X_test)
    preds=np.argmax(probs,axis=1)
    acc=float((y_test==preds).mean())
    p,r,f1,_=precision_recall_fscore_support(y_test,preds,average="macro",zero_division=0)
    print(f"Test Acc: {acc*100:.2f}%")
    print(f"Macro-P: {p*100:.2f}% | Macro-R: {r*100:.2f}% | Macro-F1: {f1*100:.2f}%")
    if args.model_path:
        model.save_model(args.model_path)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train XGBoost")
    parser.add_argument("--npz_path",type=str,default="gait1_preprocessed.npz")
    parser.add_argument("--model_path",type=str,default="best_xgboost.json")
    parser.add_argument("--seq_len",type=int,default=2048)
    parser.add_argument("--downsample_len",type=int,default=128)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--n_estimators",type=int,default=500)
    parser.add_argument("--max_depth",type=int,default=5)
    parser.add_argument("--lr",type=float,default=0.03)
    parser.add_argument("--subsample",type=float,default=0.9)
    parser.add_argument("--colsample_bytree",type=float,default=0.9)
    parser.add_argument("--n_jobs",type=int,default=4)
    parser.add_argument("--tree_method",type=str,default="hist")
    parser.add_argument("--use_sample_weight",type=int,default=1)
    parser.add_argument("--weight_method",type=str,default="effective",choices=["none","inv","sqrt_inv","effective"])
    parser.add_argument("--weight_beta",type=float,default=0.9999)
    parser.add_argument("--normalize_weights",type=int,default=1)
    main(parser.parse_args())
