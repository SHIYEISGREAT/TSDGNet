import argparse
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support
def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
class GaitNPZDataset(Dataset):
    def __init__(self,X,lengths,labels,indices,seq_len=2048):
        self.X=X
        self.lengths=lengths
        self.labels=labels
        self.indices=np.array(indices,dtype=np.int64)
        self.seq_len=seq_len
    def __len__(self):
        return len(self.indices)
    def _crop_and_pad(self,x,length):
        max_len=x.shape[0]
        L=max(1,min(int(length),max_len))
        if L>=self.seq_len:
            start=(L-self.seq_len)//2
            seg=x[start:start+self.seq_len]
        else:
            seg=np.zeros((self.seq_len,4,6),dtype=np.float32)
            seg[:L]=x[:L]
        return seg.reshape(self.seq_len,-1).transpose(1,0).astype(np.float32)
    def __getitem__(self,idx):
        i=self.indices[idx]
        x=self._crop_and_pad(self.X[i],self.lengths[i])
        y=self.labels[i]
        return torch.from_numpy(x),torch.tensor(y,dtype=torch.long)
class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,weight=None,label_smoothing=0.0):
        super().__init__()
        self.gamma=gamma
        self.weight=weight
        self.label_smoothing=label_smoothing
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,weight=self.weight,reduction="none",label_smoothing=self.label_smoothing)
        pt=torch.exp(-ce)
        return (((1-pt)**self.gamma)*ce).mean()
def train_one_epoch(model,dataloader,criterion,optimizer,device):
    model.train()
    loss_sum=0.0
    correct=0
    total=0
    for x,y in dataloader:
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        logits=model(x)
        loss=criterion(logits,y)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()*x.size(0)
        correct+=(logits.argmax(dim=1)==y).sum().item()
        total+=y.size(0)
    return loss_sum/max(total,1),correct/max(total,1)
@torch.no_grad()
def evaluate(model,dataloader,criterion,device):
    model.eval()
    loss_sum=0.0
    correct=0
    total=0
    for x,y in dataloader:
        x=x.to(device)
        y=y.to(device)
        logits=model(x)
        loss=criterion(logits,y)
        loss_sum+=loss.item()*x.size(0)
        correct+=(logits.argmax(dim=1)==y).sum().item()
        total+=y.size(0)
    return loss_sum/max(total,1),correct/max(total,1)
@torch.no_grad()
def predict(model,dataloader,device):
    model.eval()
    labels=[]
    preds=[]
    for x,y in dataloader:
        x=x.to(device)
        logits=model(x)
        labels.append(y.numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(labels),np.concatenate(preds)
def set_optimizer_lr(optimizer,lr):
    for pg in optimizer.param_groups:
        pg["lr"]=float(lr)
def cosine_lr(epoch,base_lr,eta_min,total_epochs,warmup_epochs=0):
    if warmup_epochs>0 and epoch<=warmup_epochs:
        return base_lr*epoch/warmup_epochs
    t_total=max(total_epochs-warmup_epochs,1)
    t=max(0,min(epoch-warmup_epochs-1,t_total-1))
    p=t/max(t_total-1,1)
    return eta_min+0.5*(base_lr-eta_min)*(1.0+math.cos(math.pi*p))
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, pool: bool = True, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2) if pool else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x
class AiCareCNNBaseline(nn.Module):
    def __init__(self, in_channels: int = 24, num_classes: int = 8, channels=(64, 128, 256), dropout: float = 0.3):
        super().__init__()
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, c1, kernel_size=7, pool=True, dropout=dropout * 0.5),
            ConvBNReLU(c1, c2, kernel_size=5, pool=True, dropout=dropout * 0.5),
            ConvBNReLU(c2, c3, kernel_size=3, pool=True, dropout=dropout),
            ConvBNReLU(c3, c3, kernel_size=3, pool=False, dropout=dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(c3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def extract_feature_vector(self, x):
        feat = self.encoder(x)
        vec = self.global_pool(feat).squeeze(-1)
        return vec
    def forward(self, x):
        vec = self.extract_feature_vector(x)
        logits = self.classifier(vec)
        return logits
def create_model(num_classes,args):
    return AiCareCNNBaseline(in_channels=24,num_classes=num_classes)
@torch.no_grad()
def extract_features(model,dataloader,device):
    model.eval()
    feats=[]
    labels=[]
    for x,y in dataloader:
        x=x.to(device)
        feats.append(model.extract_feature_vector(x).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats),np.concatenate(labels)
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
    train_dataset=GaitNPZDataset(X,lengths,labels,train_idx,seq_len=args.seq_len)
    val_dataset=GaitNPZDataset(X,lengths,labels,val_idx,seq_len=args.seq_len)
    test_dataset=GaitNPZDataset(X,lengths,labels,test_idx,seq_len=args.seq_len)
    train_labels=labels[train_idx]
    class_counts=np.bincount(train_labels,minlength=num_classes)
    class_weights=compute_class_weights(class_counts,method=args.weight_method,beta=args.weight_beta,normalize=bool(args.normalize_weights))
    if args.use_sampler:
        item_weights=class_weights[train_labels]
        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,sampler=WeightedRandomSampler(item_weights,len(item_weights),replacement=True),shuffle=False,num_workers=0)
    else:
        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    train_eval_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=create_model(num_classes,args).to(device)
    loss_weight=torch.tensor(class_weights,dtype=torch.float32,device=device) if args.use_class_weight_loss and args.weight_method!="none" else None
    criterion=FocalLoss(gamma=args.gamma_focal,weight=loss_weight,label_smoothing=args.label_smoothing) if args.loss_type=="focal" else nn.CrossEntropyLoss(weight=loss_weight,label_smoothing=args.label_smoothing)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best_state=None
    best_val_acc=0.0
    for epoch in range(1,args.epochs+1):
        if args.scheduler=="cosine":
            set_optimizer_lr(optimizer,cosine_lr(epoch,args.lr,args.eta_min,args.epochs,args.warmup_epochs))
        train_loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer,device)
        val_loss,val_acc=evaluate(model,val_loader,criterion,device)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            best_state=copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    if args.use_svm:
        train_feats,train_y=extract_features(model,train_eval_loader,device)
        val_feats,val_y=extract_features(model,val_loader,device)
        test_feats,test_labels=extract_features(model,test_loader,device)
        scaler=StandardScaler()
        trainval_feats=scaler.fit_transform(np.concatenate([train_feats,val_feats],axis=0))
        trainval_y=np.concatenate([train_y,val_y],axis=0)
        test_feats=scaler.transform(test_feats)
        clf=SVC(C=args.svm_c,kernel=args.svm_kernel,gamma=args.svm_gamma,class_weight="balanced",random_state=args.seed)
        clf.fit(trainval_feats,trainval_y)
        test_preds=clf.predict(test_feats)
    else:
        test_labels,test_preds=predict(model,test_loader,device)
    acc=float((test_labels==test_preds).mean())
    p,r,f1,_=precision_recall_fscore_support(test_labels,test_preds,average="macro",zero_division=0)
    print(f"Test Acc: {acc*100:.2f}%")
    print(f"Macro-P: {p*100:.2f}% | Macro-R: {r*100:.2f}% | Macro-F1: {f1*100:.2f}%")
    torch.save(model.state_dict(),args.checkpoint_path)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train AiCare CNN SVM")
    parser.add_argument("--npz_path",type=str,default="gait1_preprocessed.npz")
    parser.add_argument("--checkpoint_path",type=str,default="aicare_cnn_svm.pth")
    parser.add_argument("--epochs",type=int,default=400)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--seq_len",type=int,default=2048)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--scheduler",type=str,default="cosine",choices=["none","cosine"])
    parser.add_argument("--eta_min",type=float,default=1e-5)
    parser.add_argument("--warmup_epochs",type=int,default=10)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--use_sampler",type=int,default=1)
    parser.add_argument("--weight_method",type=str,default="effective",choices=["none","inv","sqrt_inv","effective"])
    parser.add_argument("--weight_beta",type=float,default=0.9999)
    parser.add_argument("--normalize_weights",type=int,default=1)
    parser.add_argument("--use_class_weight_loss",type=int,default=1)
    parser.add_argument("--loss_type",type=str,default="focal",choices=["ce","focal"])
    parser.add_argument("--gamma_focal",type=float,default=2.0)
    parser.add_argument("--label_smoothing",type=float,default=0.05)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--use_svm",type=int,default=1)
    parser.add_argument("--svm_c",type=float,default=10.0)
    parser.add_argument("--svm_kernel",type=str,default="rbf")
    parser.add_argument("--svm_gamma",type=str,default="scale")
    main(parser.parse_args())
