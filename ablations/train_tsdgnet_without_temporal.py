import argparse
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
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
class TSSKConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_sizes=(3,5,9),stride=1,groups=4,reduction=16):
        super().__init__()
        if out_channels%groups!=0:
            raise ValueError("out_channels must be divisible by groups")
        self.kernel_sizes=kernel_sizes
        self.branches=nn.ModuleList([nn.Conv1d(in_channels,out_channels,kernel_size=k,stride=stride,padding=k//2,groups=groups,bias=False) for k in kernel_sizes])
        d=max(out_channels//reduction,8)
        self.fc1=nn.Linear(out_channels,d)
        self.fc2=nn.Linear(d,out_channels*len(kernel_sizes))
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        branch_outs=[conv(x) for conv in self.branches]
        u=sum(branch_outs)
        z=torch.relu(self.fc1(u.mean(dim=-1)))
        z=self.fc2(z)
        b,_=z.shape
        c=branch_outs[0].shape[1]
        z=z.view(b,len(self.kernel_sizes),c)
        a=self.softmax(z)
        out=0.0
        for i,y in enumerate(branch_outs):
            out=out+a[:,i,:].unsqueeze(-1)*y
        return out
class TSSKBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_sizes=(3,5,9),stride=1,groups=4,reduction=16,use_pool=True):
        super().__init__()
        self.tssk=TSSKConv1d(in_channels,out_channels,kernel_sizes=kernel_sizes,stride=stride,groups=groups,reduction=reduction)
        self.bn=nn.BatchNorm1d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool1d(kernel_size=2) if use_pool else None
        self.res_conv=nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False) if in_channels!=out_channels or stride!=1 else None
    def forward(self,x):
        out=self.tssk(x)
        res=self.res_conv(x) if self.res_conv is not None else x
        out=self.relu(self.bn(out)+res)
        if self.pool is not None:
            out=self.pool(out)
        return out
class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads=4):
        super().__init__()
        if d_model%num_heads!=0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_head=d_model//num_heads
        self.q_proj=nn.Linear(d_model,d_model)
        self.k_proj=nn.Linear(d_model,d_model)
        self.v_proj=nn.Linear(d_model,d_model)
        self.out_proj=nn.Linear(d_model,d_model)
    def forward(self,h,adj):
        b,n,_=h.shape
        q=self.q_proj(h).view(b,n,self.num_heads,self.d_head).transpose(1,2)
        k=self.k_proj(h).view(b,n,self.num_heads,self.d_head).transpose(1,2)
        v=self.v_proj(h).view(b,n,self.num_heads,self.d_head).transpose(1,2)
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_head)
        adj_exp=adj.unsqueeze(0).unsqueeze(0) if adj.dim()==2 else adj.unsqueeze(1)
        scores=scores.masked_fill(adj_exp==0,float("-inf"))
        attn=torch.softmax(scores,dim=-1)
        out=torch.matmul(attn,v)
        out=out.transpose(1,2).contiguous().view(b,n,self.d_model)
        return self.out_proj(out)
class GraphRefineBlock(nn.Module):
    def __init__(self,d_model,num_heads=4,d_ff=128,dropout=0.3,alpha_init=0.3):
        super().__init__()
        self.attn_body=MaskedMultiHeadSelfAttention(d_model,num_heads)
        self.attn_sym=MaskedMultiHeadSelfAttention(d_model,num_heads)
        self.gate_mlp=nn.Sequential(nn.Linear(3*d_model,d_model),nn.Tanh(),nn.Linear(d_model,d_model),nn.Sigmoid())
        self.alpha=nn.Parameter(torch.tensor(alpha_init))
        self.dropout=nn.Dropout(dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(nn.Linear(d_model,d_ff),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(d_ff,d_model))
        self.norm2=nn.LayerNorm(d_model)
    def forward(self,h,adj_body,adj_sym,context):
        h_body=self.attn_body(h,adj_body)
        h_sym=self.attn_sym(h,adj_sym)
        delta_body=h_body-h
        delta_sym=h_sym-h
        c=context.unsqueeze(1).expand(-1,h.size(1),-1)
        gate=self.gate_mlp(torch.cat([delta_body,delta_sym,c],dim=-1))
        delta=self.dropout(gate*(delta_body+delta_sym)*0.5)
        h_new=self.norm1(h+self.alpha*delta)
        return self.norm2(h_new+self.ffn(h_new))
class TSDGNet(nn.Module):
    def __init__(self,in_channels=24,num_classes=8,backbone_channels=(64,128,256),d_model=64,num_graph_layers=1,num_heads=4,alpha_init=0.3):
        super().__init__()
        c1,c2,c3=backbone_channels
        self.backbone=nn.Sequential(TSSKBlock(in_channels,c1,kernel_sizes=(3,5,9),groups=4,reduction=16,use_pool=True),TSSKBlock(c1,c2,kernel_sizes=(3,5,9),groups=4,reduction=16,use_pool=True),TSSKBlock(c2,c3,kernel_sizes=(3,5,9),groups=4,reduction=16,use_pool=True))
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.pathology_proj=nn.Linear(c3,d_model)
        if c3%4!=0:
            raise ValueError("The last backbone channel must be divisible by 4")
        self.node_proj=nn.Linear(c3//4,d_model)
        adj_body=torch.tensor([[1,1,0,0],[1,1,1,1],[0,1,1,0],[0,1,0,1]],dtype=torch.float32)
        adj_sym=torch.tensor([[1,0,0,0],[0,1,1,1],[0,1,1,1],[0,1,1,1]],dtype=torch.float32)
        self.register_buffer("adj_body",adj_body)
        self.register_buffer("adj_sym",adj_sym)
        self.graph_layers=nn.ModuleList([GraphRefineBlock(d_model,num_heads=num_heads,d_ff=128,dropout=0.3,alpha_init=alpha_init) for _ in range(num_graph_layers)])
        self.classifier=nn.Sequential(nn.Linear(d_model,256),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(256,num_classes))
    def forward(self,x):
        feat=self.backbone(x)
        b,c,t=feat.shape
        g_backbone=self.gap(feat).squeeze(-1)
        context=self.pathology_proj(g_backbone)
        node_feat=feat.view(b,4,c//4,t).mean(dim=-1)
        h=self.node_proj(node_feat)
        for layer in self.graph_layers:
            h=layer(h,self.adj_body,self.adj_sym,context)
        g_graph=h.mean(dim=1)
        return self.classifier(g_graph)
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
    test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TSDGNet(in_channels=24,num_classes=num_classes,backbone_channels=(64,128,256),d_model=64,num_graph_layers=1,num_heads=4,alpha_init=args.alpha_graph).to(device)
    loss_weight=torch.tensor(class_weights,dtype=torch.float32,device=device) if args.use_class_weight_loss and args.weight_method!="none" else None
    if args.loss_type=="ce":
        criterion=nn.CrossEntropyLoss(weight=loss_weight,label_smoothing=args.label_smoothing)
    elif args.loss_type=="focal":
        criterion=FocalLoss(gamma=args.gamma_focal,weight=loss_weight,label_smoothing=args.label_smoothing)
    else:
        raise ValueError("loss_type must be ce or focal")
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
    test_labels,test_preds=predict(model,test_loader,device)
    acc=float((test_labels==test_preds).mean())
    p,r,f1,_=precision_recall_fscore_support(test_labels,test_preds,average="macro",zero_division=0)
    print(f"Test Acc: {acc*100:.2f}%")
    print(f"Macro-P: {p*100:.2f}% | Macro-R: {r*100:.2f}% | Macro-F1: {f1*100:.2f}%")
    torch.save(model.state_dict(),args.checkpoint_path)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train TSDGNet")
    parser.add_argument("--npz_path",type=str,default="gait1_preprocessed.npz")
    parser.add_argument("--checkpoint_path",type=str,default="best_tsdgnet_graph_only.pt")
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
    parser.add_argument("--alpha_graph",type=float,default=0.3)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    main(parser.parse_args())
