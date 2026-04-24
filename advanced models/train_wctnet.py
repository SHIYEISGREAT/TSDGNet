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
class Conv1dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            bias=bias,
        )
    def forward(self, x):
        in_len = x.size(-1)
        out_len = math.ceil(in_len / self.stride)
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        pad_needed = max(0, (out_len - 1) * self.stride + effective_kernel - in_len)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_needed > 0:
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)
class WideCNNBranch(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv1dSamePadding(in_channels, 16, kernel_size=kernel_size, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            Conv1dSamePadding(16, 64, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            Conv1dSamePadding(64, 32, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.pool2(x)
        return x
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
class WCTEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 32, num_heads: int = 4,
                 dim_feedforward: int = 128, dropout: float = 0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.last_attn = None
    def forward(self, x):
        attn_out, attn_w = self.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn = attn_w.detach()
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x
class WCTNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        num_classes: int = 8,
        kernel_sizes=(3, 6, 9),
        d_model: int = 32,
        token_len: int = 28,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.token_len = int(token_len)
        self.d_model = int(d_model)
        self.branches = nn.ModuleList([
            WideCNNBranch(in_channels, k, dropout=dropout)
            for k in self.kernel_sizes
        ])
        self.token_pool = nn.AdaptiveAvgPool1d(self.token_len)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=self.d_model, dropout=dropout, max_len=max(4096, self.token_len + 32)
        )
        self.encoder = WCTEncoderBlock(
            d_model=self.d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.feature_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.token_len * self.d_model, hidden_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
    def _forward_features(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        feat = torch.stack(branch_outs, dim=0).mean(dim=0)
        feat = self.token_pool(feat)
        tokens = feat.transpose(1, 2).contiguous()
        tokens = self.pos_encoder(tokens)
        tokens = self.encoder(tokens)
        tokens = self.feature_dropout(tokens)
        flat = tokens.reshape(tokens.size(0), -1)
        hidden = F.relu(self.fc1(flat), inplace=True)
        hidden = self.head_dropout(hidden)
        return hidden, tokens
    def forward(self, x):
        hidden, _ = self._forward_features(x)
        logits = self.fc_out(hidden)
        return logits
    def extract_feature_vector(self, x):
        hidden, _ = self._forward_features(x)
        return hidden
    def get_last_attention_map(self):
        return self.encoder.last_attn
def create_model(num_classes,args):
    return WCTNet(in_channels=24,num_classes=num_classes)
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
    model=create_model(num_classes,args).to(device)
    loss_weight=torch.tensor(class_weights,dtype=torch.float32,device=device) if args.use_class_weight_loss and args.weight_method!="none" else None
    if args.loss_type=="ce":
        criterion=nn.CrossEntropyLoss(weight=loss_weight,label_smoothing=args.label_smoothing)
    else:
        criterion=FocalLoss(gamma=args.gamma_focal,weight=loss_weight,label_smoothing=args.label_smoothing)
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
    parser=argparse.ArgumentParser(description="Train WCTNet")
    parser.add_argument("--npz_path",type=str,default="gait1_preprocessed.npz")
    parser.add_argument("--checkpoint_path",type=str,default="wctnet.pth")
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
    main(parser.parse_args())
