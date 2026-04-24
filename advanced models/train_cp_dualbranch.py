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
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )
    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)
class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        res = self.residual(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + res)
        out = self.dropout(out)
        return out
class PaperTCNBiLSTMStream(nn.Module):
    def __init__(self, in_channels: int, tcn_channels: int = 64, lstm_hidden: int = 300,
                 out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            ResidualTCNBlock(in_channels, tcn_channels, kernel_size=9, dilation=1, dropout=dropout),
            ResidualTCNBlock(tcn_channels, tcn_channels, kernel_size=9, dilation=2, dropout=dropout),
            ResidualTCNBlock(tcn_channels, tcn_channels, kernel_size=9, dilation=4, dropout=dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, out_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.tcn(x)
        out = out.transpose(1, 2)
        out, _ = self.bilstm(out)
        out = self.dropout(out)
        out = out.mean(dim=1)
        return self.fc(out)
class MultiScaleGraphConvBlock(nn.Module):
    def __init__(self, d_model: int, num_nodes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        A = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.float32)
        I = torch.eye(num_nodes, dtype=torch.float32)
        A2 = (A @ A > 0).float()
        self.register_buffer("A0", self._normalize_adj(I))
        self.register_buffer("A1", self._normalize_adj(A))
        self.register_buffer("A2", self._normalize_adj(A2))
    @staticmethod
    def _normalize_adj(A: torch.Tensor) -> torch.Tensor:
        A = A.clone()
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        D = torch.diag(deg_inv_sqrt)
        return D @ A @ D
    def forward(self, h):
        outs = []
        for A, linear in zip([self.A0, self.A1, self.A2], self.linears):
            agg = torch.einsum("nm,btmc->btnc", A, h)
            outs.append(linear(agg))
        out = sum(outs) / len(outs)
        out = out + h
        B, T, N, C = out.shape
        out_bn = out.reshape(B * T * N, C)
        out_bn = self.bn(out_bn)
        out = out_bn.view(B, T, N, C)
        out = self.relu(out)
        out = self.dropout(out)
        return out
class MultiScaleTemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilations=(1, 2, 4), dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ))
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        out = self.relu(out + res)
        out = self.dropout(out)
        return out
class PaperSkeletonGraphStream(nn.Module):
    def __init__(self, num_nodes: int = 4, node_input_dim: int = 6, d_model: int = 128,
                 num_gcn_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.dynamic_proj = nn.Linear(node_input_dim, d_model)
        self.node_embed = nn.Parameter(torch.randn(1, 1, num_nodes, d_model) * 0.02)
        self.gcn_layers = nn.ModuleList([
            MultiScaleGraphConvBlock(d_model=d_model, num_nodes=num_nodes, dropout=dropout)
            for _ in range(num_gcn_layers)
        ])
        self.temporal = MultiScaleTemporalConvBlock(channels=d_model, dilations=(1, 2, 4), dropout=dropout)
    def forward(self, imu):
        B, _, T = imu.shape
        x = imu.view(B, self.num_nodes, 6, T).permute(0, 3, 1, 2)
        h = self.dynamic_proj(x) + self.node_embed
        for layer in self.gcn_layers:
            h = layer(h)
        h = h.max(dim=2).values
        h = h.transpose(1, 2).contiguous()
        h = self.temporal(h)
        h = h.max(dim=-1).values
        return h
class CPPaperDualBranchNet(nn.Module):
    def __init__(self, imu_channels: int = 24, joint_pos_channels: int = 0, joint_rot_channels: int = 0,
                 num_classes: int = 8, seq_out_dim: int = 256, skel_dim: int = 128):
        super().__init__()
        self.imu_channels = imu_channels
        self.joint_pos_channels = int(joint_pos_channels)
        self.joint_rot_channels = int(joint_rot_channels)
        time_in_channels = imu_channels + self.joint_pos_channels + self.joint_rot_channels
        self.time_stream = PaperTCNBiLSTMStream(
            in_channels=time_in_channels,
            tcn_channels=64,
            lstm_hidden=300,
            out_dim=seq_out_dim,
            dropout=0.2,
        )
        self.skeleton_stream = PaperSkeletonGraphStream(
            num_nodes=4,
            node_input_dim=6,
            d_model=skel_dim,
            num_gcn_layers=2,
            dropout=0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(seq_out_dim + skel_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    def _build_time_input(self, imu, joint_pos=None, joint_rot=None):
        feats = [imu]
        if joint_pos is not None and joint_pos.numel() > 0:
            feats.append(joint_pos)
        if joint_rot is not None and joint_rot.numel() > 0:
            feats.append(joint_rot)
        return torch.cat(feats, dim=1)
    def forward(self, imu, joint_pos=None, joint_rot=None):
        x_time = self._build_time_input(imu, joint_pos, joint_rot)
        f_sequence = self.time_stream(x_time)
        f_skeleton = self.skeleton_stream(imu)
        f_all = torch.cat([f_sequence, f_skeleton], dim=-1)
        logits = self.classifier(f_all)
        return logits
    def extract_feature_vector(self, imu, joint_pos=None, joint_rot=None):
        x_time = self._build_time_input(imu, joint_pos, joint_rot)
        f_sequence = self.time_stream(x_time)
        f_skeleton = self.skeleton_stream(imu)
        return torch.cat([f_sequence, f_skeleton], dim=-1)
def create_model(num_classes,args):
    return CPPaperDualBranchNet(imu_channels=24,num_classes=num_classes)
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
    parser=argparse.ArgumentParser(description="Train CPPaperDualBranchNet")
    parser.add_argument("--npz_path",type=str,default="gait1_preprocessed.npz")
    parser.add_argument("--checkpoint_path",type=str,default="cp_dualbranch.pth")
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
