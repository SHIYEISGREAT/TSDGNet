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
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 1,
                 concat: bool = True, dropout: float = 0.3,
                 negative_slope: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, num_heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, num_heads, out_dim))
        self.last_attn = None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
    def forward(self, x, adj):
        B, N, _ = x.shape
        H, D = self.num_heads, self.out_dim
        h = self.lin(x).view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        e_src = (h * self.att_src.unsqueeze(2)).sum(dim=-1)
        e_dst = (h * self.att_dst.unsqueeze(2)).sum(dim=-1)
        e = F.leaky_relu(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2),
                         negative_slope=self.negative_slope)
        if adj.dim() == 2:
            adj_exp = adj.unsqueeze(0).unsqueeze(0)
        else:
            adj_exp = adj.unsqueeze(1)
        e = e.masked_fill(adj_exp <= 0, -1e9)
        attn = torch.softmax(e, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        self.last_attn = attn.detach()
        out = torch.matmul(attn, h)
        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().view(B, N, H * D)
        else:
            out = out.mean(dim=1)
        return out
class SymbioticGATBranch(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 96,
                 out_dim: int = 128,
                 gat_dropout: float = 0.3):
        super().__init__()
        assert hidden_dim % 3 == 0, 'hidden_dim must be divisible by 3.'
        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(gat_dropout),
        )
        self.gat1 = GraphAttentionLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim // 3,
            num_heads=3,
            concat=True,
            dropout=gat_dropout,
        )
        self.gat2 = GraphAttentionLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=1,
            concat=False,
            dropout=gat_dropout,
        )
        self.gat3 = GraphAttentionLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=1,
            concat=False,
            dropout=gat_dropout,
        )
        self.gat4 = GraphAttentionLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=1,
            concat=False,
            dropout=gat_dropout,
        )
        self.gat5 = GraphAttentionLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=1,
            concat=False,
            dropout=gat_dropout,
        )
        self.post_act = nn.ELU()
        self.post_drop = nn.Dropout(gat_dropout)
        self.sym_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, out_dim),
            nn.ELU(),
            nn.Dropout(gat_dropout),
        )
    def forward(self, x, adj):
        x0 = self.pre(x)
        x1 = self.post_drop(self.post_act(self.gat1(x0, adj)))
        x2 = self.post_drop(self.post_act(self.gat2(x1, adj)))
        pooled_ctx = torch.cat([x1.mean(dim=1), x2.mean(dim=1)], dim=-1)
        gates = self.sym_gate(pooled_ctx)
        w_t = gates[:, 0].view(-1, 1, 1)
        w_r = gates[:, 1].view(-1, 1, 1)
        td_in = (1.0 - w_t) * x1 + w_t * x2
        ra_in = (1.0 - w_r) * x1 + w_r * x2
        x4 = self.post_drop(self.post_act(self.gat4(td_in, adj)))
        x5 = self.post_drop(self.post_act(self.gat5(ra_in, adj)))
        stage_in = x2 + 0.5 * (w_t * x4 + w_r * x5)
        x3 = self.post_drop(self.post_act(self.gat3(stage_in, adj)))
        graph_token = torch.cat([
            x3.mean(dim=1),
            x4.mean(dim=1),
            x5.mean(dim=1),
        ], dim=-1)
        embed = self.embed_proj(graph_token)
        aux = {
            'gate_t': gates[:, 0],
            'gate_r': gates[:, 1],
        }
        return embed, aux
class SGATBaselineNet(nn.Module):
    def __init__(self,
                 in_channels: int = 24,
                 num_classes: int = 8,
                 target_len: int = 97,
                 branch_hidden: int = 96,
                 branch_embed: int = 128,
                 fusion_hidden: int = 64,
                 gat_dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_len = target_len
        self.branch_embed = branch_embed
        self.node_branch = SymbioticGATBranch(
            in_dim=in_channels,
            hidden_dim=branch_hidden,
            out_dim=branch_embed,
            gat_dropout=gat_dropout,
        )
        self.body_branch = SymbioticGATBranch(
            in_dim=target_len,
            hidden_dim=branch_hidden,
            out_dim=branch_embed,
            gat_dropout=gat_dropout,
        )
        self.register_buffer('adj_node', self._build_temporal_adj(target_len, win_size=5))
        self.register_buffer('adj_body', self._build_body_adj())
        self.fusion_gate = nn.Sequential(
            nn.Linear(branch_embed * 2, fusion_hidden),
            nn.ELU(),
            nn.Linear(fusion_hidden, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(branch_embed, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.last_fusion_alpha = None
    @staticmethod
    def _build_temporal_adj(target_len: int, win_size: int = 5):
        half = max(0, win_size // 2)
        adj = torch.zeros(target_len, target_len, dtype=torch.float32)
        for i in range(target_len):
            l = max(0, i - half)
            r = min(target_len, i + half + 1)
            adj[i, l:r] = 1.0
        return adj
    @staticmethod
    def _build_body_adj():
        adj = torch.tensor([
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 1],
        ], dtype=torch.float32)
        return adj
    def _time_normalize(self, x):
        return F.interpolate(x, size=self.target_len, mode='linear', align_corners=False)
    def _build_node_feature_map(self, x_norm):
        return x_norm.permute(0, 2, 1).contiguous()
    def _build_body_feature_map(self, x_norm):
        B, C, T = x_norm.shape
        sensors = x_norm.view(B, 4, 6, T)
        acc_mag = torch.linalg.norm(sensors[:, :, 0:3, :], dim=2)
        gyr_mag = torch.linalg.norm(sensors[:, :, 3:6, :], dim=2)
        body = torch.stack([
            acc_mag[:, 0, :],
            acc_mag[:, 1, :],
            acc_mag[:, 2, :],
            acc_mag[:, 3, :],
            gyr_mag[:, 2, :],
            gyr_mag[:, 3, :],
        ], dim=1)
        return body
    def encode_branches(self, x):
        x_norm = self._time_normalize(x)
        node_fm = self._build_node_feature_map(x_norm)
        body_fm = self._build_body_feature_map(x_norm)
        node_embed, node_aux = self.node_branch(node_fm, self.adj_node)
        body_embed, body_aux = self.body_branch(body_fm, self.adj_body)
        return node_embed, body_embed, node_aux, body_aux
    def fuse_embeddings(self, node_embed, body_embed):
        alpha = torch.sigmoid(self.fusion_gate(torch.cat([node_embed, body_embed], dim=-1)))
        self.last_fusion_alpha = alpha.detach()
        fused = (1.0 - alpha) * node_embed + alpha * body_embed
        return fused, alpha
    def forward(self, x):
        node_embed, body_embed, _, _ = self.encode_branches(x)
        fused, _ = self.fuse_embeddings(node_embed, body_embed)
        logits = self.classifier(fused)
        return logits
    def extract_feature_vector(self, x):
        node_embed, body_embed, _, _ = self.encode_branches(x)
        fused, _ = self.fuse_embeddings(node_embed, body_embed)
        return fused
def create_model(num_classes,args):
    return SGATBaselineNet(in_channels=24,num_classes=num_classes)
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
    parser=argparse.ArgumentParser(description="Train SGATBaselineNet")
    parser.add_argument("--npz_path",type=str,default="gait1_preprocessed.npz")
    parser.add_argument("--checkpoint_path",type=str,default="sgat.pth")
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
