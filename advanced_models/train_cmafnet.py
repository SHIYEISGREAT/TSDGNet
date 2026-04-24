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
def conv1d_out_length(L: int, kernel_size: int = 3, stride: int = 1,
                      padding: int = 1, dilation: int = 1) -> int:
    return (L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
def split_imu_to_acc_gyro(x: torch.Tensor):
    if x.dim() != 3:
        raise ValueError(f"x must have shape (B, C, T), but got {tuple(x.shape)}")
    B, C, T = x.shape
    if C % 6 != 0:
        raise ValueError(f"channel dimension must be divisible by 6, but got C={C}")
    num_sensors = C // 6
    x = x.view(B, num_sensors, 6, T)
    acc = x[:, :, :3, :].reshape(B, num_sensors * 3, T)
    gyr = x[:, :, 3:, :].reshape(B, num_sensors * 3, T)
    return acc, gyr
class ConvBNReLU1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
class CrossModalityAttention1D(nn.Module):
    def __init__(self, channels: int, attn_dim: int = None, dropout: float = 0.5):
        super().__init__()
        self.channels = channels
        self.attn_dim = int(attn_dim or channels)
        self.q_x = nn.Linear(channels, self.attn_dim, bias=False)
        self.k_x = nn.Linear(channels, self.attn_dim, bias=False)
        self.q_y = nn.Linear(channels, self.attn_dim, bias=False)
        self.k_y = nn.Linear(channels, self.attn_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_x_to_y = None
        self.last_attn_y_to_x = None
    def _cross_attention(self, q_src, k_other, v_src, q_proj, k_proj):
        q_src_t = q_src.transpose(1, 2)
        k_other_t = k_other.transpose(1, 2)
        v_src_t = v_src.transpose(1, 2)
        q = q_proj(q_src_t)
        k = k_proj(k_other_t)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.attn_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v_src_t)
        return out.transpose(1, 2), attn
    def forward(self, x, y):
        a_x_to_y, attn_x_to_y = self._cross_attention(x, y, x, self.q_x, self.k_y)
        a_y_to_x, attn_y_to_x = self._cross_attention(y, x, y, self.q_y, self.k_x)
        self.last_attn_x_to_y = attn_x_to_y.detach()
        self.last_attn_y_to_x = attn_y_to_x.detach()
        x_out = torch.cat([x, a_x_to_y], dim=1)
        y_out = torch.cat([y, a_y_to_x], dim=1)
        return x_out, y_out
class ChannelWiseFusion1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.modality_fc = nn.Linear(2, 2, bias=True)
        self.bn = nn.BatchNorm1d(channels)
        self.last_alpha = None
        self.last_beta = None
    def forward(self, x, y):
        px = self.shared_conv(self.pool(x)).squeeze(-1)
        py = self.shared_conv(self.pool(y)).squeeze(-1)
        stacked = torch.stack([px, py], dim=-1)
        logits = self.modality_fc(stacked)
        weights = torch.softmax(logits, dim=-1)
        alpha = weights[..., 0].unsqueeze(-1)
        beta = weights[..., 1].unsqueeze(-1)
        z = alpha * x + beta * y
        z = self.bn(z)
        self.last_alpha = alpha.detach()
        self.last_beta = beta.detach()
        return z
class CMAFNetPaperAdapted(nn.Module):
    def __init__(self,
                 in_channels: int = 24,
                 num_classes: int = 8,
                 seq_len: int = 2048,
                 attn_dropout: float = 0.5,
                 stride_cfg=(8, 2, 2, 2)):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2 for Acc/Gyro split")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.stride_cfg = tuple(stride_cfg)
        branch_in = in_channels // 2
        self.enc1_acc = ConvBNReLU1d(branch_in, 32, kernel_size=3, stride=self.stride_cfg[0])
        self.enc1_gyr = ConvBNReLU1d(branch_in, 32, kernel_size=3, stride=self.stride_cfg[0])
        self.cross_attn = CrossModalityAttention1D(32, attn_dim=32, dropout=attn_dropout)
        self.enc2_acc = ConvBNReLU1d(64, 128, kernel_size=3, stride=self.stride_cfg[1])
        self.enc2_gyr = ConvBNReLU1d(64, 128, kernel_size=3, stride=self.stride_cfg[1])
        self.enc3_acc = ConvBNReLU1d(128, 256, kernel_size=3, stride=self.stride_cfg[2])
        self.enc3_gyr = ConvBNReLU1d(128, 256, kernel_size=3, stride=self.stride_cfg[2])
        self.enc4_acc = ConvBNReLU1d(256, 512, kernel_size=3, stride=self.stride_cfg[3])
        self.enc4_gyr = ConvBNReLU1d(256, 512, kernel_size=3, stride=self.stride_cfg[3])
        self.fusion = ChannelWiseFusion1D(512)
        self.dec1 = ConvBNReLU1d(512, 256, kernel_size=3, stride=1)
        self.dec2 = ConvBNReLU1d(256, 128, kernel_size=3, stride=1)
        self.dec3 = ConvBNReLU1d(128, 64, kernel_size=3, stride=1)
        self.dec4 = ConvBNReLU1d(64, num_classes, kernel_size=3, stride=1)
        t_after = seq_len
        for s in self.stride_cfg:
            t_after = conv1d_out_length(t_after, kernel_size=3, stride=s, padding=1)
        self.temporal_dim_after_encoder = int(t_after)
        linear1_out = max(2 * self.temporal_dim_after_encoder, 32)
        self.linear1 = nn.Linear(num_classes * self.temporal_dim_after_encoder, linear1_out)
        self.linear2 = nn.Linear(linear1_out, 110)
        self.linear3 = nn.Linear(110, num_classes)
    def _encode(self, x):
        acc, gyr = split_imu_to_acc_gyro(x)
        xa = self.enc1_acc(acc)
        xg = self.enc1_gyr(gyr)
        xa, xg = self.cross_attn(xa, xg)
        xa = self.enc2_acc(xa)
        xg = self.enc2_gyr(xg)
        xa = self.enc3_acc(xa)
        xg = self.enc3_gyr(xg)
        xa = self.enc4_acc(xa)
        xg = self.enc4_gyr(xg)
        return xa, xg
    def _fused_decoder_features(self, x):
        xa, xg = self._encode(x)
        z = self.fusion(xa, xg)
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        z = self.dec4(z)
        flat = z.reshape(z.size(0), -1)
        h1 = F.relu(self.linear1(flat), inplace=True)
        h2 = F.relu(self.linear2(h1), inplace=True)
        return h2
    def forward(self, x):
        feat = self._fused_decoder_features(x)
        logits = self.linear3(feat)
        return logits
    def extract_feature_vector(self, x):
        return self._fused_decoder_features(x)
def create_model(num_classes,args):
    return CMAFNetPaperAdapted(in_channels=24,num_classes=num_classes,seq_len=args.seq_len)
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
    parser=argparse.ArgumentParser(description="Train CMAFNetPaperAdapted")
    parser.add_argument("--npz_path",type=str,default="gait1_preprocessed.npz")
    parser.add_argument("--checkpoint_path",type=str,default="cmafnet.pth")
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
