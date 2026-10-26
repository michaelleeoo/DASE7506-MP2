"""Fixed-schedule, seed-averaged verifiers; no per-epoch validation selection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train import labels,export_model
from common import load_inputs,parse_answer

def main():
    root=Path('runs/improve_v6');root.mkdir(exist_ok=True);data={}
    with np.load('runs/improve_v3/pca.npz') as z:pm,projection=z['mean'],z['projection']
    for split in ('train','validation'):
        bank,_=load_inputs(split)
        with np.load(Path('runs/improve_v3')/f'{split}_features.npz') as z:f={k:z[k] for k in z.files}
        y=labels(split,bank);hard=np.array([len(set(a for a in [parse_answer(c['text']) for c in b['candidates']] if a is not None))>=2 for b in bank])
        data[split]=dict(f=f,y=y,hard=hard)
    y,vy=data['train']['y'],data['validation']['y'];vhard=data['validation']['hard'];records=[];scores={}
    for dim in (128,256):
      for seed in (103,211,307,419):
        name=f'robust_{dim}_{seed}';torch.manual_seed(seed);rng=np.random.default_rng(seed)
        def build(d):
            f=d['f'];e=(f['embedding']-pm)@projection[:,:dim]
            return np.concatenate([e,e-e.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
        x,vx=build(data['train']),build(data['validation']);mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.05)
        x=torch.from_numpy((x-mean)/scale);vx=torch.from_numpy((vx-mean)/scale);ty=torch.from_numpy(y)
        net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=.00005)
        ids=np.flatnonzero(y.sum(1)<4);average=None;n=0;history=[];start=time.perf_counter()
        for epoch in range(1,101):
            net.train()
            for ii in np.array_split(rng.permutation(ids),len(ids)//128):
                s=net(x[ii]).squeeze(-1);target=ty[ii];target=.95*target/target.sum(1,keepdim=True)+.05/4
                loss=-(target*F.log_softmax(s,1)).sum(1).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            schedule.step()
            if epoch>=60 and epoch%5==0:
                state={k:v.detach().clone() for k,v in net.state_dict().items()};n+=1
                if average is None:average=state
                else:
                    for k in state:average[k]+=(state[k]-average[k])/n
            if epoch%10==0:
                net.eval()
                with torch.no_grad():vs=net(vx).squeeze(-1).numpy()
                ok=vy[np.arange(len(vy)),vs.argmax(1)]
                history.append(dict(epoch=epoch,hard_accuracy=float(ok[vhard].mean()*100),accuracy=float(ok.mean()*100)))
        net.load_state_dict(average);net.eval()
        with torch.no_grad():vs=net(vx).squeeze(-1).numpy()
        class Wrapper:pass
        w=Wrapper();w.net=net;export_model(w,mean,scale,root/f'{name}.npz');scores[name]=vs
        ok=vy[np.arange(len(vy)),vs.argmax(1)]
        rec=dict(name=name,dim=dim,kind_features='base',seed=seed,hard_accuracy=float(ok[vhard].mean()*100),
                 accuracy=float(ok.mean()*100),bytes=(root/f'{name}.npz').stat().st_size,
                 seconds=time.perf_counter()-start,history=history)
        records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
        (root/'robust_records.json').write_text(json.dumps(records,indent=2)+'\n')
        np.savez(root/'robust_scores.npz',**scores,labels=vy,hard=vhard)

if __name__=='__main__':main()
