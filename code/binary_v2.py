"""Outcome supervision on ALL training candidates, including all-correct questions."""
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
    root=Path('runs/improve_v10');root.mkdir(exist_ok=True);data={}
    with np.load('runs/improve_v3/pca.npz') as z:pm,projection=z['mean'],z['projection']
    for split in ('train','validation'):
        bank,_=load_inputs(split)
        with np.load(Path('runs/improve_v3')/f'{split}_features.npz') as z:f={k:z[k] for k in z.files}
        y=labels(split,bank);hard=np.array([len(set(a for a in [parse_answer(c['text']) for c in b['candidates']] if a is not None))>=2 for b in bank])
        data[split]=dict(f=f,y=y,hard=hard)
    y,vy=data['train']['y'],data['validation']['y'];vhard=data['validation']['hard'];records=[];scores={}
    specs=[('raw',0,'bce',17),('raw',0,'bce',43),('local',128,'bce',17),('local',128,'bce',43),
           ('context',128,'bce',17),('context',128,'bce',43),('context',256,'bce',71),
           ('context',128,'hybrid',17),('local',256,'bce',71),('embedding',0,'bce',29)]
    for mode,dim,loss_kind,seed in specs:
        name=f'binary_{mode}_{dim}_{loss_kind}_{seed}';torch.manual_seed(seed);rng=np.random.default_rng(seed)
        def build(d):
            f=d['f'];e=f['embedding'] if dim==0 else (f['embedding']-pm)@projection[:,:dim]
            if mode=='embedding':return e
            aux=f['aux']
            if mode in ('local','raw'):
                keep=[i for i in range(aux.shape[-1]) if i not in (22,23,24)]
                return np.concatenate([e,aux[...,keep]],-1)
            return np.concatenate([e,e-e.mean(1,keepdims=True),aux,f['center_aux']],-1)
        x,vx=build(data['train']),build(data['validation']);mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.1)
        x=torch.from_numpy((x-mean)/scale);vx=torch.from_numpy((vx-mean)/scale);ty=torch.from_numpy(y)
        net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
        opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.05)
        ids=np.arange(len(y));best=None;history=[];start=time.perf_counter()
        for epoch in range(1,61):
            net.train()
            for ii in np.array_split(rng.permutation(ids),len(ids)//128):
                s=net(x[ii]).squeeze(-1);target=ty[ii]
                # Class reweighting is fixed from training prevalence, not validation/test.
                weight=torch.where(target>0,1.,3.)
                loss=(F.binary_cross_entropy_with_logits(s,target,reduction='none')*weight).mean()
                if loss_kind=='hybrid':
                    m=target[:,:,None]*(1-target[:,None,:]);pair=((F.softplus(-(s[:,:,None]-s[:,None,:]))*m).sum((1,2))/m.sum((1,2)).clamp_min(1)).mean()
                    loss=loss+.5*pair
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            net.eval()
            with torch.no_grad():vs=net(vx).squeeze(-1).numpy()
            ok=vy[np.arange(len(vy)),vs.argmax(1)];key=(int(ok[vhard].sum()),int(ok.sum()),-epoch)
            history.append(dict(epoch=epoch,hard_correct=key[0],correct=key[1]))
            if best is None or key>best[0]:best=key,{k:v.detach().clone() for k,v in net.state_dict().items()},vs.copy(),epoch
        net.load_state_dict(best[1])
        class W:pass
        w=W();w.net=net;export_model(w,mean,scale,root/f'{name}.npz');scores[name]=best[2]
        one=vy.sum(1)==1;ok=vy[np.arange(len(vy)),best[2].argmax(1)]
        rec=dict(name=name,mode=mode,dim=dim,loss=loss_kind,seed=seed,epoch=best[3],
                 hard_accuracy=100*best[0][0]/vhard.sum(),accuracy=100*best[0][1]/len(vy),one_correct_accuracy=float(ok[one].mean()*100),
                 seconds=time.perf_counter()-start,bytes=(root/f'{name}.npz').stat().st_size,history=history)
        records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
        (root/'binary_records.json').write_text(json.dumps(records,indent=2)+'\n')
        np.savez(root/'binary_scores.npz',**scores,labels=vy,hard=vhard)
if __name__=='__main__':main()
