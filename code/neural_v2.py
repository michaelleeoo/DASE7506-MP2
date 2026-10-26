"""Train small verifiers with train-only PCA, harder-group weighting and marginal loss."""
import os
os.environ['MKL_CBWR']='COMPATIBLE'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import argparse
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from train import labels,export_model
from common import load_inputs,parse_answer
from features_v2 import matrix

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('runs/improve_v2'))
    parser.add_argument('--features',choices=['v2','v3'],default='v2')
    args=parser.parse_args()
    root=args.output;root.mkdir(exist_ok=True)
    data={}
    for split in ('train','validation'):
        with np.load(root/f'{split}_features.npz') as z:fs={k:z[k] for k in z.files}
        bank,arrays=load_inputs(split);y=labels(split,bank)
        hard=np.array([len(set(a for a in [parse_answer(c['text']) for c in b['candidates']] if a is not None))>=2 for b in bank])
        data[split]=(fs,y,hard)
    fs,y,hard=data['train'];vfs,vy,vhard=data['validation']
    path=root/'pca.npz'
    if not path.exists():
        pca=PCA(n_components=256,svd_solver='randomized',random_state=314)
        pca.fit(fs['embedding'].reshape(-1,1024))
        # Fractional whitening avoids amplifying small singular directions too strongly.
        projection=(pca.components_.T/np.maximum(pca.explained_variance_,.01)**.25).astype(np.float32)
        np.savez(path,mean=pca.mean_.astype(np.float32),projection=projection)
    with np.load(path) as z:pm,projection=z['mean'],z['projection']
    def build(f,mode):
        if mode=='pca':
            e=(f['embedding']-pm)@projection
            return np.concatenate([e,e-e.mean(1,keepdims=True),f['aux'],f['center_aux']],axis=-1)
        return matrix(f,mode)
    candidates=[];scores={}
    specs=[('pca',128,'marginal',s) for s in (17,29,43)]+[
           ('context',128,'marginal',17),('context',128,'marginal',29),
           ('raw',128,'marginal',17),('delta',128,'marginal',17),
           ('pca',128,'pair',17),('pca',128,'list',17),
           ('pca',256,'marginal',17),('context',256,'marginal',43),
           ('pca',64,'marginal',71)]
    if args.features=='v3':
        specs=[('pca',128,'list',s) for s in (17,29,43,71)]+[
               ('context',64,'pair',17),('context',64,'pair',43),
               ('pca',128,'marginal',17),('raw',64,'list',17)]
    for mode,width,loss_kind,seed in specs:
        name=f'nn_{mode}_{width}_{loss_kind}_{seed}';torch.manual_seed(seed);rng=np.random.default_rng(seed)
        x,vx=build(fs,mode),build(vfs,mode)
        mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.1)
        x=torch.from_numpy((x-mean)/scale);vx=torch.from_numpy((vx-mean)/scale);ty=torch.from_numpy(y)
        net=nn.Sequential(nn.Linear(x.shape[-1],width),nn.ReLU(),nn.Dropout(.25),nn.Linear(width,1))
        optim=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.05)
        ids=np.flatnonzero(hard)
        best=None;history=[];start=time.perf_counter()
        for epoch in range(1,61):
            net.train()
            for ii in np.array_split(rng.permutation(ids),max(1,len(ids)//96)):
                s=net(x[ii]).squeeze(-1);target=ty[ii]
                if loss_kind=='marginal':
                    loss=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(target==0,-1e9),1))
                elif loss_kind=='list':loss=-((target/target.sum(1,keepdim=True))*F.log_softmax(s,1)).sum(1)
                else:
                    m=target[:,:,None]*(1-target[:,None,:])
                    loss=(F.softplus(-(s[:,:,None]-s[:,None,:]))*m).sum((1,2))/m.sum((1,2)).clamp_min(1)
                # Emphasize the regime where exactly one candidate is correct.
                loss=(loss*(1+1.0*(target.sum(1)==1))).mean()
                optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optim.step()
            net.eval()
            with torch.no_grad():vs=net(vx).squeeze(-1).numpy()
            ok=vy[np.arange(len(vy)),vs.argmax(1)]
            key=(float(ok[vhard].mean()),float(ok.mean()),-epoch)
            history.append(dict(epoch=epoch,hard=key[0]*100,accuracy=key[1]*100))
            if best is None or key>best[0]:best=(key,{k:v.detach().clone() for k,v in net.state_dict().items()},vs.copy(),epoch)
        net.load_state_dict(best[1])
        class Wrapper:pass
        wrap=Wrapper();wrap.net=net
        export_model(wrap,mean,scale,root/f'{name}.npz')
        scores[name]=best[2]
        rec=dict(name=name,mode=mode,width=width,loss=loss_kind,seed=seed,epoch=best[3],
                 hard_accuracy=best[0][0]*100,accuracy=best[0][1]*100,seconds=time.perf_counter()-start,
                 bytes=(root/f'{name}.npz').stat().st_size,history=history)
        candidates.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
        (root/'neural_records.json').write_text(json.dumps(candidates,indent=2)+'\n')
        np.savez(root/'neural_scores.npz',**scores,labels=vy,hard=vhard)

if __name__=='__main__':main()
