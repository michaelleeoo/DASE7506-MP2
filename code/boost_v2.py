"""Supervised small neural projection followed by a groupwise boosted verifier."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from catboost import CatBoostRanker,Pool
from train import labels,export_model
from common import load_inputs,parse_answer
from features_v2 import matrix

def main():
    root=Path('runs/improve_v9');root.mkdir(exist_ok=True);data={}
    with np.load('runs/improve_v3/pca.npz') as z:pm,projection=z['mean'],z['projection']
    for split in ('train','validation'):
        bank,_=load_inputs(split)
        with np.load(Path('runs/improve_v3')/f'{split}_features.npz') as z:f={k:z[k] for k in z.files}
        y=labels(split,bank);hard=np.array([len(set(a for a in [parse_answer(c['text']) for c in b['candidates']] if a is not None))>=2 for b in bank])
        data[split]=dict(f=f,y=y,hard=hard)
    y,vy=data['train']['y'],data['validation']['y'];vhard=data['validation']['hard'];records=[];scores={}
    for mode,loss_kind,epochs,seed in [('context','pair',2,17),('context','pair',5,43),('context','bce',4,29),('pca','pair',3,17),('pca','bce',6,43)]:
        name=f'boost_{mode}_{loss_kind}_{epochs}_{seed}';torch.manual_seed(seed);rng=np.random.default_rng(seed)
        def build(d):
            f=d['f']
            if mode!='pca':return matrix(f,mode)
            e=(f['embedding']-pm)@projection[:,:128]
            return np.concatenate([e,e-e.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
        x,vx=build(data['train']),build(data['validation']);mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.1)
        x=torch.from_numpy((x-mean)/scale);vx=torch.from_numpy((vx-mean)/scale);ty=torch.from_numpy(y)
        net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,1));opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
        ids=np.arange(len(y)) if loss_kind=='bce' else np.flatnonzero(y.sum(1)<4)
        start=time.perf_counter()
        for _ in range(epochs):
            net.train()
            for ii in np.array_split(rng.permutation(ids),len(ids)//128):
                s=net(x[ii]).squeeze(-1);target=ty[ii]
                if loss_kind=='bce':loss=F.binary_cross_entropy_with_logits(s,target)
                else:
                    m=target[:,:,None]*(1-target[:,None,:]);loss=((F.softplus(-(s[:,:,None]-s[:,None,:]))*m).sum((1,2))/m.sum((1,2)).clamp_min(1)).mean()
                opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            h=F.relu(net[0](x)).numpy();vh=F.relu(net[0](vx)).numpy()
            ns=net(x).numpy();vns=net(vx).numpy()
        class W:pass
        w=W();w.net=net;export_model(w,mean,scale,root/f'{name}_projection.npz')
        f,vf=data['train']['f'],data['validation']['f']
        tx=np.concatenate([h,h-h.mean(1,keepdims=True),ns,ns-ns.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
        tvx=np.concatenate([vh,vh-vh.mean(1,keepdims=True),vns,vns-vns.mean(1,keepdims=True),vf['aux'],vf['center_aux']],-1)
        mask=y.sum(1)<4
        pool=Pool(tx[mask].reshape(-1,tx.shape[-1]),y[mask].ravel(),group_id=np.repeat(np.arange(mask.sum()),4))
        model=CatBoostRanker(iterations=2200,depth=4,learning_rate=.035,loss_function='YetiRank',
                  thread_count=4,random_seed=seed,l2_leaf_reg=10,border_count=64,verbose=False,allow_writing_files=False)
        model.fit(pool);best=None
        for n in range(100,2201,100):
            s=model.predict(tvx.reshape(-1,tvx.shape[-1]),ntree_end=n,thread_count=4).reshape(-1,4)
            ok=vy[np.arange(len(vy)),s.argmax(1)];key=(int(ok[vhard].sum()),int(ok.sum()),-n)
            if best is None or key>best[0]:best=key,n,s
        model.shrink(best[1]);model.save_model(str(root/f'{name}.cbm'));scores[name]=best[2]
        rec=dict(name=name,mode=mode,loss_projection=loss_kind,projection_epochs=epochs,seed=seed,trees=best[1],
                 hard_accuracy=100*best[0][0]/vhard.sum(),accuracy=100*best[0][1]/len(vy),
                 bytes=sum((root/f).stat().st_size for f in (name+'.cbm',name+'_projection.npz')),
                 seconds=time.perf_counter()-start)
        records.append(rec);print(json.dumps(rec),flush=True)
        (root/'boost_records.json').write_text(json.dumps(records,indent=2)+'\n')
        np.savez(root/'boost_scores.npz',**scores,labels=vy,hard=vhard)
if __name__=='__main__':main()
