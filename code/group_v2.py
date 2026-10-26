"""Rank distinct numeric answers using pooled traces, rather than rank traces independently."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train import labels
from common import load_inputs,parse_answer

def agreement_mask(bank):
    result=np.zeros((len(bank),4,4),np.float32)
    for i,b in enumerate(bank):
        answers=[parse_answer(c['text']) for c in b['candidates']]
        for j,a in enumerate(answers):
            for k,c in enumerate(answers):result[i,j,k]=float(j==k or (a is not None and a==c))
    return result

class AnswerVerifier(nn.Module):
    def __init__(self,dim,width):
        super().__init__();self.enc=nn.Linear(dim,width);self.mid=nn.Linear(width*5+1,64);self.out=nn.Linear(64,1);self.drop=nn.Dropout(.3)
    def forward(self,x,mask):
        h=F.relu(self.enc(x));h=self.drop(h)
        count=mask.sum(2,keepdim=True)
        group=(mask/count)@h
        groupmax=h[:,None,:,:].expand(-1,4,-1,-1).masked_fill(mask[:,:,:,None]==0,-1e9).amax(2)
        mean=h.mean(1,keepdim=True).expand_as(h);mx=h.amax(1,keepdim=True).expand_as(h)
        z=torch.cat([group,groupmax,group-mean,group*mean,mx,count/4],-1)
        return self.out(self.drop(F.relu(self.mid(z)))).squeeze(-1)

def numpy_forward(x,mask,z):
    h=np.maximum(x@z['enc_w']+z['enc_b'],0);count=mask.sum(2,keepdims=True)
    group=(mask/count)@h
    groupmax=np.where(mask[:,:,:,None]>0,h[:,None,:,:],-1e9).max(2)
    mean=h.mean(1,keepdims=True);maximum=np.broadcast_to(h.max(1,keepdims=True),h.shape)
    combined=np.concatenate([group,groupmax,group-mean,group*mean,maximum,count/4],-1)
    return (np.maximum(combined@z['mid_w']+z['mid_b'],0)@z['out_w']+z['out_b'])[...,0]

def main():
    root=Path('runs/improve_v7');root.mkdir(exist_ok=True);data={}
    with np.load('runs/improve_v3/pca.npz') as z:pm,projection=z['mean'],z['projection']
    for split in ('train','validation'):
        bank,_=load_inputs(split)
        with np.load(Path('runs/improve_v3')/f'{split}_features.npz') as z:f={k:z[k] for k in z.files}
        y=labels(split,bank);mask=agreement_mask(bank);hard=mask.sum(2).max(1)<4
        data[split]=dict(f=f,y=y,mask=mask,hard=hard)
    y,vy=data['train']['y'],data['validation']['y'];vhard=data['validation']['hard'];records=[];scores={}
    specs=[(128,128,'group',s) for s in (17,29,43,71)]+[(256,128,'group',17),(128,64,'group',43),
            (128,128,'pair',17),(64,128,'group',71)]
    for dim,width,loss_kind,seed in specs:
        name=f'group_{dim}_{width}_{loss_kind}_{seed}';torch.manual_seed(seed);rng=np.random.default_rng(seed)
        def build(d):
            f=d['f'];e=(f['embedding']-pm)@projection[:,:dim]
            return np.concatenate([e,e-e.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
        x,vx=build(data['train']),build(data['validation']);mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.05)
        x=torch.from_numpy((x-mean)/scale);vx=torch.from_numpy((vx-mean)/scale);ty=torch.from_numpy(y)
        mask=torch.from_numpy(data['train']['mask']);vmask=torch.from_numpy(data['validation']['mask'])
        net=AnswerVerifier(x.shape[-1],width);opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.05)
        ids=np.flatnonzero(y.sum(1)<4);best=None;history=[];start=time.perf_counter()
        for epoch in range(1,81):
            net.train()
            for ii in np.array_split(rng.permutation(ids),len(ids)//96):
                s=net(x[ii],mask[ii]);target=ty[ii]
                if loss_kind=='group':
                    adjusted=s-mask[ii].sum(2).log()
                    loss=torch.logsumexp(adjusted,1)-(s*target).sum(1)/target.sum(1)
                else:
                    pairmask=target[:,:,None]*(1-target[:,None,:]);loss=(F.softplus(-(s[:,:,None]-s[:,None,:]))*pairmask).sum((1,2))/pairmask.sum((1,2)).clamp_min(1)
                loss=(loss*(1+(target.sum(1)==1))).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            net.eval()
            with torch.no_grad():vs=net(vx,vmask).numpy()
            ok=vy[np.arange(len(vy)),vs.argmax(1)];key=(int(ok[vhard].sum()),int(ok.sum()),-epoch)
            history.append(dict(epoch=epoch,hard_correct=key[0],correct=key[1]))
            if best is None or key>best[0]:best=key,{k:v.detach().clone() for k,v in net.state_dict().items()},vs.copy(),epoch
        net.load_state_dict(best[1]);weights=dict(mean=mean,scale=scale)
        for prefix,layer in [('enc',net.enc),('mid',net.mid),('out',net.out)]:
            weights[prefix+'_w']=layer.weight.detach().numpy().T.copy();weights[prefix+'_b']=layer.bias.detach().numpy().copy()
        np.savez(root/f'{name}.npz',**weights);scores[name]=best[2]
        np.testing.assert_allclose(numpy_forward(vx.numpy(),data['validation']['mask'],weights),best[2],atol=2e-4,rtol=2e-4)
        one=vy.sum(1)==1;ok=vy[np.arange(len(vy)),best[2].argmax(1)]
        rec=dict(name=name,dim=dim,kind_features='base',width=width,loss=loss_kind,seed=seed,epoch=best[3],
                 hard_accuracy=100*best[0][0]/vhard.sum(),accuracy=100*best[0][1]/len(vy),one_correct_accuracy=float(ok[one].mean()*100),
                 seconds=time.perf_counter()-start,bytes=(root/f'{name}.npz').stat().st_size,history=history)
        records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
        (root/'group_records.json').write_text(json.dumps(records,indent=2)+'\n')
        np.savez(root/'group_scores.npz',**scores,labels=vy,hard=vhard)

if __name__=='__main__':main()
