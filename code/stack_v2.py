"""Question-level out-of-fold stacking. Never reads test inputs/references."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
from pathlib import Path
import json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier,CatBoostRanker,Pool
from train import labels
from group_v2 import AnswerVerifier,agreement_mask
from common import load_inputs,parse_answer
from features_v2 import matrix
from submission_v6 import forward_numpy,relational_forward

def probability(scores,valid):
    s=scores-scores.mean(1,keepdims=True)
    s/=np.maximum(np.sqrt(np.mean(s*s,1,keepdims=True)),.01)
    s=np.where(valid,s,-1e9);ex=np.exp(s-s.max(1,keepdims=True))
    return ex/ex.sum(1,keepdims=True)

def meta_input(logits,features,pm,projection,valid):
    parts=[]
    for s in logits:
        p=probability(s,valid)
        parts.extend([p[:,:,None],(p-p.max(1,keepdims=True))[:,:,None],
                      np.broadcast_to((-(p*np.log(np.maximum(p,1e-10))).sum(1))[:,None,None],(*p.shape,1))])
    average=np.mean([probability(s,valid) for s in logits],axis=0)
    parts.extend([average[:,:,None],features['aux'],features['center_aux']])
    return np.concatenate(parts,-1).astype(np.float32)

def main():
    root=Path('runs/improve_v14');root.mkdir(exist_ok=True);data={}
    with np.load('runs/improve_v3/pca.npz') as z:pm,projection=z['mean'],z['projection']
    for split in ('train','validation'):
        bank,_=load_inputs(split)
        with np.load(Path('runs/improve_v3')/f'{split}_features.npz') as z:f={k:z[k] for k in z.files}
        y=labels(split,bank);valid=np.array([[parse_answer(c['text']) is not None for c in b['candidates']] for b in bank])
        hard=np.array([len(set(a for a in [parse_answer(c['text']) for c in b['candidates']] if a is not None))>=2 for b in bank])
        data[split]=dict(f=f,y=y,valid=valid,hard=hard,mask=agreement_mask(bank))
    def input_for(f,kind):
        aux=f['aux']
        if kind=='binary':return np.concatenate([f['embedding'],aux[...,[i for i in range(aux.shape[-1]) if i not in (22,23,24)]]],-1)
        if kind=='boost':return matrix(f,'context')
        e=(f['embedding']-pm)@projection[:,:128]
        return np.concatenate([e,e-e.mean(1,keepdims=True),aux,f['center_aux']],-1)
    specs=[('binary',43,15,128),('group',71,3,128),('robust',103,100,128),('boost',29,4,128),('group64',43,7,64)]
    y=data['train']['y'];vy=data['validation']['y'];f=data['train']['f'];vf=data['validation']['f']
    oof=np.zeros((5,len(y),4),np.float32);full_validation=[]
    cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=6701)
    for mi,(kind,seed,epochs,width) in enumerate(specs):
        cache=root/f'oof_{mi}.npy'
        if cache.exists():oof[mi]=np.load(cache);continue
        x_all=input_for(f,'group' if kind=='group64' else kind)
        for fold,(train_idx,hold_idx) in enumerate(cv.split(np.arange(len(y)),y.sum(1))):
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            mean=x_all[train_idx].mean((0,1));scale=np.maximum(x_all[train_idx].std((0,1)),.1 if kind in ('binary','boost') else .05)
            x=torch.from_numpy((x_all-mean)/scale);ty=torch.from_numpy(y);mask=torch.from_numpy(data['train']['mask'])
            if kind.startswith('group'):net=AnswerVerifier(x.shape[-1],width)
            elif kind=='boost':net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,1))
            elif kind=='robust':net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
            else:net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
            lr=.001 if kind in ('robust','boost') else (.0007 if kind=='binary' else .0005)
            opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=.1 if kind=='robust' else .05)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=.00005) if kind=='robust' else None
            eligible=train_idx if kind in ('binary','boost') else train_idx[y[train_idx].sum(1)<4]
            average=None;count=0;tick=time.perf_counter()
            for epoch in range(1,epochs+1):
                net.train()
                for ii in np.array_split(rng.permutation(eligible),max(1,len(eligible)//128)):
                    s=net(x[ii],mask[ii]) if kind.startswith('group') else net(x[ii]).squeeze(-1)
                    target=ty[ii]
                    if kind in ('binary','boost'):
                        loss=F.binary_cross_entropy_with_logits(s,target,reduction='none')
                        if kind=='binary':loss=loss*torch.where(target>0,1.,3.)
                        loss=loss.mean()
                    elif kind.startswith('group'):
                        loss=(torch.logsumexp(s-mask[ii].sum(2).log(),1)-(s*target).sum(1)/target.sum(1))
                        loss=(loss*(1+(target.sum(1)==1))).mean()
                    else:
                        target=.95*target/target.sum(1,keepdim=True)+.05/4
                        loss=-(target*F.log_softmax(s,1)).sum(1).mean()
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
                if schedule:schedule.step()
                if kind=='robust' and epoch>=60 and epoch%5==0:
                    state={k:v.detach().clone() for k,v in net.state_dict().items()};count+=1
                    if average is None:average=state
                    else:
                        for k in state:average[k]+=(state[k]-average[k])/count
            if average:net.load_state_dict(average)
            net.eval()
            if kind=='boost':
                with torch.no_grad():h=F.relu(net[0](x)).numpy();ns=net(x).numpy()
                bx=np.concatenate([h,h-h.mean(1,keepdims=True),ns,ns-ns.mean(1,keepdims=True),f['aux'],f['center_aux']],-1)
                rank_ids=train_idx[y[train_idx].sum(1)<4]
                pool=Pool(bx[rank_ids].reshape(-1,bx.shape[-1]),y[rank_ids].ravel(),group_id=np.repeat(np.arange(len(rank_ids)),4))
                model=CatBoostRanker(iterations=300,depth=4,learning_rate=.035,loss_function='YetiRank',thread_count=4,
                                    random_seed=29,l2_leaf_reg=10,border_count=64,verbose=False,allow_writing_files=False)
                model.fit(pool);s=model.predict(bx[hold_idx].reshape(-1,bx.shape[-1]),thread_count=4).reshape(-1,4)
            else:
                with torch.no_grad():s=(net(x[hold_idx],mask[hold_idx]) if kind.startswith('group') else net(x[hold_idx]).squeeze(-1)).numpy()
            oof[mi,hold_idx]=s
            print('OOF',kind,fold,'seconds',round(time.perf_counter()-tick,2),flush=True)
        np.save(cache,oof[mi])
    # Full-fit frozen base learners: none of their validation scores entered OOF targets.
    bundle=Path('model_bundle_v6');config=json.loads((bundle/'config.json').read_text())
    for m in config['members']:
        kind=m['kind'];x=input_for(vf,'binary' if kind=='binary' else ('boost' if kind=='boost' else 'group'))
        if kind=='boost':
            with np.load(bundle/m['projection_file']) as z:
                xx=(x-z['mean'])/z['scale'];h=np.maximum(xx@z['w0']+z['b0'],0);ns=forward_numpy(xx,z)[:,:,None]
                bx=np.concatenate([h,h-h.mean(1,keepdims=True),ns,ns-ns.mean(1,keepdims=True),vf['aux'],vf['center_aux']],-1)
            model=CatBoostRanker();model.load_model(str(bundle/m['file']));s=model.predict(bx.reshape(-1,bx.shape[-1]),thread_count=4).reshape(-1,4)
        else:
            with np.load(bundle/m['file']) as z:
                xx=(x-z['mean'])/z['scale'];s=relational_forward(xx,z,'group',data['validation']['mask']) if kind=='group' else forward_numpy(xx,z)
        full_validation.append(s)
    tx=meta_input(oof,f,pm,projection,data['train']['valid']);vx=meta_input(full_validation,vf,pm,projection,data['validation']['valid'])
    mask=y.sum(1)<4;train_pool=Pool(tx[mask].reshape(-1,tx.shape[-1]),y[mask].ravel(),group_id=np.repeat(np.arange(mask.sum()),4))
    records=[];vscores={};hard=data['validation']['hard']
    base=np.mean([probability(s,data['validation']['valid']) for s in full_validation],axis=0)
    for loss,depth in [('YetiRank',3),('Logloss',3),('YetiRank',4)]:
        name=f'stack_{loss}_{depth}'
        args=dict(iterations=1200,depth=depth,learning_rate=.025,loss_function=loss,thread_count=4,
                  random_seed=1337,l2_leaf_reg=30,border_count=32,verbose=False,allow_writing_files=False)
        model=CatBoostClassifier(**args) if loss=='Logloss' else CatBoostRanker(**args)
        if loss=='Logloss':model.fit(tx[mask].reshape(-1,tx.shape[-1]),y[mask].ravel())
        else:model.fit(train_pool)
        best=None
        for trees in (100,200,400,600,800,1200):
            kwargs=dict(ntree_end=trees,thread_count=4)
            if loss=='Logloss':kwargs['prediction_type']='RawFormulaVal'
            logits=model.predict(vx.reshape(-1,vx.shape[-1]),**kwargs).reshape(-1,4)
            prob=probability(logits,data['validation']['valid'])
            for weight in (0.,.25,.5,.75,1.):
                s=weight*prob+(1-weight)*base;mass=(s*vy).sum(1)
                nll=float(-np.log(np.maximum(mass[hard],1e-10)).mean());ok=vy[np.arange(len(vy)),s.argmax(1)]
                key=(-nll,int(ok.sum()),-trees)
                if best is None or key>best[0]:best=key,trees,weight,s,logits
        model.shrink(best[1]);model.save_model(str(root/f'{name}.cbm'));vscores[name]=best[3]
        rec=dict(name=name,loss=loss,depth=depth,trees=best[1],stack_weight=best[2],nll=-best[0][0],accuracy=100*best[0][1]/len(vy))
        records.append(rec);print(json.dumps(rec),flush=True)
    (root/'stack_records.json').write_text(json.dumps(records,indent=2)+'\n')
    np.savez(root/'stack_scores.npz',**vscores,labels=vy,hard=hard)

if __name__=='__main__':main()
