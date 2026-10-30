"""Strumenti forniti per Computer Vision, CNN Training Studio (L12)."""
from pathlib import Path
import copy,csv,json,time,hashlib,platform
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110,'savefig.dpi':180})
TEAL='#087F8C'; CORAL='#C64547'; GRAY='#536379'
def configure(seed=20261022):
    np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def clone_state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}
def same_state(a,b):return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
def init_pair(cls,seed=20261022):
    configure(seed);a=cls(True);b=cls(False)
    for m in a.modules():
        if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu')
        elif isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    common={k:v.clone() for k,v in a.state_dict().items() if k in b.state_dict()}
    b.load_state_dict(common);assert all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items())
    return a,b

def loader(dataset,seed=20261022,shuffle=False,batch=64):
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),num_workers=0,drop_last=False)
def metrics(cm):
    cm=np.asarray(cm);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(10,dtype=float),where=den!=0)
    return {'N':int(cm.sum()),'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean())}
def evaluate(model,ds,batch=64):
    mode=model.training;before=clone_state(model);model.eval();total=0.;ys=[];zs=[]
    with torch.no_grad():
        for x,y,ids in loader(ds,batch=batch):
            z=model(x);total+=float(nn.functional.cross_entropy(z,y,reduction='sum'));ys.append(y.numpy());zs.append(z.numpy())
    y=np.concatenate(ys);z=np.concatenate(zs);pred=z.argmax(1);cm=np.bincount(10*y+pred,minlength=100).reshape(10,10)
    assert same_state(before,clone_state(model)), 'La valutazione ha modificato lo stato'
    model.train(mode)
    return {**metrics(cm),'loss':total/len(y),'confusion':cm.tolist(),'pred':pred.tolist(),'logits':z.tolist()}
def brief(m):return {k:m[k] for k in ('loss','accuracy','macro_f1')}
def run(model,ds,seed=20261022,epochs=30):
    opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    train=loader(ds['train'],seed,True);history=[];probes=[];ids_history=[];best=-1.;best_state=None;t0=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();total=0.;seen=0;epoch_ids=[]
        for batchno,(x,y,ids) in enumerate(train):
            inspect=batchno==0 and epoch in (1,epochs);stats={};handles=[]
            if inspect:
                before={n:p.detach().clone() for n,p in model.named_parameters()}
                def hook(name):
                    def f(m,inp,out):
                        t=out.detach();stats[name]={'mean':float(t.mean()),'std':float(t.std(unbiased=False)),'zero_fraction':float((t==0).float().mean())}
                    return f
                for name in ['conv1','relu1','conv2','relu2']:handles.append(getattr(model,name).register_forward_hook(hook(name)))
            opt.zero_grad(set_to_none=True);z=model(x);loss=nn.functional.cross_entropy(z,y);loss.backward()
            gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()} if inspect else {}
            opt.step()
            for h in handles:h.remove()
            if inspect:
                changes={}
                for n,p in model.named_parameters():
                    norm=float(before[n].norm());delta=float((p.detach()-before[n]).norm())
                    changes[n]={'weight_norm_before':norm,'gradient_norm':gradients[n],'update_norm':delta,'ratio':delta/norm if norm else None}
                probes.append({'epoch':epoch,'batch':0,'mode':'train','ids':ids.tolist(),'activation':stats,'parameters':changes})
            total+=float(loss.detach())*len(y);seen+=len(y);epoch_ids.extend(ids.tolist())
        tr=evaluate(model,ds['train']);va=evaluate(model,ds['val'])
        history.append({'epoch':epoch,'online_loss':total/seen,**{'train_eval_'+k:v for k,v in brief(tr).items()},**{'validation_'+k:v for k,v in brief(va).items()}})
        ids_history.append(hashlib.sha256(np.asarray(epoch_ids,dtype=np.int64).tobytes()).hexdigest())
        if va['macro_f1']>best:
            best=va['macro_f1'];best_epoch=epoch;best_state=clone_state(model)
    elapsed=time.perf_counter()-t0;last_state=clone_state(model);model.load_state_dict(best_state);besttr=evaluate(model,ds['train']);bestva=evaluate(model,ds['val'])
    return {'history':history,'best_epoch':best_epoch,'best_train':besttr,'best_val':bestva,'last':history[-1],'seconds_training_and_eval':elapsed,'steps':epochs*len(train),'batch_hashes':ids_history,'probes':probes,'best_state':best_state,'last_state':last_state}
def tiny_fit(model,ds):
    x,y,ids=ds.tensors;chosen=[]
    for k in range(10):chosen.extend(sorted(torch.where(y==k)[0].tolist(),key=lambda j:int(ids[j]))[:2])
    x=x[chosen];y=y[chosen];model=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0);history=[];model.train();t0=time.perf_counter()
    for step in range(201):
        opt.zero_grad(set_to_none=True);z=model(x);loss=nn.functional.cross_entropy(z,y)
        if step in [0,1,10,50,100,200]:history.append({'step':step,'loss':float(loss.detach()),'accuracy':float((z.argmax(1)==y).float().mean())})
        if step<200:loss.backward();opt.step()
    return {'ids':ids[chosen].tolist(),'history':history,'seconds':time.perf_counter()-t0}
def challenges(choice,model,ds,cls,seed):
    if choice=='C01':
        good=copy.deepcopy(model);bad=copy.deepcopy(model);before=clone_state(bad);bad.train()
        with torch.no_grad():
            for x,y,ids in loader(ds['val']):bad(x)
        after=clone_state(bad);changed=[k for k in before if not torch.equal(before[k],after[k])]
        correct=evaluate(good,ds['val']);contaminated=evaluate(bad,ds['val'])
        return {'changed_state_keys':changed,'parameters_unchanged':all(torch.equal(before[k],after[k]) for k,_ in bad.named_parameters()),'clean_f1':correct['macro_f1'],'contaminated_f1':contaminated['macro_f1'],'clean_state_invariant':same_state(clone_state(good),clone_state(model))}
    if choice=='C02':
        a,b=init_pair(cls,seed);fresh=cls(False);common_equal=all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items());wrong_equal=all(torch.equal(a.state_dict()[k],v) for k,v in fresh.state_dict().items())
        dl=loader(ds['train'],seed,True);first=next(iter(dl))[2];advanced=next(iter(dl))[2];freshids=next(iter(loader(ds['train'],seed,True)))[2]
        return {'paired_common_weights':common_equal,'fresh_unpaired_weights_equal':wrong_equal,'fresh_loader_same_ids':torch.equal(first,freshids),'advanced_loader_same_ids':torch.equal(first,advanced)}
    if choice=='C03':
        x,y,_=next(iter(loader(ds['train'])));m=copy.deepcopy(model);before=clone_state(m);m.train()
        with torch.no_grad():trainloss=float(nn.functional.cross_entropy(m(x),y))
        m.load_state_dict(before);m.eval()
        with torch.no_grad():evalloss=float(nn.functional.cross_entropy(m(x),y))
        return {'same_batch_train_mode_loss':trainloss,'same_batch_eval_mode_loss':evalloss,'original_state_unchanged':same_state(before,clone_state(model))}
    raise ValueError('Scegli C01, C02 o C03')
def savefig(fig,out,name):
    fig.tight_layout();fig.savefig(Path(out)/name,bbox_inches='tight');plt.show();plt.close(fig)
def data_fig(images,y,ids,out):
    chosen=[min(i for i in ids if y[i]==k) for k in range(10)];fig,axs=plt.subplots(2,5,figsize=(12,4.7))
    for ax,i in zip(axs.flat,chosen):ax.imshow(images[i],cmap='gray',vmin=0,vmax=16,interpolation='nearest');ax.set_title(f'Classe {y[i]} · ID {i}');ax.axis('off')
    savefig(fig,out,'01_digits.png')
def plot_tiny(tiny,out):
    fig,ax=plt.subplots(figsize=(10.5,4.4));h=tiny['history'];ax.plot([r['step']for r in h],[r['loss']for r in h],'o-',color=TEAL);ax.set(xlabel='Aggiornamenti Adam',ylabel='Cross-entropy [nat/immagine]',title='20 immagini training · B senza BN · misura reale');ax.grid(alpha=.2);savefig(fig,out,'02_tiny_fit.png')
def plots(a,b,out):
    for key,filename,ylabel in [('loss','03_loss_ab.png','Cross-entropy [nat/immagine]'),('macro_f1','04_f1_ab.png','Macro-F1 validation')]:
        fig,ax=plt.subplots(figsize=(11,4.5))
        for tag,r,col in [('A con BN',a,TEAL),('B senza BN',b,CORAL)]:
            h=r['history'];ax.plot([v['epoch']for v in h],[v['validation_'+key]for v in h],label=tag,color=col,lw=2.5)
            ax.axvline(r['best_epoch'],color=col,ls=':',alpha=.65)
        ax.set(xlabel='Epoca',ylabel=ylabel,xlim=(1,30));ax.set_xticks([1,5,10,15,20,25,30]);ax.legend();ax.grid(alpha=.2);savefig(fig,out,filename)
    fig,ax=plt.subplots(figsize=(11,4.5))
    for key,label,col in [('online_loss','Online train',GRAY),('train_eval_loss','train_eval',TEAL),('validation_loss','validation',CORAL)]:ax.plot([r['epoch'] for r in a['history']],[r[key] for r in a['history']],label=label,color=col,lw=2)
    ax.set(xlabel='Epoca',ylabel='Cross-entropy [nat/immagine]',title='A con BN · definizioni distinte');ax.legend();ax.grid(alpha=.2);savefig(fig,out,'05_loss_definitions.png')
def visual(initial,trained,images,ds,result,out):
    x,y,ids=ds['train'].tensors;i=int(torch.argmin(ids));sample=x[i:i+1];both=[copy.deepcopy(initial).eval(),copy.deepcopy(trained).eval()]
    weights=[m.conv1.weight.detach()[:,0].numpy() for m in both];lim=max(abs(w).max() for w in weights);fig,axs=plt.subplots(2,8,figsize=(13,4))
    for row in range(2):
        for col in range(8):
            ax=axs[row,col];im=ax.imshow(weights[row][col],cmap='coolwarm',vmin=-lim,vmax=lim,interpolation='nearest');ax.set_title(f'K{col}');ax.set_xticks([]);ax.set_yticks([])
        axs[row,0].set_ylabel('Prima' if row==0 else 'Dopo')
    fig.subplots_adjust(right=.85,wspace=.25,hspace=.4);cax=fig.add_axes([.88,.22,.018,.56]);fig.colorbar(im,cax=cax,label='Peso, scala comune');fig.savefig(Path(out)/'06_kernels.png',bbox_inches='tight');plt.show();plt.close(fig)
    feats=[]
    with torch.no_grad():
        for m in both:feats.append(m.relu1(m.bn1(m.conv1(sample)))[0].numpy())
    vmax=max(f.max() for f in feats);fig,axs=plt.subplots(2,8,figsize=(13,4))
    for row in range(2):
        for col in range(8):
            ax=axs[row,col];im=ax.imshow(feats[row][col],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');ax.set_title(f'C{col}');ax.set_xticks([]);ax.set_yticks([])
        axs[row,0].set_ylabel('Prima'if row==0 else 'Dopo')
    fig.suptitle(f'BN1 + ReLU · eval · stesso input training ID {int(ids[i])}, target {int(y[i])}');fig.subplots_adjust(right=.85,wspace=.25,hspace=.4);cax=fig.add_axes([.88,.22,.018,.56]);fig.colorbar(im,cax=cax,label='Attivazione, scala comune');fig.savefig(Path(out)/'07_feature_maps.png',bbox_inches='tight');plt.show();plt.close(fig)
    v_ids=ds['val'].tensors[2].numpy();v_y=ds['val'].tensors[1].numpy();pred=np.asarray(result['best_val']['pred']);bad=np.flatnonzero(v_y!=pred);bad=bad[np.argsort(v_ids[bad])][:3];fig,axs=plt.subplots(1,3,figsize=(10.5,3.7))
    for ax in axs:ax.axis('off')
    for ax,j in zip(axs,bad):ax.imshow(images[v_ids[j]],cmap='gray',vmin=0,vmax=16);ax.set_title(f'ID {v_ids[j]}\nVero {v_y[j]}, predetto {pred[j]}')
    fig.suptitle(f'A: primi {len(bad)} errori validation in ordine di ID');savefig(fig,out,'08_errors_val.png')
def serialize_run(r):return {k:v for k,v in r.items() if k not in ['best_state','last_state']}
def save_run(tag,r,out,mu,sigma,seed,data_hash,ds):
    out=Path(out);(out/f'risultati_{tag}.json').write_text(json.dumps(serialize_run(r),indent=2))
    with (out/f'history_{tag}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=r['history'][0]);w.writeheader();w.writerows(r['history'])
    torch.save({'state_dict':r['best_state'],'mean':mu,'std':sigma,'seed':seed,'best_epoch':r['best_epoch'],'data_sha256':data_hash,'train_ids':ds['train'].tensors[2],'val_ids':ds['val'].tensors[2],'config':{'batch_size':64,'epochs':30,'optimizer':'Adam','lr':.001,'batch_norm':tag=='A'}},out/f'checkpoint_{tag}.pt')
