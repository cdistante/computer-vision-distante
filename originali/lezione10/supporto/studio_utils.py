"""Supporto fornito: misure, figure, export e prove controllate. Computer Vision L10."""
from pathlib import Path
import copy,csv,json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':180})
TEAL='#087F8C'; CORAL='#C64547'; GRAY='#536379'
def metrics(cm):
    cm=np.asarray(cm); tp=np.diag(cm); true=cm.sum(1); pred=cm.sum(0)
    precision=np.divide(tp,pred,out=np.zeros(10,dtype=float),where=pred!=0)
    recall=np.divide(tp,true,out=np.zeros(10,dtype=float),where=true!=0)
    f1=np.divide(2*tp,true+pred,out=np.zeros(10,dtype=float),where=(true+pred)!=0)
    return {'N':int(cm.sum()),'accuracy':float(tp.sum()/cm.sum()),'precision_macro':float(precision.mean()),
            'recall_macro':float(recall.mean()),'macro_f1':float(f1.mean()),'support':true.tolist(),
            'per_class_f1':f1.tolist(),'zero_denominator_classes':np.flatnonzero((true==0)|(pred==0)).tolist()}
def evaluate(model,loader,criterion):
    model.eval(); total=0.; seen=0; truth=[]; predictions=[]; scores=[]
    with torch.no_grad():
        for x,y in loader:
            z=model(x); loss=criterion(z,y)
            total+=float(loss)*len(y); seen+=len(y)
            truth.append(y.numpy()); predictions.append(z.argmax(1).numpy());scores.append(z.numpy())
    target=np.concatenate(truth); pred=np.concatenate(predictions); logits=np.concatenate(scores)
    cm=np.bincount(10*target+pred,minlength=100).reshape(10,10)
    return {**metrics(cm),'loss':total/seen,'confusion':cm.tolist(),'target':target.tolist(),'pred':pred.tolist(),'logits':logits.tolist()}
def save_figure(fig,out,name):
    fig.savefig(Path(out)/name,bbox_inches='tight'); plt.show(); plt.close(fig)
def plot_data(images,targets,ids,out):
    chosen=[int(next(i for i in ids if targets[i]==k)) for k in range(10)]
    fig,ax=plt.subplots(2,5,figsize=(12,4.8))
    for a,i in zip(ax.flat,chosen):
        a.imshow(images[i],cmap='gray',vmin=0,vmax=16,interpolation='nearest');a.set_title(f'Classe {targets[i]} · ID {i}',fontsize=13);a.axis('off')
    fig.suptitle('Digits reali del training · intensità 0–16',fontsize=16);fig.tight_layout()
    save_figure(fig,out,'01_digits.png')
def plot_scaling(images,targets,i,mean,scale,out):
    raw=images[i]; standardized=((raw.reshape(64)-mean)/scale).reshape(8,8)
    fig,axs=plt.subplots(1,2,figsize=(10,4.1))
    a=axs[0].imshow(raw,cmap='gray',vmin=0,vmax=16,interpolation='nearest');fig.colorbar(a,ax=axs[0],label='Intensità');axs[0].set_title(f'Originale · ID {i}, target {targets[i]}')
    vmax=max(abs(standardized.min()),abs(standardized.max()))
    b=axs[1].imshow(standardized,cmap='coolwarm',vmin=-vmax,vmax=vmax,interpolation='nearest');fig.colorbar(b,ax=axs[1],label='Z-score per pixel');axs[1].set_title('Statistiche del solo training')
    for ax in axs:ax.set_xticks([0,3,7]);ax.set_yticks([0,3,7]);ax.set_xlabel('Colonna');ax.set_ylabel('Riga')
    fig.tight_layout();save_figure(fig,out,'02_scaling.png')
def plot_history(history,best,out):
    e=[h['epoch']for h in history]
    for name,key,label in [('03_loss.png','loss','Cross-entropy media [nat/immagine]'),('04_f1.png','macro_f1','Macro-F1 (10 classi)')]:
        fig,ax=plt.subplots(figsize=(10.5,4.5))
        for prefix,color,label2 in [('train_eval',TEAL,'train_eval a pesi fissi'),('validation',CORAL,'validation')]:
            ax.plot(e,[h[prefix+'_'+key]for h in history],color=color,lw=2.5,label=label2)
        ax.axvline(best,color=GRAY,ls='--',label=f'Checkpoint: epoca {best}')
        ax.set(xlabel='Epoca',ylabel=label,xlim=(1,30));ax.set_xticks([1,5,10,15,20,25,30]);ax.grid(alpha=.2);ax.legend(fontsize=12);fig.tight_layout();save_figure(fig,out,name)
def plot_confusion(cm,out):
    cm=np.asarray(cm);fig,ax=plt.subplots(figsize=(7.8,6.3));im=ax.imshow(cm,cmap='Blues',vmin=0)
    for i in range(10):
        for j in range(10):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white'if cm[i,j]>cm.max()/2 else '#101827',fontsize=12)
    ax.set(xticks=range(10),yticks=range(10),xlabel='Classe predetta',ylabel='Classe vera',title='Test L9 riusato · conteggi su 360 immagini');fig.colorbar(im,ax=ax,label='Numero di immagini');fig.tight_layout();save_figure(fig,out,'05_confusion.png')
def plot_errors(images,targets,ids,pred,out):
    pred=np.asarray(pred);sel=np.flatnonzero(targets[ids]!=pred);sel=sel[np.argsort(ids[sel])][:8]
    fig,axs=plt.subplots(2,4,figsize=(11.5,5.2))
    for ax in axs.flat:ax.axis('off')
    for ax,j in zip(axs.flat,sel):
        i=ids[j];ax.imshow(images[i],cmap='gray',vmin=0,vmax=16,interpolation='nearest');ax.set_title(f'ID {i}: vero {targets[i]}, predetto {pred[j]}',fontsize=13)
    fig.suptitle('Errori in ordine di ID · primi 8, senza selezione estetica',fontsize=16);fig.tight_layout();save_figure(fig,out,'06_errors.png')
def challenge(choice,model,dataset,criterion,lr):
    model=copy.deepcopy(model);model.eval();loader=DataLoader(dataset,batch_size=64,shuffle=False)
    if choice=='A':
        (x1,y1),(x2,y2)=list(loader)[:2]
        grads=[]
        for x,y in [(x1,y1),(x2,y2)]:
            model.zero_grad(set_to_none=True);criterion(model(x),y).backward()
            grads.append(torch.cat([p.grad.flatten().clone() for p in model.parameters()]))
        model.zero_grad(set_to_none=True);criterion(model(x1),y1).backward();criterion(model(x2),y2).backward()
        accumulated=torch.cat([p.grad.flatten().clone() for p in model.parameters()])
        error=float(torch.max(torch.abs(accumulated-(grads[0]+grads[1]))))
        assert error<1e-6
        return {'challenge':'A','weights_fixed':True,'norm_g1':float(grads[0].norm()),'norm_g2':float(grads[1].norm()),'norm_sum':float(accumulated.norm()),'max_abs_error_sum':error}
    if choice=='B':
        results={}
        for batch in [64,37]:
            means=[];sizes=[]
            with torch.no_grad():
                for x,y in DataLoader(dataset,batch_size=batch,shuffle=False):means.append(float(criterion(model(x),y)));sizes.append(len(y))
            results[str(batch)]={'N':sum(sizes),'weighted':float(np.average(means,weights=sizes)),'unweighted':float(np.mean(means)),'last_batch':sizes[-1]}
        return {'challenge':'B','real_validation':results,'constructed_example':{'weighted':400/360,'unweighted':7/6}}
    if choice=='C':
        alias=model.state_dict();snapshot=copy.deepcopy(alias);key=next(iter(alias));initial=snapshot[key].flatten()[0].item()
        with torch.no_grad():next(model.parameters()).view(-1)[0].add_(1.)
        return {'challenge':'C','parameter':key,'initial':initial,'alias_after':alias[key].flatten()[0].item(),'copy_after':snapshot[key].flatten()[0].item(),'perturbation':1.0}
    raise ValueError('Scegli A, B o C')
def export_run(out,config,indices,mean,scale,run,final,repro,challenge_result):
    out=Path(out)
    (out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    with (out/'history.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(run['history'][0]));writer.writeheader();writer.writerows(run['history'])
    payload={k:v for k,v in final.items() if k not in ['logits']};payload.update({'best_epoch':run['best_epoch'],'best_validation_macro_f1':run['best_f1'],'replication':repro,'seconds_first':run['seconds']})
    (out/'metriche_finali.json').write_text(json.dumps(payload,indent=2))
    np.savetxt(out/'confusion_matrix.csv',np.asarray(final['confusion']),fmt='%d',delimiter=',')
    (out/'challenge.json').write_text(json.dumps(challenge_result,indent=2))
    torch.save({'state_dict':run['state'],'epoch':run['best_epoch'],'validation_macro_f1':run['best_f1'],
                'mean':torch.from_numpy(mean.copy()),'scale':torch.from_numpy(scale.copy()),
                'split_indices':{k:torch.from_numpy(v.copy())for k,v in indices.items()},'config':config},out/'best_checkpoint.pt')
    np.savez(out/'predizioni.npz',target=final['target'],pred=final['pred'],logits=final['logits'],test_idx=indices['test'])
