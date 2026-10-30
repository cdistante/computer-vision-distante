"""Computer Vision 2026, L14. Protocollo CPU locale, senza uso del test."""
from pathlib import Path
import copy, hashlib, json, platform, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import timm
SEED=20261029
CONFIG={'seed':SEED,'batch_size':32,'lp_epochs':20,'ft_epochs':3,'lp_lr':1e-3,'head_lr':1e-3,'backbone_lr':1e-4,'weight_decay':0.0,'selection':'prima massima accuracy validation pulita, separatamente LP e FT','corruptions':['shift_right_1_pixel_zero_fill','contrast_0.5_black_anchor'],'test_evaluated':False}
WEIGHTS={'resnet18':('resnet18-f37072fd.pth','https://download.pytorch.org/models/resnet18-f37072fd.pth'),'deit_tiny':('deit_tiny_patch16_224-a1311bcf.pth','https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth')}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def setup():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(SEED);np.random.seed(SEED)
 return {'python':platform.python_version(),'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'timm':timm.__version__,'numpy':np.__version__,'platform':platform.platform(),'processor':platform.machine(),'device':'cpu','threads':1,'workers':0,'dtype':'float32','hosted_colab':'google.colab' in __import__('sys').modules}
def load_data(base):
 path=Path(base)/'dati/digits_cv2026.npz';a=np.load(path)
 # The test index is checked for split disjointness only. No test image is selected.
 sets=[set(a[k].tolist()) for k in ['train_idx','val_idx','test_idx']]
 assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);assert len(set.union(*sets))==1797
 data={k:torch.tensor(a[k].copy()) for k in ['images','target','train_idx','val_idx']}
 data['images']=data['images'].float()/16.;data['target']=data['target'].long();data['sha256']=sha(path)
 return data
MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1);STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)
def preprocess(x):
 assert x.ndim==3 and x.shape[-2:]==(8,8)
 rgb=x[:,None].repeat(1,3,1,1)
 rgb=F.interpolate(rgb,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
 return (rgb-MEAN)/STD

def make_model(name,base):
 torch.manual_seed(SEED)
 if name=='resnet18':
  m=resnet18(weights=None);state=torch.load(Path(base)/'pesi'/WEIGHTS[name][0],map_location='cpu',weights_only=True);m.load_state_dict(state,strict=True);dim=m.fc.in_features;m.fc=nn.Linear(dim,10)
 else:
  m=timm.create_model('deit_tiny_patch16_224',pretrained=False);state=torch.load(Path(base)/'pesi'/WEIGHTS[name][0],map_location='cpu',weights_only=True)['model'];m.load_state_dict(state,strict=True);dim=m.head.in_features;m.head=nn.Linear(dim,10)
 m.eval();return m,dim

def head_of(m,name):return m.fc if name=='resnet18' else m.head



def batch_orders(n,epochs,offset=0):
 g=torch.Generator().manual_seed(SEED+offset)
 return [torch.randperm(n,generator=g) for _ in range(epochs)]

def features(m,name,x,batch=32):
 out=[];t=time.perf_counter()
 with torch.no_grad():
  for a in range(0,len(x),batch):
   z=preprocess(x[a:a+batch]);h=m.forward_features(z) if name=='deit_tiny' else nn.Sequential(*list(m.children())[:-1])(z)
   h=m.forward_head(h,pre_logits=True) if name=='deit_tiny' else h.flatten(1)
   out.append(h)
 return torch.cat(out),time.perf_counter()-t

def metrics(y,logits):
 pred=logits.argmax(1);cm=torch.zeros((10,10),dtype=torch.long)
 for a,b in zip(y,pred):cm[a,b]+=1
 den=2*cm.diag()+cm.sum(0)-cm.diag()+cm.sum(1)-cm.diag()
 f1=torch.where(den>0,2*cm.diag()/den,0.)
 return {'n':len(y),'correct':int((pred==y).sum()),'accuracy':float((pred==y).double().mean()),'macro_f1':float(f1.double().mean()),'ce':float(F.cross_entropy(logits,y)),'confusion':cm.tolist(),'predictions':pred.tolist()}

def eval_model(m,x,y):
 m.eval();t=time.perf_counter();outs=[]
 with torch.no_grad():
  for a in range(0,len(x),CONFIG['batch_size']):outs.append(m(preprocess(x[a:a+CONFIG['batch_size']])))
 return metrics(y,torch.cat(outs)),time.perf_counter()-t

def clone_state(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}
def corrupt(x,kind):
 if kind=='shift_right_1_pixel_zero_fill':
  out=torch.zeros_like(x);out[:,:,1:]=x[:,:,:-1];return out
 if kind=='contrast_0.5_black_anchor':return .5*x
 raise ValueError(kind)

def measure_latency(m,x,repeats=7):
 # CPU synchronous, input preprocessing included, disk/model loading excluded.
 m.eval();vals=[]
 with torch.no_grad():
  for _ in range(3):m(preprocess(x[:1]))
  for _ in range(repeats):
   t=time.perf_counter();m(preprocess(x[:1]));vals.append(1000*(time.perf_counter()-t))
 return {'unit':'ms / immagine, batch1','median_ms':float(np.median(vals)),'q25_ms':float(np.quantile(vals,.25)),'q75_ms':float(np.quantile(vals,.75)),'repeats_ms':vals,'warmups':3,'repeats':repeats,'includes':'preprocessing + forward CPU','excludes':'lettura disco, caricamento pesi, cold start'}

def run_lp(name,m,data,out):
 configure_stage(m,name,'lp');ix=data['train_idx'];iv=data['val_idx'];x=data['images'];y=data['target'];head=head_of(m,name)
 ftr,ttr=features(m,name,x[ix]);fva,tva=features(m,name,x[iv]);torch.save({'train':ftr,'val':fva,'train_ids':ix,'val_ids':iv},out/f'features_{name}.pt')
 orders=batch_orders(len(ix),CONFIG['lp_epochs']);opt=torch.optim.Adam(head.parameters(),lr=CONFIG['lp_lr']);hist=[];best=-1;beststate=None;fitstart=time.perf_counter()
 for e,order in enumerate(orders,1):
  for batch in order.split(CONFIG['batch_size']):
   opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head(ftr[batch]),y[ix][batch]);loss.backward();opt.step()
  with torch.no_grad():val=metrics(y[iv],head(fva));train=metrics(y[ix],head(ftr))
  hist.append({'epoch':e,'train_ce':train['ce'],'val_ce':val['ce'],'val_accuracy':val['accuracy'],'val_correct':val['correct']})
  if val['correct']>best:best=val['correct'];beststate=clone_state(head);bestepoch=e;bestmetric=val
 tfit=time.perf_counter()-fitstart;head.load_state_dict(beststate);torch.save(beststate,out/f'lp_head_{name}.pt')
 return {'stage':'linear_probe','history':hist,'selected_epoch':bestepoch,'validation':bestmetric,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'feature_seconds_train':ttr,'feature_seconds_val':tva,'fit_plus_evaluation_seconds':tfit,'total_seconds':ttr+tva+tfit,'updates':CONFIG['lp_epochs']*((len(ix)+31)//32),'order_sha256':hashlib.sha256(torch.stack(orders).numpy().tobytes()).hexdigest()}

def run_ft(name,m,data,out):
 names=configure_stage(m,name,'ft');ix=data['train_idx'];iv=data['val_idx'];x=data['images'];y=data['target'];before=clone_state(m)
 bn_before={k:v.clone() for k,v in m.named_buffers()};frozen_before={k:v.clone() for k,v in m.named_parameters() if not v.requires_grad}
 orders=batch_orders(len(ix),CONFIG['ft_epochs'],offset=100);opt=torch.optim.Adam(optimizer_groups(m,name),weight_decay=0.);hist=[];best=-1;beststate=None;train_seconds=0.;eval_seconds=0.
 for e,order in enumerate(orders,1):
  m.eval();t=time.perf_counter()
  for batch in order.split(CONFIG['batch_size']):
   ids=ix[batch];opt.zero_grad(set_to_none=True);logits=m(preprocess(x[ids]));loss=F.cross_entropy(logits,y[ids]);loss.backward();opt.step()
  train_seconds+=time.perf_counter()-t;val,t=eval_model(m,x[iv],y[iv]);eval_seconds+=t;train,t=eval_model(m,x[ix],y[ix]);eval_seconds+=t
  hist.append({'epoch':e,'train_ce':train['ce'],'val_ce':val['ce'],'val_accuracy':val['accuracy'],'val_correct':val['correct']})
  if val['correct']>best:best=val['correct'];beststate=clone_state(m);bestepoch=e;bestmetric=val
  print(name,'FT',e,'validation',val['correct'],'/360',flush=True)
 assert all(torch.equal(v,dict(m.named_buffers())[k]) for k,v in bn_before.items())
 assert all(torch.equal(v,dict(m.named_parameters())[k]) for k,v in frozen_before.items())
 m.load_state_dict(beststate)
 delta={k:v for k,v in beststate.items() if k in names};torch.save(delta,out/f'ft_delta_{name}.pt')
 changed=[k for k in names if not torch.equal(before[k],beststate[k])]
 assert changed and len(changed)==len(names)
 return {'stage':'partial_fine_tuning','history':hist,'selected_epoch':bestepoch,'validation':bestmetric,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'trainable_names':names,'changed_trainable_tensors':changed,'frozen_tensors_unchanged':True,'bn_buffers_unchanged':True,'train_seconds':train_seconds,'evaluation_seconds':eval_seconds,'total_seconds':train_seconds+eval_seconds,'updates':CONFIG['ft_epochs']*((len(ix)+31)//32),'order_sha256':hashlib.sha256(torch.stack(orders).numpy().tobytes()).hexdigest()}

def stress_result(m,data):
 x=data['images'][data['val_idx']];y=data['target'][data['val_idx']];clean,_=eval_model(m,x,y);out={'clean':clean}
 for kind in CONFIG['corruptions']:
  val,seconds=eval_model(m,corrupt(x,kind),y);val['delta_accuracy_pp']=100*(val['accuracy']-clean['accuracy']);val['elapsed_seconds']=seconds;out[kind]=val
 return out

def export_comparison(results,out):
 rows=[]
 for name,r in results.items():
  for stage in ['lp','ft']:
   a=r[stage];rows.append({'model':name,'stage':stage,'selected_epoch':a['selected_epoch'],'accuracy':a['validation']['accuracy'],'macro_f1':a['validation']['macro_f1'],'trainable_parameters':a['trainable_parameters'],'total_seconds':a['total_seconds']})
 import csv
 with open(out/'confronto.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 return rows
