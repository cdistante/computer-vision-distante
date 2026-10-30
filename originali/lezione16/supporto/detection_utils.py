"""Computer Vision, L16. Protocollo didattico Penn train / Fudan holdout."""
from pathlib import Path
import hashlib,json,time,random,platform,zipfile,urllib.request,io,contextlib,copy
import numpy as np
import torch,torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn,FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
SEED=20261112
DATA_URL='https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
WEIGHTS=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
CONFIG={'seed':SEED,'epochs':3,'batch_size':2,'learning_rate':.005,'momentum':.9,'weight_decay':.0005,'train_source':'Penn','holdout_source':'Fudan','selection':'fixed_epoch3_no_holdout_tuning','augmentation':None,'score_floor':.001,'nms_iou':.5,'max_detections':100,'rpn_eval_pre_post_top_n':150,'min_size':320,'max_size':640,'trainable':'roi_heads.box_predictor','area':'original bounding-box area','coco_iou_thresholds':[round(.5+.05*i,2) for i in range(10)]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(device='auto'):
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(1)
 if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
 if device not in ('cpu','cuda'):raise ValueError('Usare cpu, cuda oppure auto')
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA non disponibile: scegliere CPU o runtime GPU')
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if device=='cuda':torch.cuda.manual_seed_all(SEED)
 # Alcuni kernel ROIAlign/CUDA hanno backward non deterministico. Non promettiamo identità CPU/GPU.
 return {'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'python':platform.python_version(),'device':device,'processor':platform.processor(),'platform':platform.platform(),'threads':torch.get_num_threads(),'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None}
def sync(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize()
def prepare(base,download=True):
 base=Path(base);(base/'dati').mkdir(exist_ok=True);(base/'pesi/checkpoints').mkdir(parents=True,exist_ok=True)
 z=base/'dati/PennFudanPed.zip';w=base/'pesi/checkpoints'/WEIGHTS.url.rsplit('/',1)[-1]
 for path,url in [(z,DATA_URL),(w,WEIGHTS.url)]:
  if not path.exists():
   if not download:raise FileNotFoundError(f'Manca {path}; eseguire la preparazione online prima della lezione.')
   urllib.request.urlretrieve(url,path)
 d=base/'dati/PennFudanPed'
 if not (d/'PNGImages').exists():
  with zipfile.ZipFile(z) as a:
   for name in a.namelist():
    dest=(base/'dati'/name).resolve()
    if not dest.is_relative_to((base/'dati').resolve()):raise ValueError('Percorso ZIP non sicuro')
   a.extractall(base/'dati')
 torch.hub.set_dir(str(base/'pesi'))
 return {'data_archive_sha256':sha(z),'weights_sha256':sha(w),'data_url':DATA_URL,'weights_url':WEIGHTS.url}
def mask_to_boxes(mask):
 raise NotImplementedError("Completare C07 del notebook")

class Pedestrians(torch.utils.data.Dataset):
 def __init__(self,base,source):
  self.root=Path(base)/'dati/PennFudanPed';self.files=sorted((self.root/'PNGImages').glob(source+'*.png'));self.source=source
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  f=self.files[i];im=np.array(Image.open(f).convert('RGB'));mask=np.array(Image.open(self.root/'PedMasks'/(f.stem+'_mask.png')))
  boxes=mask_to_boxes(mask);area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]);h,w=im.shape[:2]
  target={'boxes':boxes,'labels':torch.ones(len(boxes),dtype=torch.int64),'image_id':torch.tensor(i+1),'area':area,'iscrowd':torch.zeros(len(boxes),dtype=torch.int64)}
  assert len(boxes)>0 and torch.all(area>0) and torch.all(boxes[:,[0,2]]<=w) and torch.all(boxes[:,[1,3]]<=h)
  return torch.from_numpy(im).permute(2,0,1).float()/255,target
 def summary(self):
  areas=np.concatenate([self[i][1]['area'].numpy() for i in range(len(self))]);return {'images':len(self),'instances':len(areas),'small':int(sum((areas>=0)&(areas<=1024))),'medium':int(sum((areas>=1024)&(areas<=9216))),'large':int(sum((areas>=9216)&(areas<=1e10))),'boundary_areas':int(sum((areas==1024)|(areas==9216))),'min_area':float(areas.min()),'max_area':float(areas.max()),'files':[p.name for p in self.files]}
def split_audit(train,holdout):
 h=lambda f:hashlib.sha256(np.array(Image.open(f).convert('RGB')).tobytes()).hexdigest()
 a={h(p) for p in train.files};b={h(p) for p in holdout.files};assert not a&b
 # dHash è uno screening, non una prova universale dell'assenza di duplicati.
 def dh(f):
  v=np.array(Image.open(f).convert('L').resize((9,8)));return (v[:,1:]>v[:,:-1]).ravel()
 aa=[dh(f) for f in train.files];bb=[dh(f) for f in holdout.files];pairs=[]
 for i,x in enumerate(aa):
  for j,y in enumerate(bb):pairs.append((int(sum(x!=y)),train.files[i].name,holdout.files[j].name))
 pairs.sort();return {'split_rule':'Penn vs Fudan, distinct acquisition sites','cross_exact_pixel_duplicates':len(a&b),'minimum_cross_dhash_distance':pairs[0][0],'nearest_pairs':pairs[:5],'dhash_caution':'screening only; distinct acquisition source is the grouping criterion'}
def make_model(base,device='cpu'):
 torch.hub.set_dir(str(Path(base)/'pesi'));torch.manual_seed(SEED)
 m=fasterrcnn_mobilenet_v3_large_320_fpn(weights=WEIGHTS,progress=False,trainable_backbone_layers=0,box_score_thresh=CONFIG['score_floor'],box_nms_thresh=.5,box_detections_per_img=100)
 m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,2)
 configure_trainable(m);return m.to(device)
def configure_trainable(model):
 raise NotImplementedError("Completare C12 del notebook")

def training_mode(m):
 m.train();m.backbone.eval();m.rpn.eval();m.roi_heads.box_head.eval()
 # RPN congelato in eval: usa 150 proposte, non calcola loss RPN; ROI train aggiunge GT e campiona.
 return m
@torch.no_grad()
def predict(m,ds,device):
 m.eval();preds=[];sync(device);start=time.perf_counter()
 for i in range(len(ds)):
  image,_=ds[i];p=m([image.to(device)])[0];preds.append({k:v.cpu() for k,v in p.items()})
 sync(device);return preds,time.perf_counter()-start

def xyxy_to_xywh(box):
 raise NotImplementedError("Completare C19 del notebook")

def coco_eval(ds,preds):
 gt=COCO();images=[];anns=[];detections=[];aid=1
 for i,p in enumerate(preds):
  image,t=ds[i];h,w=image.shape[-2:];images.append({'id':i+1,'height':h,'width':w,'file_name':ds.files[i].name})
  for b,area in zip(t['boxes'].tolist(),t['area'].tolist()):
   x1,y1,x2,y2=b;anns.append({'id':aid,'image_id':i+1,'category_id':1,'bbox':xyxy_to_xywh([x1,y1,x2,y2]),'area':area,'iscrowd':0});aid+=1
  for b,score,label in zip(p['boxes'].tolist(),p['scores'].tolist(),p['labels'].tolist()):
   if label!=1:continue
   x1,y1,x2,y2=b;detections.append({'image_id':i+1,'category_id':1,'bbox':xyxy_to_xywh([x1,y1,x2,y2]),'score':score})
 gt.dataset={'images':images,'annotations':anns,'categories':[{'id':1,'name':'person'}],'info':{'description':'PennFudan bbox from masks half-open'}}
 with contextlib.redirect_stdout(io.StringIO()):
  gt.createIndex()
  if detections:dt=gt.loadRes(detections)
  else:dt=COCO();dt.dataset={**gt.dataset,'annotations':[]};dt.createIndex()
  e=COCOeval(gt,dt,'bbox');e.params.imgIds=list(range(1,len(ds)+1));e.params.catIds=[1];e.evaluate();e.accumulate();e.summarize()
 def clean(x):return None if float(x)<0 else float(x)
 names=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
 stats={k:clean(v) for k,v in zip(names,e.stats)};stats['AP_person']=stats['AP'];stats['support']=ds.summary();stats['detections']=len(detections)
 stats['precision_50_recall101']=e.eval['precision'][0,:,0,0,2].tolist();stats['recall_grid']=e.params.recThrs.tolist()
 stats['evaluation']={'implementation':'pycocotools.COCOeval','iouType':'bbox','iouThrs':e.params.iouThrs.tolist(),'maxDets':e.params.maxDets,'areaRng':e.params.areaRng,'areaRngLbl':e.params.areaRngLbl,'recThrs_n':len(e.params.recThrs)}
 return stats

def train_fixed(m,ds,device,out):
 out=Path(out);out.mkdir(exist_ok=True);frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad};buffers={n:p.detach().cpu().clone() for n,p in m.named_buffers()}
 before={n:p.detach().cpu().clone() for n,p in m.roi_heads.box_predictor.named_parameters()}
 opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
 histories=[];orders=[];sync(device);start=time.perf_counter()
 for ep in range(3):
  training_mode(m);order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(SEED+ep)).tolist();orders.append(order);hist=[]
  for b in range(0,len(order),2):
   items=[ds[i] for i in order[b:b+2]];ims=[x.to(device) for x,t in items];targets=[{k:v.to(device) for k,v in t.items()} for x,t in items]
   opt.zero_grad(set_to_none=True);losses=m(ims,targets);loss=sum(losses.values());assert torch.isfinite(loss);loss.backward();opt.step();hist.append({k:float(v.detach()) for k,v in losses.items()})
  means={k:float(np.mean([r[k] for r in hist])) for k in hist[0]};histories.append({'epoch':ep+1,'updates':len(hist),**means});print('Epoca',ep+1,means,flush=True)
 sync(device);elapsed=time.perf_counter()-start;m.eval()
 unchanged=all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in m.named_parameters() if n in frozen);buffer_ok=all(torch.equal(p.detach().cpu(),buffers[n]) for n,p in m.named_buffers());changed=[n for n,p in m.roi_heads.box_predictor.named_parameters() if not torch.equal(p.detach().cpu(),before[n])]
 assert unchanged and buffer_ok and len(changed)==4
 torch.save({k:v.cpu() for k,v in m.roi_heads.box_predictor.state_dict().items()},out/'predictor_epoch3.pt')
 return {'history':histories,'seconds':elapsed,'updates':sum(h['updates'] for h in histories),'frozen_parameters_unchanged':unchanged,'buffers_unchanged':buffer_ok,'predictor_tensors_changed':changed,'order_sha256':hashlib.sha256(json.dumps(orders).encode()).hexdigest(),'selected_epoch':3,'selection':'fixed before observing holdout'}

def match_at_threshold(ds,preds,score=.5,iou=.5):
 from torchvision.ops import box_iou
 tp=fp=fn=0;by_image=[]
 for i,p in enumerate(preds):
  _,t=ds[i];b=p['boxes'][p['scores']>=score];scores=p['scores'][p['scores']>=score];order=torch.argsort(scores,descending=True,stable=True);g=t['boxes'];used=set();hits=0
  ious=box_iou(b[order],g)
  for row in ious:
   cand=[(float(v),j) for j,v in enumerate(row) if j not in used and float(v)>=iou]
   if cand:_,j=max(cand,key=lambda x:(x[0],-x[1]));used.add(j);hits+=1
  f=len(b)-hits;miss=len(g)-hits;tp+=hits;fp+=f;fn+=miss;by_image.append({'file':ds.files[i].name,'TP':hits,'FP':f,'FN':miss})
 return {'score':score,'IoU':iou,'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'per_image':by_image}

def report_latency(m,ds,device):
 x=ds[0][0].to(device);m.eval();times=[]
 with torch.no_grad():
  for _ in range(3):m([x])
  for _ in range(7):
   sync(device);t=time.perf_counter();m([x]);sync(device);times.append(1000*(time.perf_counter()-t))
 return {'median_ms':float(np.median(times)),'q25_ms':float(np.quantile(times,.25)),'q75_ms':float(np.quantile(times,.75)),'batch':1,'input_file':ds.files[0].name,'includes':'internal resize+normalization+detector+NMS+postprocess','excludes':'disk decode, transfer to device, model loading','warmup':3,'repetitions':7}
