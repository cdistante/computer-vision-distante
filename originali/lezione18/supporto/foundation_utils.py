from pathlib import Path
import hashlib,json,time,os,urllib.request,zipfile,subprocess,sys
import numpy as np
import torch
from PIL import Image
NAMES=['PennPed00014','PennPed00022','PennPed00035','FudanPed00042','FudanPed00058','FudanPed00063']
SAM_COMMIT='2b90b9f5ceec907a1c18123530e92e794ad901a4'
SAM_URL='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
LR_URL='https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fetch(url,path,expected=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():urllib.request.urlretrieve(url,path)
 if expected and sha(path)!=expected:raise ValueError('Hash diverso per '+path.name)
 return path

def bbox(g):
 y,x=np.where(g)
 if len(x)==0:raise ValueError('Maschera vuota')
 return np.array([x.min(),y.min(),x.max()+1,y.max()+1],np.float32)
def fixed_sample(data,name):
 im=Image.open(Path(data)/'PNGImages'/f'{name}.png').convert('RGB');labels=np.array(Image.open(Path(data)/'PedMasks'/f'{name}_mask.png'));ids=np.unique(labels);ids=ids[ids>0];areas=[int((labels==k).sum())for k in ids];target=int(ids[np.argmax(areas)]);gt=labels==target;b=bbox(gt);yx=np.argwhere(gt);pos=yx[np.argmin(((yx-yx.mean(0))**2).sum(1))][::-1].astype(float);h,w=gt.shape;bw=b[2]-b[0];bh=b[3]-b[1];loose=np.array([max(0,b[0]-.25*bw),max(0,b[1]-.25*bh),min(w,b[2]+.25*bw),min(h,b[3]+.25*bh)],np.float32)
 return {'name':name,'image':im,'rgb':np.array(im),'labels':labels,'ids':ids,'target_id':target,'gt':gt,'box':b,'point':pos,'loose_box':loose}
def sync(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize()
def timed(fn,device):
 sync(device);st=time.perf_counter();value=fn();sync(device);return value,time.perf_counter()-st

def load_sam(deps,device):
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 # No connected-components CUDA extension or hidden hole filling in this protocol.
 model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(Path(deps)/'sam2.1_hiera_tiny.pt'),device=str(device),apply_postprocessing=False)
 return SAM2ImagePredictor(model,max_hole_area=0,max_sprinkle_area=0)
def run_sam_sample(predictor,sample,device):
 arrays={'gt':sample['gt'],'labels':sample['labels']};row={'name':sample['name'],'shape':list(sample['rgb'].shape),'target_id':sample['target_id'],'instances':len(sample['ids']),'box':sample['box'].tolist(),'point':sample['point'].tolist(),'loose_box':sample['loose_box'].tolist(),'prompts':{}}
 with torch.inference_mode():
  _,row['sam_encoder_seconds']=timed(lambda:predictor.set_image(sample['rgb']),device)
  for mode,kw in [('point',dict(point_coords=sample['point'][None],point_labels=np.array([1]),multimask_output=True)),('box',dict(box=sample['box'],multimask_output=False)),('loose_box',dict(box=sample['loose_box'],multimask_output=False))]:
   (ms,sc,lg),sec=timed(lambda:predictor.predict(**kw),device);j=int(np.argmax(sc));arrays[mode]=ms[j].astype(bool);row['prompts'][mode]={'score':float(sc[j]),'decoder_seconds':sec,'candidate_scores':sc.tolist()}
   if mode=='point':arrays['point_candidates']=ms.astype(bool)
  fp=np.argwhere(arrays['loose_box']&~sample['gt'])
  # Controlled oracle correction: candidate error centroid-nearest false positive.
  if len(fp):neg=fp[np.argmin(((fp-fp.mean(0))**2).sum(1))][::-1].astype(float);negative_available=True
  else:
   bg=np.argwhere(~sample['gt']);neg=bg[0][::-1].astype(float);negative_available=False
  (ms,sc,lg),sec=timed(lambda:predictor.predict(box=sample['loose_box'],point_coords=np.stack([sample['point'],neg]),point_labels=np.array([1,0]),multimask_output=False),device)
  arrays['corrected']=ms[0].astype(bool);row['negative_point']=neg.tolist();row['negative_from_false_positive']=negative_available;row['prompts']['corrected']={'score':float(sc[0]),'decoder_seconds':sec}
  all_masks=[];times=[]
  for k in sample['ids']:
   (ms,sc,lg),sec=timed(lambda:predictor.predict(box=bbox(sample['labels']==k),multimask_output=False),device);all_masks.append(ms[0].astype(bool));times.append(sec)
  arrays['sam_instances']=np.stack(all_masks);row['sam_union_decoder_seconds']=sum(times)
 return row,arrays

def load_lraspp(deps,device):
 from torchvision.models.segmentation import lraspp_mobilenet_v3_large,LRASPP_MobileNet_V3_Large_Weights
 model=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None);model.load_state_dict(torch.load(Path(deps)/'lraspp_mobilenet_v3_large-d234d4ea.pth',map_location='cpu',weights_only=True));model.eval().to(device)
 return model,LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
def run_lraspp(model,transform,sample,device):
 def infer():
  with torch.inference_mode():
   x=transform(sample['image']).unsqueeze(0).to(device);logits=model(x)['out'];logits=torch.nn.functional.interpolate(logits,size=sample['gt'].shape,mode='bilinear',align_corners=False);return (logits.argmax(1)[0]==15).cpu().numpy()
 return timed(infer,device)

def show_masks(sample,masks,titles,points=None,boxes=None,points_by_panel=None,boxes_by_panel=None):
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,len(masks),figsize=(6*len(masks),5),squeeze=False)
 for j,(mask,title)in enumerate(zip(masks,titles)):
  ax=axs[0,j];ax.imshow(sample['rgb']);overlay=np.zeros((*mask.shape,4));overlay[mask]=[.03,.65,.72,.5];ax.imshow(overlay);ax.contour(sample['gt'],levels=[.5],colors=['#ffdc40'],linewidths=1.2);ax.set_title(title,fontsize=15);ax.axis('off')
  current_points=points_by_panel[j] if points_by_panel is not None else points
  current_boxes=boxes_by_panel[j] if boxes_by_panel is not None else boxes
  if current_points is not None:
   for pt,label in current_points:ax.scatter(*pt,c='#31e074'if label else'#ec3344',s=75,marker='+'if label else'x')
  if current_boxes is not None:
   from matplotlib.patches import Rectangle
   for b in current_boxes:ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#f03d64',linewidth=1.5))
 fig.tight_layout();return fig
