"""Supporto leggibile del laboratorio 08. Nessun accesso alla rete durante gli esperimenti.

Coordinate: X2=R X1+t, pixel (u,v), d=uL-uR, metri per le distanze 3D.
Il riferimento di disparità entra soltanto nella valutazione e nei controlli della demo.
"""
from pathlib import Path
import json,time,csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

SEED=8
COLORS=['#008b9a','#7954cf','#e16b3d','#278559','#cf4360']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':110,'savefig.dpi':180,'axes.titleweight':'bold'})

def load_data(folder):
    p=Path(folder); c=json.loads((p/'calibrazione.json').read_text())
    left=cv2.cvtColor(cv2.imread(str(p/'im0.png')),cv2.COLOR_BGR2RGB)
    right=cv2.cvtColor(cv2.imread(str(p/'im1.png')),cv2.COLOR_BGR2RGB)
    ref=np.load(p/'riferimento.npz')
    assert left.shape==right.shape==(c['height'],c['width'],3)
    c['K0']=np.array(c['K0']);c['K1']=np.array(c['K1'])
    return {'left':left,'right':right,'gt':ref['disp0'],'gt_right':ref['disp1'],'calib':c}

def quantiles(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return {'n':int(len(x)),'median':float(np.median(x)) if len(x) else None,
            'p90':float(np.percentile(x,90)) if len(x) else None}

def skew(t):
    x,y,z=np.ravel(t);return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])

def homogeneous(x):return np.c_[x,np.ones(len(x))]

def sampson_px(F,p0,p1):
    """Radice del residuo di Sampson: unità pixel se gli input sono pixel."""
    a=homogeneous(p0);b=homogeneous(p1);Fa=a@F.T;Ftb=b@F
    den=Fa[:,0]**2+Fa[:,1]**2+Ftb[:,0]**2+Ftb[:,1]**2
    return np.abs(np.sum(b*Fa,axis=1))/np.sqrt(np.maximum(den,1e-20))

def normalize(p,K):return cv2.undistortPoints(np.asarray(p,float).reshape(-1,1,2),K,None).reshape(-1,2)

def coverage(points,shape):
    h,w=shape[:2]
    if not len(points):return 0.
    cells=np.clip((points/np.array([w,h])*4).astype(int),0,3)
    return len(np.unique(cells[:,1]*4+cells[:,0]))/16.

def control_points(data,step=24):
    """Griglia GT fuori dalla stima: visibile a destra con consistenza GT <= 0.75 px."""
    g=data['gt'];r=data['gt_right'];h,w=g.shape
    v,u=np.mgrid[16:h-16:step,96:w-16:step];d=g[v,u]
    xr=np.rint(u-np.where(np.isfinite(d),d,0)).astype(int)
    ok=np.isfinite(d)&(xr>=0)&(xr<w)
    dr=r[v,np.clip(xr,0,w-1)]
    ok&=np.isfinite(dr)&(np.abs(np.where(np.isfinite(dr),dr,0)-np.where(np.isfinite(d),d,0))<.75)
    return np.c_[u[ok],v[ok]].astype(float),np.c_[u[ok]-d[ok],v[ok]].astype(float)

def match_images(left,right,ratio=.75,nfeatures=3500):
    sift=cv2.SIFT_create(nfeatures=nfeatures)
    k0,d0=sift.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_RGB2GRAY),None)
    k1,d1=sift.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_RGB2GRAY),None)
    if d0 is None or d1 is None:raise ValueError('Nessun descrittore: esaminare texture, esposizione e caricamento.')
    matches=cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0,d1,k=2)
    good=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<ratio*pair[1].distance]
    # Una corrispondenza per feature destra: conservare la distanza migliore.
    best={}
    for m in sorted(good,key=lambda m:m.distance):best.setdefault(m.trainIdx,m)
    good=list(best.values())
    if len(good)<12:raise ValueError(f'Solo {len(good)} match: stima robusta non utilizzabile.')
    p0=np.array([k0[m.queryIdx].pt for m in good],float)
    p1=np.array([k1[m.trainIdx].pt for m in good],float)
    return p0,p1,{'keypoints_left':len(k0),'keypoints_right':len(k1),'candidates':len(good),'ratio':ratio}

def matching_lab(data,right=None):
    cv2.setRNGSeed(SEED)
    p0,p1,metrics=match_images(data['left'],data['right'] if right is None else right)
    ids=np.random.default_rng(SEED).permutation(len(p0));n=int(.8*len(ids))
    train=ids[:n];held=ids[n:]
    F,m=cv2.findFundamentalMat(p0[train],p1[train],cv2.FM_RANSAC,1.,.999,10000)
    if F is None or F.shape!=(3,3):raise ValueError('F non stimabile: supporto o geometria insufficienti.')
    inliers=np.zeros(len(p0),bool);inliers[train]=m.ravel().astype(bool)
    c0,c1=control_points(data)
    metrics.update({'n_fit':len(train),'n_heldout':len(held),'inliers_F':int(inliers.sum()),
                    'inlier_ratio_F':float(inliers.sum()/len(train)),
                    'coverage_left':coverage(p0[inliers],data['left'].shape),
                    'coverage_right':coverage(p1[inliers],data['right'].shape),
                    'sampson_fit':quantiles(sampson_px(F,p0[inliers],p1[inliers])),
                    'sampson_heldout':quantiles(sampson_px(F,p0[held],p1[held])),
                    'sampson_control_gt':quantiles(sampson_px(F,c0,c1))})
    return {'p0':p0,'p1':p1,'F':F,'train':train,'heldout':held,'inliers':inliers,'metrics':metrics}

def warp_points(p,H):
    a=homogeneous(p)@H.T
    return a[:,:2]/a[:,2,None]

def rectification_lab(data,angle_deg=3.):
    c=data['calib'];K0,K1=c['K0'],c['K1'];h,w=data['gt'].shape
    def roll(deg):
        a=np.radians(deg);return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    A0,A1=roll(angle_deg),roll(-angle_deg);G0=K0@A0@np.linalg.inv(K0);G1=K1@A1@np.linalg.inv(K1)
    R=A1@A0.T;t=A1@np.array([-c['baseline_m'],0,0.])
    b0=cv2.warpPerspective(data['left'],G0,(w,h));b1=cv2.warpPerspective(data['right'],G1,(w,h))
    white=np.ones((h,w),np.uint8)
    v0=cv2.warpPerspective(white,G0,(w,h),flags=cv2.INTER_NEAREST)
    v1=cv2.warpPerspective(white,G1,(w,h),flags=cv2.INTER_NEAREST)
    R0r,R1r,P0,P1,Q,roi0,roi1=cv2.stereoRectify(K0,None,K1,None,(w,h),R,t,
                                      flags=cv2.CALIB_ZERO_DISPARITY,alpha=1)
    map0=cv2.initUndistortRectifyMap(K0,None,R0r,P0,(w,h),cv2.CV_32FC1)
    map1=cv2.initUndistortRectifyMap(K1,None,R1r,P1,(w,h),cv2.CV_32FC1)
    r0=cv2.remap(b0,*map0,cv2.INTER_LINEAR);r1=cv2.remap(b1,*map1,cv2.INTER_LINEAR)
    valid0=cv2.remap(v0,*map0,cv2.INTER_NEAREST).astype(bool)
    valid1=cv2.remap(v1,*map1,cv2.INTER_NEAREST).astype(bool)
    c0,c1=control_points(data);p0=warp_points(c0,G0);p1=warp_points(c1,G1)
    q0=cv2.undistortPoints(p0[:,None],K0,None,R=R0r,P=P0).reshape(-1,2)
    q1=cv2.undistortPoints(p1[:,None],K1,None,R=R1r,P=P1).reshape(-1,2)
    keep=np.ones(len(c0),bool)
    for p,mask in [(p0,v0),(p1,v1),(q0,valid0),(q1,valid1)]:
        ix=np.rint(p[:,0]).astype(int);iy=np.rint(p[:,1]).astype(int)
        keep&=(ix>=2)&(ix<w-2)&(iy>=2)&(iy<h-2)
        keep&=mask[np.clip(iy,0,h-1),np.clip(ix,0,w-1)].astype(bool)
    before=np.abs(p0[keep,1]-p1[keep,1]);after=np.abs(q0[keep,1]-q1[keep,1])
    return {'before0':b0,'before1':b1,'rect0':r0,'rect1':r1,'valid0':valid0,'valid1':valid1,
            'G0':G0,'G1':G1,'R':R,'t':t,'P0':P0,'P1':P1,'Q':Q,
            'p0':p0[keep],'p1':p1[keep],'q0':q0[keep],'q1':q1[keep],
            'before_residual':before,'after_residual':after,
            'metrics':{'n_control':int(keep.sum()),'before':quantiles(before),'after':quantiles(after),
                       'valid_area_left':float(valid0.mean()),'valid_area_right':float(valid1.mean()),
                       'angle_deg':angle_deg}}

def stereo(left,right,ndisp=80,block_size=5):
    a=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=ndisp,blockSize=block_size,
        P1=8*block_size**2,P2=32*block_size**2,disp12MaxDiff=1,uniquenessRatio=10,
        speckleWindowSize=50,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    raw=matcher.compute(a,b);d=raw.astype(np.float32)/16.
    valid=raw>-16
    d[~valid]=np.nan
    return d,valid

def depth_from_disparity(d,c,factor_f=1.):
    den=d+c['doffs'];z=np.full(d.shape,np.nan,np.float32)
    valid=np.isfinite(d)&(den>1e-9)
    z[valid]=factor_f*c['K0'][0,0]*c['baseline_m']/den[valid]
    return z

def evaluation_mask(data,vertical_shift=0):
    h,w=data['gt'].shape;v,u=np.mgrid[:h,:w];n=data['calib']['ndisp']
    return np.isfinite(data['gt'])&(data['gt']>0)&(u>=n+8)&(u<w-8)&(v>=8+abs(vertical_shift))&(v<h-8-abs(vertical_shift))

def dense_metrics(d,z,data,eval_mask):
    mask=eval_mask&np.isfinite(d)&np.isfinite(z)&(z>0)
    n=int(mask.sum());total=int(eval_mask.sum())
    err=np.abs(d[mask]-data['gt'][mask]);zgt=depth_from_disparity(data['gt'],data['calib'])
    rel=np.abs(z[mask]-zgt[mask])/zgt[mask]
    return {'n_evaluable':total,'n_valid':n,'coverage':n/total if total else None,
            'epe_px':float(np.mean(err)) if n else None,'bad2_pct':float(100*np.mean(err>2)) if n else None,
            'median_relative_depth_pct':float(100*np.median(rel)) if n else None}

def stereo_lab(data,right=None,eval_mask=None,factor_f=1.):
    d,valid=stereo(data['left'],data['right'] if right is None else right,data['calib']['ndisp'])
    z=depth_from_disparity(d,data['calib'],factor_f)
    ev=evaluation_mask(data) if eval_mask is None else eval_mask
    return {'disparity':d,'depth':z,'valid':valid,'eval_mask':ev,
            'metrics':dense_metrics(d,z,data,ev)}

def project(K,R,t,X):
    xc=X@R.T+np.ravel(t);p=xc@K.T
    with np.errstate(divide='ignore',invalid='ignore'):uv=p[:,:2]/p[:,2,None]
    return uv,xc[:,2]

def triangulate(K0,K1,p0,p1,R,t):
    a=normalize(p0,K0);b=normalize(p1,K1)
    H=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[R,np.ravel(t)],a.T,b.T).T
    X=np.full((len(H),3),np.nan);ok=np.abs(H[:,3])>1e-10
    X[ok]=H[ok,:3]/H[ok,3,None]
    return X

def sparse_lab(data,match,reproj_px=2.,min_angle_deg=1.):
    c=data['calib'];K0,K1=c['K0'],c['K1'];p0,p1=match['p0'],match['p1'];train=match['train']
    a=normalize(p0,K0);b=normalize(p1,K1);cv2.setRNGSeed(SEED)
    E,mask=cv2.findEssentialMat(a[train],b[train],np.eye(3),method=cv2.RANSAC,prob=.999,
                threshold=1./K0[0,0],maxIters=10000)
    if E is None:raise ValueError('E non stimabile.')
    if E.shape!=(3,3):raise ValueError('Più soluzioni E: geometria non sufficientemente determinata.')
    maskE=np.zeros(len(p0),bool);maskE[train]=mask.ravel().astype(bool)
    count,R,t,mask_pose=cv2.recoverPose(E,a[train],b[train],np.eye(3),mask=mask.copy())
    if count<12:raise ValueError(f'Posa non utilizzabile: {count} punti con cheiralità.')
    # t ha norma unitaria. B è un dato esterno: si scala l'intera geometria una volta sola.
    t=np.ravel(t)*c['baseline_m'];X=triangulate(K0,K1,p0,p1,R,t)
    u0,z0=project(K0,np.eye(3),np.zeros(3),X);u1,z1=project(K1,R,t,X)
    e0=np.linalg.norm(u0-p0,axis=1);e1=np.linalg.norm(u1-p1,axis=1)
    C1=-R.T@t;v0=X;v1=X-C1
    den=np.linalg.norm(v0,axis=1)*np.linalg.norm(v1,axis=1)
    with np.errstate(invalid='ignore',divide='ignore'):
        angle=np.degrees(np.arccos(np.clip(np.sum(v0*v1,axis=1)/den,-1,1)))
    finite=np.all(np.isfinite(X),axis=1);front=finite&(z0>0)&(z1>0)
    reproj=front&(e0<reproj_px)&(e1<reproj_px);good=maskE&reproj&(angle>=min_angle_deg)
    Xknown=triangulate(K0,K1,p0,p1,np.eye(3),np.array([-c['baseline_m'],0,0]))
    ix=np.rint(p0[:,0]).astype(int);iy=np.rint(p0[:,1]).astype(int)
    zref=depth_from_disparity(data['gt'],c)[iy,ix];score=good&np.isfinite(zref)&(zref>0)
    relative=np.abs(X[score,2]-zref[score])/zref[score]
    knownrel=np.abs(Xknown[score,2]-zref[score])/zref[score]
    Rerr=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    terr=np.degrees(np.arccos(np.clip(np.dot(t,np.array([-1.,0,0]))/np.linalg.norm(t),-1,1)))
    c0,c1=control_points(data);Fe=np.linalg.inv(K1).T@E@np.linalg.inv(K0)
    metrics={'candidates':len(p0),'fit':len(train),'inliers_E':int(maskE.sum()),
             'finite_E':int((maskE&finite).sum()),'front_E':int((maskE&front).sum()),
             'reprojection_E':int((maskE&reproj).sum()),'accepted':int(good.sum()),
             'cheirality_fraction':float((maskE&front).sum()/max((maskE&finite).sum(),1)),
             'reprojection_left_px':quantiles(e0[good]),'reprojection_right_px':quantiles(e1[good]),
             'angles_deg':quantiles(angle[good]),'rotation_error_deg':float(Rerr),'translation_direction_error_deg':float(terr),
             'n_depth_reference':int(score.sum()),'median_relative_depth_pct':float(100*np.median(relative)) if len(relative) else None,
             'known_pose_depth_pct_same_points':float(100*np.median(knownrel)) if len(knownrel) else None,
             'sampson_control_E_px':quantiles(sampson_px(Fe,c0,c1)),
             'baseline_m':c['baseline_m'],'reprojection_threshold_px':reproj_px,'min_angle_deg':min_angle_deg}
    return {'X':X,'X_known':Xknown,'R':R,'t':t,'C1':C1,'E':E,'maskE':maskE,'good':good,'front':front,
            'angle':angle,'e0':e0,'e1':e1,'zref':zref,'metrics':metrics}

def challenge(data,baseline,case='A'):
    """Una sola causa. B usa una maschera comune ridotta per lo shift verticale."""
    if case=='A':
        right=cv2.GaussianBlur(data['right'],(0,0),2.)
        result=stereo_lab(data,right);base=baseline;setting='blur destra, sigma=2 px'
    elif case=='B':
        shift=4;h,w=data['gt'].shape
        right=cv2.warpAffine(data['right'],np.float32([[1,0,0],[0,1,shift]]),(w,h))
        ev=evaluation_mask(data,shift);result=stereo_lab(data,right,ev)
        base={**baseline,'eval_mask':ev,'metrics':dense_metrics(baseline['disparity'],baseline['depth'],data,ev)}
        setting='shift verticale destra +4 px, modello invariato'
    elif case=='C':
        right=data['right'];base=baseline;z=depth_from_disparity(baseline['disparity'],data['calib'],1.1)
        result={**baseline,'depth':z,'metrics':dense_metrics(baseline['disparity'],z,data,baseline['eval_mask'])}
        setting='f moltiplicata per 1.10 nella sola conversione d-Z'
    else:raise ValueError('Scegliere A, B oppure C.')
    common=base['eval_mask']&result['eval_mask']&np.isfinite(base['disparity'])&np.isfinite(result['disparity'])
    metrics={'case':case,'setting':setting,'baseline':base['metrics'],'modified':result['metrics'],
             'common_baseline':dense_metrics(base['disparity'],base['depth'],data,common),
             'common_modified':dense_metrics(result['disparity'],result['depth'],data,common)}
    if case=='A':
        metrics['matching_modified']=matching_lab(data,right)['metrics']
    if case=='B':metrics['vertical_error_control_px']=4.
    return {'baseline':base,'modified':result,'right':right,'metrics':metrics}

def show_metrics(metrics):
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

def save_figure(fig,path):
    fig.savefig(path,bbox_inches='tight',facecolor='white')
    return fig

def plot_pair(data):
    fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for a,img,name in zip(ax,[data['left'],data['right']],['Vista sinistra','Vista destra']):a.imshow(img);a.set_title(name);a.axis('off')
    return fig

def plot_matches(data,m,grid=False):
    h,w=data['gt'].shape;fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
    ax.imshow(np.concatenate([data['left'],data['right']],axis=1))
    ids=np.random.default_rng(SEED).choice(len(m['p0']),min(85,len(m['p0'])),replace=False)
    for i in ids:
        p,q=m['p0'][i],m['p1'][i]+[w,0]
        col=COLORS[0] if m['inliers'][i] else COLORS[4]
        ax.plot([p[0],q[0]],[p[1],q[1]],color=col,lw=.65,alpha=.75)
        ax.scatter([p[0],q[0]],[p[1],q[1]],color=col,s=7)
    if grid:
        for x in np.linspace(0,2*w,9):ax.axvline(x,color='white',lw=.8,alpha=.8)
        for y in np.linspace(0,h,5):ax.axhline(y,color='white',lw=.8,alpha=.8)
    ax.set_title('Campione di match: verde = inlier F del fit, rosso = altri candidati')
    ax.axis('off');return fig

def plot_residuals(data,m):
    fig,ax=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    sets=[('Fit inlier',sampson_px(m['F'],m['p0'][m['inliers']],m['p1'][m['inliers']])),
          ('Candidati esclusi dal fit',sampson_px(m['F'],m['p0'][m['heldout']],m['p1'][m['heldout']]))]
    for (label,values),col in zip(sets,COLORS):
        x=np.sort(values);ax[0].plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=col)
    ax[0].set(xlabel='Radice Sampson [px]',ylabel='Frazione cumulativa',xlim=(0,3));ax[0].legend(fontsize=9)
    c0,c1=control_points(data);r=sampson_px(m['F'],c0,c1)
    ax[1].imshow(data['left']);sct=ax[1].scatter(c0[:,0],c0[:,1],c=r,s=18,cmap='magma',vmin=0,vmax=max(.5,np.percentile(r,95)))
    ax[1].set_title('Controllo su punti GT fuori dal fit');ax[1].axis('off');fig.colorbar(sct,ax=ax[1],label='px',shrink=.8)
    return fig

def plot_rectification(r):
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for a,img,title in zip(ax.flat,[r['before0'],r['before1'],r['rect0'],r['rect1']],
                 ['B: sinistra con roll +3°','B: destra con roll −3°','Rettificata sinistra','Rettificata destra']):
        a.imshow(img)
        for y in np.arange(60,img.shape[0],65):a.axhline(y,color='#00e7eb',lw=.75,alpha=.9)
        a.set_title(title);a.axis('off')
    return fig

def plot_rectification_masks(r):
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    ax[0].imshow(r['valid0'],cmap='gray',vmin=0,vmax=1);ax[0].set_title('Area valida, sinistra');ax[0].axis('off')
    ax[1].imshow(r['valid1'],cmap='gray',vmin=0,vmax=1);ax[1].set_title('Area valida, destra');ax[1].axis('off')
    ax[2].hist(r['before_residual'],bins=20,color=COLORS[0],alpha=.8)
    ax[2].set(xlabel='Residuo verticale prima [px]',ylabel='Punti',title='Stessi punti prima e dopo')
    ax[2].text(.98,.95,f"Dopo: p90 {r['metrics']['after']['p90']:.1e} px",ha='right',va='top',transform=ax[2].transAxes,fontsize=10)
    return fig

def plot_dense(data,s):
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    cm=plt.get_cmap('turbo').copy();cm.set_bad('#e2e5e9')
    images=[s['disparity'],s['depth'],s['valid'],np.abs(s['disparity']-data['gt'])]
    titles=['Disparità stimata [px]','Profondità stimata [m]','Predizioni valide (bianco)','Errore assoluto di disparità [px]']
    for i,(a,img,title) in enumerate(zip(ax.flat,images,titles)):
        lim=[(0,80),(1.4,3.2),(0,1),(0,5)][i]
        m=a.imshow(img,cmap='gray' if i==2 else cm,vmin=lim[0],vmax=lim[1]);a.set_title(title);a.axis('off')
        if i!=2:fig.colorbar(m,ax=a,shrink=.72)
    return fig

def plot_reference(data,s):
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    cm=plt.get_cmap('turbo').copy();cm.set_bad('#e2e5e9')
    for a,img,title in zip(ax,[data['gt'],s['disparity'],s['eval_mask']],['Disparità di riferimento','StereoSGBM','Regione di valutazione']):
        im=a.imshow(img,cmap=cm if img.dtype!=bool else 'gray',vmin=0,vmax=80 if img.dtype!=bool else 1)
        a.set_title(title);a.axis('off')
    return fig

def region_windows(data):
    # Posizioni fisse alla risoluzione didattica. La variabilità locale è riportata, non presunta.
    return [(260,280,'Dettaglio'),(585,85,'Area poco strutturata'),(450,300,'Bordo')]

def scanline_costs(data):
    a=cv2.cvtColor(data['left'],cv2.COLOR_RGB2GRAY).astype(float)
    b=cv2.cvtColor(data['right'],cv2.COLOR_RGB2GRAY).astype(float)
    curves=[];radius=4
    for u,v,label in region_windows(data):
        window=a[v-radius:v+radius+1,u-radius:u+radius+1];cost=[]
        for d in range(data['calib']['ndisp']):cost.append(np.mean(np.abs(window-b[v-radius:v+radius+1,u-d-radius:u-d+radius+1])))
        curves.append({'u':u,'v':v,'label':label,'std':float(window.std()),'cost':cost,'gt':float(data['gt'][v,u])})
    return curves

def plot_scanline(data):
    from matplotlib.patches import Rectangle

    curves=scanline_costs(data)
    fig,ax=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    # The photograph supplies context; the outlined 9x9 support supplies the SAD.
    radius=4
    context_radius=25
    roi_color='#ffd43b'
    max_cost=max(float(np.max(c['cost'])) for c in curves)
    ymax=10*np.ceil(max_cost/10)
    for j,c in enumerate(curves):
        u,v=c['u'],c['v']
        letter=chr(ord('A')+j)
        context=data['left'][v-context_radius:v+context_radius+1,
                             u-context_radius:u+context_radius+1]
        ax[0,j].imshow(context,interpolation='nearest')
        ax[0,j].add_patch(Rectangle(
            (context_radius-radius-.5,context_radius-radius-.5),
            2*radius+1,2*radius+1,fill=False,edgecolor=roi_color,linewidth=2.2))
        ax[0,j].plot(context_radius,context_radius,'+',color=roi_color,
                     markersize=6,markeredgewidth=1.2)
        ax[0,j].text(context_radius,context_radius-radius-2,'9×9',
                     ha='center',va='bottom',fontsize=9,color='black',
                     bbox=dict(facecolor=roi_color,edgecolor='none',pad=1.2))
        ax[0,j].axis('off')
        ax[0,j].set_title(
            f"{letter} · Contesto 51×51: {c['label'].lower()}\n(u={u}, v={v})",
            fontsize=10.5)

        disparities=np.arange(len(c['cost']))
        best=int(np.argmin(c['cost']))
        ax[1,j].plot(disparities,c['cost'],color=COLORS[j],linewidth=1.6)
        ax[1,j].axvline(c['gt'],ls='--',color='black',lw=1,label='Riferimento GT')
        ax[1,j].scatter(best,c['cost'][best],s=45,color=COLORS[j],
                        edgecolor='black',linewidth=.7,zorder=3,
                        label='Minimo SAD',clip_on=False)
        std_label=f"{c['std']:.2f}"
        ax[1,j].set(xlabel='Disparità candidata d [px]',
                    ylabel='SAD medio [livelli di grigio]',
                    xlim=(0,len(disparities)-1),ylim=(0,ymax),
                    xticks=[0,20,40,60,len(disparities)-1])
        ax[1,j].set_title(f"{letter} · Finestra 9×9: dev. standard {std_label}",
                          fontsize=10.5)
        ax[1,j].legend(fontsize=8.5,loc='upper left')
    return fig

def equal_axes(ax,X):
    low=np.nanpercentile(X,1,axis=0);high=np.nanpercentile(X,99,axis=0)
    low=np.minimum(low,[0,0,0]);high=np.maximum(high,[.2,0,0])
    center=(low+high)/2;span=(high-low).max()/2
    ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span)
    ax.set_box_aspect([1,1,1]);ax.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]');ax.view_init(elev=-70,azim=-90)

def plot_sparse(data,m,s):
    fig=plt.figure(figsize=(12,5.8),layout='constrained')
    colors=data['left'][np.rint(m['p0'][:,1]).astype(int),np.rint(m['p0'][:,0]).astype(int)]/255.
    visible=s['front']&(s['X'][:,2]<8)&(np.abs(s['X'][:,0])<8)&(np.abs(s['X'][:,1])<8)
    for j,mask in enumerate([visible,s['good']]):
        ax=fig.add_subplot(1,2,j+1,projection='3d');X=s['X'][mask]
        ax.scatter(*X.T,c=colors[mask],s=4,depthshade=False)
        ax.scatter([0,s['C1'][0]],[0,s['C1'][1]],[0,s['C1'][2]],c=[COLORS[0],COLORS[1]],s=45,marker='^')
        equal_axes(ax,s['X'][s['good']]);ax.set_title('Candidati visibili, prima dei filtri' if j==0 else f"Punti accettati: {s['metrics']['accepted']}")
    return fig

def plot_sparse_quality(s):
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    mask=s['maskE']&s['front'];a=s['angle'][mask];e=np.maximum(s['e0'][mask],s['e1'][mask])
    ax[0].scatter(a,e,s=9,c=COLORS[0],alpha=.7);ax[0].axhline(2,color=COLORS[4],ls='--');ax[0].axvline(1,color=COLORS[4],ls='--')
    ax[0].set(xlabel='Angolo [gradi]',ylabel='Riproiezione max [px]',ylim=(0,min(6,max(2.5,np.nanpercentile(e,95)))))
    ax[1].hist(s['angle'][s['good']],bins=20,color=COLORS[1]);ax[1].set(xlabel='Angolo accettato [gradi]',ylabel='Punti')
    ok=s['good']&np.isfinite(s['zref']);ax[2].scatter(s['zref'][ok],s['X'][ok,2],s=8,alpha=.5,color=COLORS[0],label='Posa stimata')
    ax[2].scatter(s['zref'][ok],s['X_known'][ok,2],s=8,alpha=.35,color=COLORS[2],label='Posa nota')
    lo,hi=np.percentile(s['zref'][ok],[1,99]);ax[2].plot([lo,hi],[lo,hi],c='black',lw=1)
    ax[2].set(xlabel='Z riferimento [m]',ylabel='Z triangolata [m]');ax[2].legend(fontsize=8)
    return fig

def plot_challenge(data,ch):
    fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained');case=ch['metrics']['case']
    ax[0].imshow(ch['right']);ax[0].set_title('Vista destra usata');ax[0].axis('off')
    cm=plt.get_cmap('turbo').copy();cm.set_bad('#e2e5e9')
    key='depth' if case=='C' else 'disparity';lim=(1.4,3.4) if case=='C' else (0,80)
    for a,k,title in zip(ax[1:],['baseline','modified'],['Baseline','Dopo la modifica']):
        im=a.imshow(ch[k][key],cmap=cm,vmin=lim[0],vmax=lim[1]);a.set_title(title);a.axis('off')
    fig.colorbar(im,ax=ax[1:],shrink=.7,label='m' if case=='C' else 'px')
    return fig

def plot_uncertainty():
    z=np.linspace(.5,6,100);sigma=z*z*.5/(800*.12)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.plot(z,sigma*100,color=COLORS[0],lw=2.5)
    ax.scatter([2,4],[4*.5/96*100,16*.5/96*100],color=COLORS[2],s=50)
    ax.set(xlabel='Distanza Z [m]',ylabel='Incertezza σZ [cm]',title='Esempio calcolato: f=800 px, B=0.12 m, σd=0.5 px')
    ax.grid(alpha=.2);return fig

def export_ply(path,data,m,s):
    mask=s['good'];X=s['X'][mask];p=np.rint(m['p0'][mask]).astype(int);rgb=data['left'][p[:,1],p[:,0]]
    with open(path,'w') as f:
        f.write('ply\nformat ascii 1.0\ncomment units meters; frame camera left; scale fixed by known baseline\n')
        f.write(f'element vertex {len(X)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for x,col in zip(X,rgb):f.write(' '.join([*(f'{v:.7f}' for v in x),*(str(int(v)) for v in col)])+'\n')
