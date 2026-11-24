"""Track identity and select the best SAM/fused-boundary reference frame causally."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .fusion import mask_quad,public

class Tracker:
    def __init__(self,cfg):
        self.cfg=cfg;self.tracks={};self.next_id=0;self.previous_gray=None;self.previous_time=None
        self.locked=None;self.pending=None;self.streak=0
    def update(self,image,candidates,frame,timestamp):
        # Keep the selected lost identity, but release other expired image masks.
        self.tracks={tid:t for tid,t in self.tracks.items() if t['state']!='EXPIRED' or tid==self.locked}
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);h,w=gray.shape;active=[]
        for tid,t in self.tracks.items():
            if t['state']=='EXPIRED':continue
            t['current']=None
            if self.previous_gray is not None:
                region=mask_quad(t['quad'],gray.shape);region[int(h*self.cfg['suction_edge_fraction']):]=False
                p=cv2.goodFeaturesToTrack(self.previous_gray,120,.01,4,mask=region.astype('uint8')*255)
                reliable=False
                if p is not None and len(p)>=8:
                    q,st,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,p,None,winSize=(21,21),maxLevel=3)
                    if q is not None:
                        back,sb,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,q,None,winSize=(21,21),maxLevel=3)
                        good=(st[:,0]>0)&(sb[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1.5)
                        if good.sum()>=8:
                            H,inliers=cv2.findHomography(p[good],q[good],cv2.RANSAC,2)
                            if H is not None and inliers.sum()>=8 and inliers.mean()>=.5:
                                new=cv2.perspectiveTransform(t['quad'].astype('float32')[None],H)[0]
                                ratio=abs(cv2.contourArea(new))/max(1,abs(cv2.contourArea(t['quad'].astype('float32'))))
                                if np.isfinite(new).all() and .4<ratio<2.5 and np.max(np.linalg.norm(new-t['quad'],axis=1))<.35*max(h,w):
                                    t['quad']=new;reliable=True
                t['state']='VISUAL_PREDICTION' if reliable else 'HELD_PREDICTION'
            if timestamp-t['last_observed_time']>self.cfg['max_prediction_seconds']:t['state']='EXPIRED'
            else:active.append(tid)
        assigned={}
        if active and candidates:
            costs=np.array([[1-float((mask_quad(self.tracks[t]['quad'],gray.shape)&c['mask']).sum())/max(1,(mask_quad(self.tracks[t]['quad'],gray.shape)|c['mask']).sum()) for c in candidates] for t in active])
            aa,bb=linear_sum_assignment(costs)
            for a,b in zip(aa,bb):
                if costs[a,b]<.8:assigned[b]=active[a]
        for i,c in enumerate(candidates):
            tid=assigned.get(i)
            if tid is None:
                tid=self.next_id;self.next_id+=1
                self.tracks[tid]=dict(best=None,state='OBSERVED')
            t=self.tracks[tid];t.update(quad=np.asarray(c['quad_px']),current=c,state='OBSERVED',last_observed_time=timestamp)
            if t['best'] is None or c['boundary_error']<t['best']['boundary_error']:
                t['best']=dict(source_frame=frame,timestamp_s=timestamp,**public(c))
        # Candidate selection is separate from best reference-frame selection.
        current=[(tid,t) for tid,t in self.tracks.items() if t['state']=='OBSERVED']
        roi=np.zeros((h,w),bool);x0,y0,x1,y1=self.cfg['selection_roi_fraction']
        roi[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=True
        ranks=sorted([(float((t['current']['mask']&roi).sum()/max(1,roi.sum())),tid) for tid,t in current],reverse=True)
        if self.locked is None:
            unique=bool(ranks and ranks[0][0]>=self.cfg['min_roi_coverage'] and (len(ranks)==1 or ranks[1][0]<ranks[0][0]*self.cfg['ambiguity_ratio']))
            chosen=ranks[0][1] if unique else None
            self.streak=self.streak+1 if chosen is not None and chosen==self.pending else int(chosen is not None)
            self.pending=chosen
            if self.streak>=self.cfg['confirmation_observations']:self.locked=chosen
        self.previous_gray=gray;self.previous_time=timestamp
        shown=sorted(self.tracks,key=lambda tid:(tid==self.locked,self.tracks[tid]['state']=='OBSERVED',self.tracks[tid]['last_observed_time']),reverse=True)[:3]
        return [dict(track_id=tid,state=self.tracks[tid]['state'],quad_px=self.tracks[tid]['quad'].tolist(),
            current=public(self.tracks[tid]['current']) if self.tracks[tid].get('current') and self.tracks[tid]['state']=='OBSERVED' else None,
            reference=self.tracks[tid]['best'],selected=tid==self.locked) for tid in shown]
