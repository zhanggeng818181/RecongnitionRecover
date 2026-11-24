"""Depth-only connected plane proposals, independent of object/SAM recognition."""
import numpy as np,cv2
from .geometry import fit_plane,GeometryError

def segment_planes(points,valid,edge_y=160,threshold=.01,min_area=500,max_planes=8):
 p=np.asarray(points,float);valid=np.asarray(valid,bool)&np.isfinite(p).all(-1)&(p[...,2]>0);valid[int(edge_y):]=False
 remaining=cv2.erode(valid.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool);clean=np.where(valid[...,None],p,0);found=[]
 for attempt in range(max_planes):
  if remaining.sum()<min_area:break
  try:plane,_=fit_plane(p[remaining],threshold=threshold,min_ratio=.12,iterations=220,max_points=4000,seed=attempt)
  except GeometryError:break
  n=np.asarray(plane['normal_camera']);inside=remaining&(np.abs(clean@n+plane['d_m'])<=threshold)
  count,labels,stats,_=cv2.connectedComponentsWithStats(inside.astype('uint8'),8)
  remaining &= ~inside
  for k in range(1,count):
   region=labels==k
   if stats[k,cv2.CC_STAT_AREA]<min_area:continue
   try:local,_=fit_plane(p[region],threshold=threshold,min_ratio=.8,iterations=60,max_points=2500)
   except GeometryError:continue
   nn=np.array(local['normal_camera']);region &= np.abs(clean@nn+local['d_m'])<=threshold
   if region.sum()<min_area:continue
   merged=False
   for entry in found:
    if np.dot(nn,entry[0])>np.cos(np.deg2rad(8)) and abs(local['d_m']-entry[1])<threshold:
     # Merge only touching patches of the same estimated plane.
     if (cv2.dilate(region.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)&entry[2]).any():entry[2]|=region;merged=True;break
   if not merged:found.append([nn,local['d_m'],region,local['inlier_rmse_m']])
 found.sort(key=lambda e:-e[2].sum());records=[dict(normal_camera=e[0].tolist(),d_model_units=e[1],area_px=int(e[2].sum()),rmse_model_units=e[3]) for e in found]
 return np.stack([e[2] for e in found]) if found else np.zeros((0,*valid.shape),bool),records
