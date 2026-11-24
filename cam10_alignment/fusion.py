"""Depth-first plane extraction, RGB instance association, rectangular completion."""
import cv2
import numpy as np
from .geometry import fit_plane,GeometryError
from .depth_planes import segment_planes

def mask_quad(q,shape):
    m=np.zeros(shape,np.uint8)
    if np.isfinite(q).all() and np.max(np.abs(q))<100000:
        cv2.fillConvexPoly(m,np.rint(q).astype('int32'),1)
    return m.astype(bool)

def boundary_error(a,b):
    """Symmetric contour Chamfer distance, normalized by image diagonal."""
    kernel=np.ones((3,3),np.uint8)
    ea=a&~cv2.erode(a.astype('uint8'),kernel).astype(bool)
    eb=b&~cv2.erode(b.astype('uint8'),kernel).astype(bool)
    if not ea.any() or not eb.any():return float('inf')
    da=cv2.distanceTransform((~ea).astype('uint8'),cv2.DIST_L2,3)
    db=cv2.distanceTransform((~eb).astype('uint8'),cv2.DIST_L2,3)
    return float((db[ea].mean()+da[eb].mean())/(2*np.hypot(*a.shape)))

def points_from_depth(depth,K):
    y,x=np.indices(depth.shape)
    return np.stack([(x-K[0,2])*depth/K[0,0],(y-K[1,2])*depth/K[1,1],depth],-1)

def fuse(depth,valid,K,masks,scores,cfg):
    points=points_from_depth(depth,K)
    good=valid&np.isfinite(points).all(-1)&(depth>0)
    h,w=depth.shape;edge=int(h*cfg['suction_edge_fraction'])
    good[edge:]=False
    threshold=cfg['plane_threshold_model_units']
    planes,_=segment_planes(points,good,edge_y=edge,threshold=threshold,min_area=cfg['min_plane_pixels'])
    candidates=[];rejections=[]
    for idx,mask in enumerate(masks):
        overlap=[int((mask&p).sum()) for p in planes]
        if not overlap or max(overlap)<cfg['min_plane_pixels']:
            rejections.append(dict(instance_id=idx,reason='no_shared_depth_plane'));continue
        shared=mask&planes[int(np.argmax(overlap))]&good
        try:plane,_=fit_plane(points[shared],threshold=threshold,min_ratio=.7,iterations=160,max_points=4000)
        except GeometryError as e:
            rejections.append(dict(instance_id=idx,reason=str(e)));continue
        n=np.asarray(plane['normal_camera']);d=plane['d_m']
        support=mask&good&(np.abs(points@n+d)<=threshold)
        if support.sum()<cfg['min_plane_pixels']:continue
        xyz=points[support];xyz=xyz-(xyz@n+d)[:,None]*n
        center=xyz.mean(0);_,_,vt=np.linalg.svd(xyz-center,full_matrices=False);basis=vt[:2]
        rect=cv2.boxPoints(cv2.minAreaRect(((xyz-center)@basis.T).astype('float32')))
        corners=center+rect@basis
        if np.any(corners[:,2]<=0):continue
        p=corners@K.T;q=p[:,:2]/p[:,2,None]
        region=mask_quad(q,depth.shape)
        if not region.any():continue
        error=boundary_error(region,mask)
        candidates.append(dict(instance_id=idx,score=float(scores[idx]),normal_camera=n.tolist(),
            d_model_units=float(d),corners_camera_model_units=corners.tolist(),quad_px=q.tolist(),
            center_camera_model_units=corners.mean(0).tolist(),boundary_error=error,
            sam_quad_iou=float((region&mask).sum()/max(1,(region|mask).sum())),
            support_pixels=int(support.sum()),rmse_model_units=plane['inlier_rmse_m'],
            mask=mask,observed_mask=support))
    return candidates,rejections

def public(candidate):
    return {k:v for k,v in candidate.items() if k not in ('mask','observed_mask')}
