import hashlib,json,time,shutil
from pathlib import Path
import cv2,numpy as np
from .inputs import FrameSource,JointStates
from .fusion import fuse,mask_quad
from .tracking import Tracker
from .robot import RobotInterface

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def image_hash(image):return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

def validate_config(c):
    if c['camera']['id']!='cam10' or c['camera']['projection']!='rectified_pinhole':
        raise ValueError('only cam10 rectified pinhole input supported')
    if c['input']['mode'] not in ('live','replay'):raise ValueError('mode must be live or replay')
    if c['input']['mode']=='live' and c['models']['backend']=='cache':raise ValueError('cache is replay-only')
    if c['input']['stride']<1:raise ValueError('positive stride required')
    if c['input']['mode']=='live' and c['input']['clock']!='unix_seconds':
        raise ValueError('OpenCV live timestamps use unix_seconds; hardware timestamps need a custom FrameSource')
    if c['input']['max_frame_age_s']<=0:raise ValueError('positive frame age required')
    K=np.asarray(c['camera']['K_pixels'],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('invalid camera K')
    if not 0<c['fusion']['suction_edge_fraction']<=1:raise ValueError('invalid edge fraction')
    if c['fusion']['confirmation_observations']<1 or c['fusion']['max_prediction_seconds']<0:raise ValueError('invalid tracking settings')
    return K

class CacheModels:
    """Explicit replay test adapter; exact decoded image hash prevents stale reuse."""
    def __init__(self,root):self.root=Path(root)
    def read(self,image,index):
        p=self.root/f'{index:08d}.npz'
        a=np.load(p,allow_pickle=False)
        if str(a['rgb_sha256'])!=image_hash(image):raise ValueError('replay cache image mismatch')
        return a

def snapshot(out,track,image,state,cfg):
    ref=track['reference'];path=out/'references'/f"track-{track['track_id']:04d}"/f"frame-{ref['source_frame']:08d}"
    if path.exists():return str(path.relative_to(out))
    path.mkdir(parents=True);cv2.imwrite(str(path/'rgb.png'),image)
    write(path/'reference.json',dict(schema_version=1,camera_frame='cam10_optical',clock=cfg['input']['clock'],
        geometry=ref,joint_state=state,can_return_to_reference=state is not None,
        urdf_sha256=digest(out/'robot.urdf'),config_sha256=digest(out/'config.json'),
        capture_rgb_sha256=image_hash(image),robot_motion_ready=False))
    return str(path.relative_to(out))

def run(cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);K=validate_config(cfg)
    write(out/'config.json',cfg);shutil.copy2(cfg['robot']['urdf'],out/'robot.urdf')
    robot=RobotInterface(cfg['robot']);tracker=Tracker(cfg['fusion']);source=FrameSource(cfg['input'])
    states=JointStates(cfg['input'].get('joint_states'),cfg['input']['clock'])
    models=None;writer=None;rows=0;references={};start=time.perf_counter()
    records=open(out/'frames.jsonl','w',buffering=1)
    try:
        if cfg['models']['backend']=='cache':models=CacheModels(cfg['models']['cache_dir'])
        elif cfg['models']['backend']=='trt':
            from .adapters import Models
            models=Models(cfg['models'],out)
        else:raise ValueError('unsupported model backend')
        for index,stamp,image in source:
            started=time.perf_counter();h,w=image.shape[:2]
            if [w,h]!=cfg['camera']['image_size']:raise ValueError('image size does not match calibration; no implicit resize')
            if cfg['input'].get('stop_file') and Path(cfg['input']['stop_file']).exists():break
            if isinstance(models,CacheModels):
                a=models.read(image,index);depth=a['depth'];valid=a['valid'].astype(bool)
                if not np.allclose(a['K_pixels'],K,atol=.01):raise ValueError('cache intrinsics mismatch')
                masks=a['masks'].astype(bool);scores=a['scores']
            else:
                depth,valid=models.depth(image,K) # depth first, original RGB SAM second
                masks,scores=models.sam(image,index)
            if depth.shape!=(h,w) or valid.shape!=(h,w) or masks.shape[1:]!=(h,w) or len(scores)!=len(masks):
                raise ValueError('model output shape mismatch')
            candidates,rejected=fuse(depth,valid,K,masks,scores,cfg['fusion'])
            tracks=tracker.update(image,candidates,index,stamp);state=states.get(stamp,cfg['robot']['max_state_age_s'])
            for tid,entry in tracker.tracks.items():
                if entry['best']['source_frame']==index:
                    references[tid]=snapshot(out,dict(track_id=tid,reference=entry['best']),image,state,cfg)
            for t in tracks:
                t['reference_bundle']=references.get(t['track_id'])
            selected=next((t for t in tracks if t['selected']),None)
            proposal=robot.propose(selected['current'],state,stamp) if selected and selected['current'] else dict(status='no_current_selected_plane',robot_motion_ready=False)
            if not cfg['camera']['intrinsics_verified']:
                proposal=dict(status='camera_intrinsics_unverified',robot_motion_ready=False)
            if cfg['input']['mode']=='live' and time.time()-stamp>cfg['input']['max_frame_age_s']:
                proposal=dict(status='frame_too_old_for_control',robot_motion_ready=False)
            overlay=image.copy()
            for t in tracks:
                region=mask_quad(np.asarray(t['quad_px']),(h,w));observed=np.zeros((h,w),bool)
                if t['current']:
                    observed=next(c['observed_mask'] for c in candidates if c['instance_id']==t['current']['instance_id'])
                predicted=region&~observed
                if t['state']!='EXPIRED':
                    overlay[observed]=.75*overlay[observed]+.25*np.array([80,220,70])
                    overlay[predicted]=.90*overlay[predicted]+.10*np.array([0,180,255])
                cv2.polylines(overlay,[np.rint(t['quad_px']).astype('int32')],True,(128,128,128) if t['state']=='EXPIRED' else (0,255,0) if t['selected'] else (0,180,255),1)
                if t['selected'] and t['current']:
                    center=np.asarray(t['current']['center_camera_model_units']);end=center+.1*np.asarray(t['current']['normal_camera'])
                    if end[2]>0:
                        pts=np.stack([center,end])@K.T;uv=np.rint(pts[:,:2]/pts[:,2,None]).astype(int)
                        cv2.arrowedLine(overlay,tuple(uv[0]),tuple(uv[1]),(255,80,80),1,tipLength=.25)
            view=np.hstack([image,overlay])
            if writer is None:
                writer=cv2.VideoWriter(str(out/'review.webm'),cv2.VideoWriter_fourcc(*'VP80'),cfg['output']['preview_fps'],(w*2,h))
                if not writer.isOpened():raise RuntimeError('VP8 video encoder unavailable')
            writer.write(view)
            row=dict(schema_version=1,source_frame=index,timestamp_s=stamp,clock=cfg['input']['clock'],
                tracks=tracks,selected_track_id=tracker.locked,robot_proposal=proposal,rejections=rejected,
                joint_state_synchronized=state is not None,model_timing=getattr(models,'last_timing',None),processing_ms=(time.perf_counter()-started)*1000,
                frame_age_s=time.time()-stamp if cfg['input']['mode']=='live' else None,robot_motion_ready=False)
            records.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');rows+=1
            write(out/'status.json',dict(state='running',observations=rows,source_frame=index))
            if cfg['input'].get('max_observations') and rows>=cfg['input']['max_observations']:break
        summary=dict(state='complete',observations=rows,elapsed_s=time.perf_counter()-start,
            selected_track_id=tracker.locked,best_references=references,robot_motion_ready=False)
        write(out/'summary.json',summary);write(out/'status.json',summary)
        (out/'index.html').write_text('<meta charset="utf-8"><title>cam10 主线验证</title><h1>cam10 深度 / SAM 融合平面</h1><p>左原图，右融合与预测。绿：深度观测；淡橙：补全或历史预测；灰：过期。预览为固定帧率，实时采集时间请查看 JSONL。</p><video controls width="1000" src="review.webm"></video><p><a href="frames.jsonl">逐帧结果</a> · <a href="summary.json">最优参考帧索引</a></p>')
        return summary
    except Exception as e:
        write(out/'status.json',dict(state='failed',observations=rows,error=str(e)));raise
    finally:
        source.close();records.close()
        if writer:writer.release()
        if models and hasattr(models,'close'):models.close()
