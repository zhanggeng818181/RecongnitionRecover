"""Calibrated EVT kinematics boundary. Generates proposals, never actuator commands."""
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import Chain,transform

def unit(v):
    v=np.asarray(v,float);length=np.linalg.norm(v)
    if not np.isfinite(v).all() or length<1e-10:raise ValueError('invalid direction')
    return v/length

def target_frame(current_suction,normal_base,corners_base,standoff,footprint=None):
    """+Z points into box; +X aligns with a rectangle edge, choosing least rotation."""
    n=unit(normal_base);z=-n;options=[]
    for i in range(4):
        edge=corners_base[(i+1)%4]-corners_base[i];edge=edge-z*(edge@z)
        for sign in (-1,1):
            x=sign*unit(edge);y=unit(np.cross(z,x));R=np.column_stack((x,y,z))
            cost=np.linalg.norm(Rotation.from_matrix(R@current_suction[:3,:3].T).as_rotvec())
            extent=np.ptp((corners_base-corners_base.mean(0))@R,axis=0)[:2]
            fits=footprint is None or np.all(extent>=footprint)
            options.append((not fits,cost,i,R))
    _,_,edge_index,R=min(options,key=lambda x:(x[0],x[1]))
    target=np.eye(4);target[:3,:3]=R;target[:3,3]=np.mean(corners_base,axis=0)+n*standoff
    return target,edge_index

class RobotInterface:
    def __init__(self,cfg):
        self.cfg=cfg
        self.chain=Chain(cfg['urdf'],cfg['base_link'],cfg['flange_link'],cfg['joint_order'],cfg['urdf_sha256'])
    def propose(self,face,state,frame_timestamp):
        out=dict(status='calibration_required',robot_motion_ready=False,collision_checked=False)
        required=['T_flange_cam10_optical','T_flange_suction_contact','depth_scale_to_m','standoff_m','suction_size_m']
        missing=[k for k in required if self.cfg.get(k) is None]
        if not self.cfg.get('calibration_verified'):missing.append('calibration_verified')
        if missing:return dict(out,missing=missing)
        if state is None:return dict(out,status='synchronized_joint_state_required')
        if abs(float(state['timestamp_s'])-frame_timestamp)>self.cfg['max_state_age_s']:
            return dict(out,status='stale_joint_state')
        if state.get('position_unit')!='rad':raise ValueError('joint state position_unit must be rad')
        q=np.array([state['positions'][name] for name in self.chain.order],float)
        if not np.isfinite(q).all():raise ValueError('nonfinite joints')
        for name,value in zip(self.chain.order,q):
            lo,hi=self.chain.bounds[name]
            if not lo<=value<=hi:return dict(out,status='joint_limits_exceeded',joint=name)
        scale=float(self.cfg['depth_scale_to_m']);distance=float(self.cfg['standoff_m'])
        if not np.isfinite([scale,distance]).all() or scale<=0 or distance<=0:raise ValueError('positive verified scale/standoff required')
        fc=transform(self.cfg['T_flange_cam10_optical']);fs=transform(self.cfg['T_flange_suction_contact'])
        bf=self.chain.fk(q);bc=bf@fc;bs=bf@fs
        n=unit(bc[:3,:3]@unit(face['normal_camera']))
        corners=np.asarray(face['corners_camera_model_units'])*scale
        if corners.shape!=(4,3) or not np.isfinite(corners).all():raise ValueError('invalid corners')
        corners=corners@bc[:3,:3].T+bc[:3,3]
        required_size=np.asarray(self.cfg['suction_size_m'],float)+2*float(self.cfg['edge_margin_m'])
        target,edge=target_frame(bs,n,corners,distance,required_size)
        # Footprint requirement uses actual tool dimensions and configurable safety inset.
        local=(corners-corners.mean(0))@target[:3,:3]
        extents=np.ptp(local[:,:2],axis=0);size=np.asarray(self.cfg['suction_size_m'],float)
        if size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0):raise ValueError('invalid suction_size_m')
        margin=float(self.cfg['edge_margin_m'])
        if margin<0:raise ValueError('negative margin')
        fits=bool(np.all(extents>=size+2*margin))
        flange=target@np.linalg.inv(fs);ik=self.chain.ik(flange,q)
        relative=Rotation.from_matrix(target[:3,:3]@bs[:3,:3].T)
        return dict(out,status='pose_proposal' if fits and ik['success'] else 'footprint_or_ik_rejected',
            frame_id=self.cfg['base_link'],T_base_suction_target=target.tolist(),T_base_flange_target=flange.tolist(),
            rotation_delta_base_axis_angle_rad=relative.as_rotvec().tolist(),
            normal_sine_residual=float(np.linalg.norm(np.cross(target[:3,2],n))),
            normal_facing_dot=float(target[:3,2]@(-n)),
            edge_sine_residual=float(np.linalg.norm(np.cross(target[:3,0],unit(corners[(edge+1)%4]-corners[edge])))),
            chosen_edge_index=edge,standoff_m=distance,footprint_fits_predicted_rectangle=fits,
            predicted_edge_margins_m=((extents-size)/2).tolist(),ik=ik,
            required_controller_checks=['measured calibration and scale','hidden extent uncertainty','collision and workspace','trajectory and velocity limits','contact and pressure interlocks'])
