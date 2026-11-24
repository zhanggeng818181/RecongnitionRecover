"""Explicit calibrated-frame transforms and orientation-only candidate poses."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares, minimize_scalar


def transform(value):
    t=np.asarray(value,float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-5):raise ValueError('invalid SE(3) transform')
    return t


def shortest_rotation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);v=np.cross(a,b);c=np.clip(a@b,-1,1)
    if np.linalg.norm(v)<1e-9:
        if c>0:return np.eye(3)
        axis=np.cross(a,np.eye(3)[np.argmin(abs(a))]);axis/=np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi*axis).as_matrix()
    return Rotation.from_rotvec(v/np.linalg.norm(v)*np.arccos(c)).as_matrix()


class Chain:
    def __init__(self,path,base,tip,joint_order,expected_sha256):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('URDF hash differs from approved calibration')
        root=ET.fromstring(raw);parents={j.find('child').get('link'):j for j in root.findall('joint')};chain=[];seen=set()
        while tip!=base:
            if tip in seen or tip not in parents:raise ValueError('invalid URDF chain')
            seen.add(tip);j=parents[tip];chain.append(j);tip=j.find('parent').get('link')
        self.chain=chain[::-1];self.order=joint_order;self.bounds={}
        for j in self.chain:
            if j.get('type')=='fixed':continue
            if j.get('type') not in ['revolute','continuous','prismatic'] or j.find('mimic') is not None:raise ValueError('unsupported joint')
            if j.get('name') not in joint_order:raise ValueError('unobserved moving joint in chain')
            lim=j.find('limit');self.bounds[j.get('name')]=(-np.pi*2,np.pi*2) if j.get('type')=='continuous' else (float(lim.get('lower')),float(lim.get('upper')))
        if set(self.bounds)!=set(joint_order):raise ValueError('joint order does not match chain')
    def fk(self,q):
        values=dict(zip(self.order,q,strict=True));t=np.eye(4)
        for j in self.chain:
            o=j.find('origin');step=np.eye(4)
            if o is not None:step[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');step[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            t=t@step
            if j.get('type')=='fixed':continue
            axis_node=j.find('axis');axis=np.fromstring(axis_node.get('xyz') if axis_node is not None else '1 0 0',sep=' ');axis/=np.linalg.norm(axis);step=np.eye(4)
            if j.get('type')=='prismatic':step[:3,3]=axis*values[j.get('name')]
            else:step[:3,:3]=Rotation.from_rotvec(axis*values[j.get('name')]).as_matrix()
            t=t@step
        return t
    def ik(self,target,q):
        lower,upper=np.array([self.bounds[n] for n in self.order]).T
        def residual(x):
            t=self.fk(x);return np.r_[t[:3,3]-target[:3,3],Rotation.from_matrix(target[:3,:3]@t[:3,:3].T).as_rotvec()]
        sol=least_squares(residual,np.clip(q,lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=200)
        e=residual(sol.x);return dict(success=bool(sol.success and np.linalg.norm(e[:3])<.002 and np.linalg.norm(e[3:])<np.radians(1)),joints_rad=sol.x.tolist(),position_error_m=float(np.linalg.norm(e[:3])),rotation_error_deg=float(np.degrees(np.linalg.norm(e[3:]))),collision_checked=False)

