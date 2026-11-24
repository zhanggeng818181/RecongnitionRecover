import json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from cam10_alignment.fusion import boundary_error,fuse
from cam10_alignment.tracking import Tracker
from cam10_alignment.robot import target_frame,RobotInterface
from cam10_alignment.cli import load_config,reference_plan
from cam10_alignment.pipeline import run,image_hash

ROOT=Path(__file__).resolve().parents[1]
class MainlineTests(unittest.TestCase):
 def setUp(self):self.cfg=load_config(ROOT/'configs/mainline/replay.json')
 def test_plane_and_boundary(self):
  z=np.ones((256,256),np.float32);mask=np.zeros_like(z,bool);mask[30:130,50:200]=True
  candidates,_=fuse(z,np.ones_like(mask),np.asarray(self.cfg['camera']['K_pixels']),mask[None],np.array([.9]),self.cfg['fusion'])
  self.assertEqual(len(candidates),1);c=candidates[0]
  self.assertGreater(abs(c['normal_camera'][2]),.999)
  self.assertLess(c['boundary_error'],.01)
  self.assertEqual(boundary_error(mask,mask),0)
 def test_directed_parallel_and_edge(self):
  current=np.eye(4);n=np.array([.3,.2,-1.]);n/=np.linalg.norm(n)
  x=np.cross(n,[0,1,0]);x/=np.linalg.norm(x);y=np.cross(n,x)
  corners=np.array([-.4*x-.2*y,.4*x-.2*y,.4*x+.2*y,-.4*x+.2*y])+[0,0,1]
  target,edge=target_frame(current,n,corners,.08)
  self.assertAlmostEqual(target[:3,2]@(-n),1)
  self.assertLess(np.linalg.norm(np.cross(target[:3,0],corners[(edge+1)%4]-corners[edge])),1e-8)
  self.assertAlmostEqual((target[:3,3]-corners.mean(0))@n,.08)
 def test_evt_fk_and_calibration_guard(self):
  robot=RobotInterface(self.cfg['robot']);self.assertEqual(len(robot.chain.order),6)
  self.assertTrue(np.isfinite(robot.chain.fk(np.zeros(6))).all())
  self.assertEqual(robot.propose({},None,0)['status'],'calibration_required')
 def test_calibrated_pose_and_stale_state(self):
  cfg=self.cfg['robot'];cfg.update(calibration_verified=True,T_flange_cam10_optical=np.eye(4).tolist(),T_flange_suction_contact=np.eye(4).tolist(),depth_scale_to_m=1.,standoff_m=.1,suction_size_m=[.1,.1])
  robot=RobotInterface(cfg)
  face=dict(normal_camera=[0,0,-1],corners_camera_model_units=[[-.2,-.2,.1],[.2,-.2,.1],[.2,.2,.1],[-.2,.2,.1]])
  state=dict(timestamp_s=1.,position_unit='rad',positions={n:0. for n in robot.chain.order})
  result=robot.propose(face,state,1.)
  self.assertEqual(result['status'],'pose_proposal')
  self.assertTrue(result['ik']['success'])
  self.assertLess(result['normal_sine_residual'],1e-8)
  self.assertGreater(result['normal_facing_dot'],.999)
  self.assertFalse(result['robot_motion_ready'])
  self.assertEqual(robot.propose(face,state,2.)['status'],'stale_joint_state')
 def test_best_reference_not_first_or_latest(self):
  tracker=Tracker(self.cfg['fusion']);mask=np.zeros((256,256),bool);mask[60:150,60:200]=True
  image=np.zeros((256,256,3),np.uint8)
  for frame,error in enumerate([.05,.01,.04]):
   c=dict(mask=mask,observed_mask=mask,quad_px=[[60,60],[200,60],[200,150],[60,150]],boundary_error=error,instance_id=0,normal_camera=[0,0,-1])
   tracks=tracker.update(image,[c],frame,frame*.1)
  self.assertEqual(tracks[0]['reference']['source_frame'],1)
  lost=tracker.update(image,[],4,.4)
  self.assertIsNone(lost[0]['current'])
 def test_end_to_end_bundle_and_missing_state(self):
  with tempfile.TemporaryDirectory() as tmp:
   tmp=Path(tmp);video=tmp/'input.avi';cache=tmp/'cache';cache.mkdir()
   w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(256,256))
   for _ in range(4):w.write(np.full((256,256,3),100,np.uint8))
   w.release();cap=cv2.VideoCapture(str(video));i=0
   while True:
    ok,image=cap.read()
    if not ok:break
    mask=np.zeros((1,256,256),bool);mask[:,50:150,40:220]=True
    np.savez(cache/f'{i:08d}.npz',rgb_sha256=image_hash(image),depth=np.ones((256,256)),valid=np.ones((256,256),bool),masks=mask,scores=[.9],K_pixels=self.cfg['camera']['K_pixels']);i+=1
   cap.release()
   self.cfg['models'].update(backend='cache',cache_dir=str(cache))
   self.cfg['input'].update(source=str(video),stride=1)
   result=run(self.cfg,tmp/'run');self.assertEqual(result['observations'],4)
   bundle=next((tmp/'run/references').glob('*/*'))
   self.assertEqual(reference_plan(bundle)['status'],'reference_has_no_synchronized_joints')
   ref=json.loads((bundle/'reference.json').read_text())
   ref['joint_state']=dict(timestamp_s=ref['geometry']['timestamp_s'],clock='replay_seconds',position_unit='rad',positions={n:0. for n in self.cfg['robot']['joint_order']})
   (bundle/'reference.json').write_text(json.dumps(ref))
   plan=reference_plan(bundle)
   self.assertEqual(plan['stages'][0]['positions_rad'],[0.]*6)
   self.assertFalse(plan['robot_motion_ready'])
   (tmp/'run/robot.urdf').write_text('changed')
   with self.assertRaises(ValueError):reference_plan(bundle)

if __name__=='__main__':unittest.main()
