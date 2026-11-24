"""Single public entry point for replay/live inference and controller handoff."""
import argparse,json
from pathlib import Path
from .pipeline import run,digest,write
from .robot import RobotInterface

def load_config(path):
    p=Path(path).resolve();cfg=json.loads(p.read_text());root=(p.parent/cfg.pop('project_root','../..')).resolve()
    fields=[(cfg['input'],'source'),(cfg['input'],'joint_states'),(cfg['input'],'stop_file'),
        (cfg['robot'],'urdf'),(cfg['models'],'moge_weights'),(cfg['models'],'sam_executable'),
        (cfg['models'],'sam_tokenizer'),(cfg['models'],'sam_engines'),(cfg['models'],'cache_dir')]
    for section,key in fields:
        value=section.get(key)
        if value and '://' not in str(value) and not (key=='source' and cfg['input']['mode']=='live' and str(value).isdigit()):
            section[key]=str((root/value).resolve())
    return cfg

def reference_plan(bundle):
    bundle=Path(bundle).resolve();ref=json.loads((bundle/'reference.json').read_text());run_root=bundle.parents[2]
    cfg=json.loads((run_root/'config.json').read_text())
    if digest(run_root/'config.json')!=ref['config_sha256'] or digest(run_root/'robot.urdf')!=ref['urdf_sha256']:
        raise ValueError('reference URDF/config fingerprint mismatch')
    state=ref['joint_state']
    if state is None:return dict(status='reference_has_no_synchronized_joints',robot_motion_ready=False)
    cfg['robot']['urdf']=str(run_root/'robot.urdf');robot=RobotInterface(cfg['robot'])
    import numpy as np
    q=[state['positions'][n] for n in cfg['robot']['joint_order']]
    if state.get('position_unit')!='rad' or state.get('clock')!=ref['clock'] or not np.isfinite(q).all():
        raise ValueError('invalid reference joint-state units/clock/values')
    if abs(state['timestamp_s']-ref['geometry']['timestamp_s'])>cfg['robot']['max_state_age_s']:
        raise ValueError('reference joint state not synchronized')
    for name,value in zip(robot.chain.order,q):
        lower,upper=robot.chain.bounds[name]
        if not lower<=value<=upper:raise ValueError('reference joint limit exceeded')
    alignment=robot.propose(ref['geometry'],state,ref['geometry']['timestamp_s'])
    return dict(schema_version=1,status='controller_handoff_draft',robot_motion_ready=False,
        reference_bundle=str(bundle),urdf_sha256=ref['urdf_sha256'],config_sha256=ref['config_sha256'],
        stages=[dict(action='return_reference',joint_names=cfg['robot']['joint_order'],positions_rad=q,
                     require_collision_checked_trajectory=True),
                dict(action='verify_reference',joint_tolerance_rad=.01,require_fresh_joint_feedback=True,
                     require_target_reobserved=True,require_base_unchanged=True),
                dict(action='align_and_standoff',proposal=alignment,require_controller_approval=True)],
        note='No actuator transport is included. Historical geometry requires same reference pose and renewed target observation.')

def main():
    ap=argparse.ArgumentParser(description='cam10 depth -> RGB SAM -> fused planes -> EVT pose proposals')
    sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('run');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('check');p.add_argument('--config',required=True)
    p=sub.add_parser('reference-plan');p.add_argument('--bundle',required=True);p.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.action=='run':result=run(load_config(args.config),args.out)
    elif args.action=='check':
        from .pipeline import validate_config
        cfg=load_config(args.config);validate_config(cfg);robot=RobotInterface(cfg['robot'])
        result=dict(configuration_valid=True,urdf_chain=robot.chain.order,calibration_verified=cfg['robot']['calibration_verified'],robot_motion_ready=False)
    else:result=reference_plan(args.bundle);write(args.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
