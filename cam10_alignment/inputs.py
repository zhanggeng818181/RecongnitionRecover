"""OpenCV replay/live frame source and timestamp-matched joint-state input."""
import json,time,threading
from pathlib import Path
import cv2

class FrameSource:
    def __init__(self,cfg):
        self.cfg=cfg;source=cfg['source']
        self.cap=cv2.VideoCapture(int(source) if cfg['mode']=='live' and str(source).isdigit() else source)
        if not self.cap.isOpened():raise ValueError('cannot open cam10 video source')
        self.fps=float(self.cap.get(cv2.CAP_PROP_FPS))
        if cfg['mode']=='replay' and self.fps<=0:raise ValueError('video has no valid FPS')
        self.stop=threading.Event();self.condition=threading.Condition();self.latest=None;self.ended=False
        if cfg['mode']=='live':
            self.thread=threading.Thread(target=self.capture,daemon=True);self.thread.start()
    def capture(self):
        index=0
        try:
            while not self.stop.is_set():
                ok,image=self.cap.read()
                if not ok:break
                with self.condition:
                    self.latest=(index,time.time(),image);self.condition.notify_all()
                index+=1
        finally:
            with self.condition:self.ended=True;self.condition.notify_all()
    def __iter__(self):
        if self.cfg['mode']=='live':
            consumed=-1
            while not self.stop.is_set():
                with self.condition:
                    ready=self.condition.wait_for(lambda:self.ended or (self.latest is not None and self.latest[0]>consumed),timeout=10)
                    if not ready:raise TimeoutError('no live cam10 frame for 10 seconds')
                    if self.latest is None or self.latest[0]<=consumed:return
                    item=self.latest;consumed=item[0]
                yield item
        else:
            index=0
            while True:
                ok,image=self.cap.read()
                if not ok:return
                if index>=self.cfg.get('start_frame',0) and index%self.cfg['stride']==0:
                    yield index,self.cfg.get('timestamp_origin_s',0)+index/self.fps,image
                index+=1
                if self.cfg.get('end_frame_inclusive') is not None and index>self.cfg['end_frame_inclusive']:return
    def close(self):
        self.stop.set();self.cap.release()
        if hasattr(self,'thread'):self.thread.join(timeout=2)

class JointStates:
    def __init__(self,path,clock):
        self.path=Path(path) if path else None;self.clock=clock
    def get(self,timestamp,max_age):
        if self.path is None:return None
        try:
            text=self.path.read_text()
            records=[json.loads(line) for line in text.splitlines() if line.strip()]
        except (OSError,json.JSONDecodeError):return None
        records=[r for r in records if r.get('clock')==self.clock and isinstance(r.get('timestamp_s'),(int,float))]
        if not records:return None
        nearest=min(records,key=lambda s:abs(s['timestamp_s']-timestamp))
        return nearest if abs(nearest['timestamp_s']-timestamp)<=max_age else None
