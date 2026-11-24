"""Resident MoGe2 and TensorRT SAM3; both consume exactly the same RGB frame."""
from pathlib import Path
import time,subprocess,selectors
import cv2,numpy as np

class Models:
    def __init__(self,cfg,out):
        import torch
        from moge.model.v2 import MoGeModel
        from tokenizers import Tokenizer
        self.torch=torch;self.cfg=cfg;self.out=Path(out);self.process=None
        self.depth_model=MoGeModel.from_pretrained(cfg['moge_weights']).to(cfg['device']).eval()
        tokenizer=Tokenizer.from_file(cfg['sam_tokenizer']);tokenizer.enable_padding(length=32,pad_id=49407);tokenizer.enable_truncation(max_length=32)
        t=tokenizer.encode(cfg['prompt']);tokens=self.out/'sam_tokens.txt';tokens.write_text(' '.join(map(str,t.ids+t.attention_mask)))
        self.log=open(self.out/'sam.log','w')
        self.process=subprocess.Popen([cfg['sam_executable'],cfg['sam_engines'],'--serve',str(tokens),cfg['prompt'],'unused'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        try:self.wait('READY')
        except Exception:self.close();raise
    def wait(self,marker):
        # The C++ program emits model logs before readiness. Read one byte at a time
        # through os.read to avoid text buffering defeating selector timeouts.
        import os
        end=time.monotonic()+self.cfg.get('worker_timeout_s',120);buffer=b''
        sel=selectors.DefaultSelector();sel.register(self.process.stdout,selectors.EVENT_READ)
        try:
            while time.monotonic()<end:
                if not sel.select(max(0,end-time.monotonic())):break
                c=os.read(self.process.stdout.fileno(),1)
                if not c:raise RuntimeError('SAM worker exited')
                buffer+=c
                if c==b'\n':
                    line=buffer.decode(errors='replace').strip();self.log.write(line+'\n');buffer=b''
                    if line.startswith(marker):return line
            raise TimeoutError('SAM worker timeout')
        finally:sel.close()
    def depth(self,image,K):
        started=time.perf_counter();torch=self.torch;rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);h,w=image.shape[:2]
        if not np.allclose([K[0,0],K[0,2],K[1,2]],[K[1,1],(w-1)/2,(h-1)/2],atol=.01):
            raise ValueError('MoGe adapter requires centered square-pixel rectified pinhole K')
        tensor=torch.from_numpy(rgb.copy()).permute(2,0,1).float().to(self.cfg['device'])/255
        fov=float(np.degrees(2*np.arctan(w/(2*K[0,0]))))
        with torch.inference_mode():
            a=self.depth_model.infer(tensor,resolution_level=self.cfg['resolution_level'],fov_x=fov,use_fp16=self.cfg['device'].startswith('cuda'))
        result={k:v.detach().float().cpu().numpy() for k,v in a.items()}
        self.last_timing={'depth_wall_ms':(time.perf_counter()-started)*1000}
        if self.cfg['device'].startswith('cuda'):
            self.last_timing.update(torch_allocated_bytes=int(torch.cuda.memory_allocated()),torch_reserved_bytes=int(torch.cuda.memory_reserved()))
        return result['depth'],result['mask']>0
    def sam(self,image,index):
        folder=self.out/'model_cache'/f'{index:08d}';folder.mkdir(parents=True)
        path=folder/'rgb.png';cv2.imwrite(str(path),image)
        result=folder/'sam';self.process.stdin.write(str(path)+'\t'+str(result)+'\n');self.process.stdin.flush()
        if self.wait('DONE')!='DONE 0':raise RuntimeError('SAM inference failed')
        import json
        meta=json.loads((result/'detections.json').read_text())
        self.last_timing.update(sam_cuda_event_span_ms=meta.get('cuda_event_span_ms'),sam_wall_ms=meta.get('inference_wall_ms'))
        masks=[cv2.imread(str(result/d['mask']),0)>0 for d in meta['detections']]
        return np.stack(masks) if masks else np.zeros((0,*image.shape[:2]),bool),np.array([d['score'] for d in meta['detections']])
    def close(self):
        if self.process:
            if self.process.stdin:self.process.stdin.close()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if hasattr(self,'log'):self.log.close()
