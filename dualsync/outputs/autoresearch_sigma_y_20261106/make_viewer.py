"""Build a self-contained local HTML viewer: Ctrl + left click = synchronized observation.

CPU only, reads existing arrays. Magnitudes are divided by the input p99.5 (same scale
as all reports) and stored as uint16 (value * 10000). Writes figures/viewer.html once.
"""
import base64
import json
from pathlib import Path
import sys

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as harness  # noqa: E402
from evaluate import load_evaluator  # noqa: E402

K2 = Path('/data0/zhuoxu/code/DualSync/outputs/autoresearch_snr_refine_20261106/full38_outputs')
CASES = ['MSP-FSE-T/slice_06', 'DYF-FSE-T/slice_01']
CANDIDATES = [('K2 alpha=0.5 (seed avg)', K2), ('sigma_y=0.35', HERE / 'runs/sy0.35'),
              ('sigma_y=0.5', HERE / 'runs/sy0.5'), ('eta=0.5', HERE / 'runs/eta0.5')]
Q = 10000.


def encode(mag):
    q = np.clip(np.round(mag * Q), 0, 65535).astype('<u2')
    return base64.b64encode(q.tobytes()).decode()


def main():
    harness.require_isolated_python()
    out = HERE / 'figures' / 'viewer.html'
    if out.exists():
        raise RuntimeError('refusing to overwrite ' + str(out))
    protocol, base = harness.load_protocols()
    ev = load_evaluator(protocol)
    data = {}
    for case in CASES:
        ref = sio.loadmat(str(Path(protocol['anchor_root']) / (case + '.mat')))
        cg, tuned = np.squeeze(ref['cg_sense']), np.squeeze(ref['diff'])
        refs = {case: dict(cg_sense=cg, tuned=tuned, coeff=float(np.squeeze(ref['coeff'])),
                           orig_shape=tuple(np.ravel(ref['orig_shape']).astype(int)))}
        scale = float(np.percentile(np.abs(cg), 99.5))
        panels = [dict(name='Original input', label='', ok=None, img=encode(np.abs(cg) / scale)),
                  dict(name='Registered final (Tuned)', label='reference', ok=None,
                       img=encode(np.abs(tuned) / scale))]
        for name, root in CANDIDATES:
            m = sio.loadmat(str(root / (case + '.mat')))
            if not np.array_equal(np.squeeze(m['cg_sense']), cg):
                raise ValueError('preprocessing differs: ' + name)
            img = np.squeeze(m['diff'])
            rep = ev.evaluate_images({case: img}, refs, [case], base)
            row = rep['per_case'][case]
            failed = sorted({v['metric'].replace('_', ' ') for v in rep['violations']})
            panels.append(dict(name=name, ok=not failed, img=encode(np.abs(img) / scale),
                               label='bg x%.2f, in-band x%.2f, %s' % (
                                   row['bg_snr_gain'], row['inband_snr_gain'],
                                   'passes all gates' if not failed else 'fails: ' + ', '.join(failed))))
        data[case] = dict(size=int(cg.shape[0]), panels=panels)
    out.write_text(TEMPLATE.replace('__DATA__', json.dumps(data)).replace('__Q__', str(Q)))
    print('wrote', out, round(out.stat().st_size / 1e6, 2), 'MB')


TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>DualSync 结果同步观察</title>
<style>
 body{font-family:-apple-system,"PingFang SC",sans-serif;background:#111;color:#ddd;margin:16px}
 h1{font-size:18px;margin:0 0 6px} .hint{color:#ffd166;font-size:14px;margin-bottom:10px}
 .bar{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:12px;font-size:13px}
 .grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
 .cell{background:#1b1b1b;padding:6px;border-radius:6px} .cell h3{font-size:12px;margin:0 0 4px;height:30px}
 canvas{width:100%;image-rendering:pixelated;display:block;cursor:crosshair}
 .lab{font-size:11px;min-height:28px;margin-top:4px} .ok{color:#06d6a0} .bad{color:#ef476f}
 .val{font-size:11px;font-family:Menlo,monospace;color:#bbb;margin-top:3px;white-space:pre}
 #obsTitle{margin:16px 0 6px;font-size:14px}
</style></head><body>
<h1>DualSync：最终图 vs 各候选（同一输入 p99.5 窗口）</h1>
<div class="hint">按住 <b>Ctrl</b> + <b>鼠标左键</b>点击任意一张图 → 所有图同步放大观察该位置（下排）。</div>
<div class="bar">
 <label>切片 <select id="case"></select></label>
 <label>窗口上限 <input id="wmax" type="range" min="0.3" max="1.5" step="0.05" value="1.0"> <span id="wv">1.00</span></label>
 <label>观察区域 <select id="crop"><option>24</option><option>32</option><option selected>48</option><option>64</option><option>96</option></select> 像素</label>
</div>
<div class="grid" id="main"></div>
<div id="obsTitle">观察区域：尚未选择（Ctrl + 左键点击上方任一图像）</div>
<div class="grid" id="obs"></div>
<script>
const DATA=__DATA__, Q=__Q__;
const caseSel=document.getElementById('case'), wmax=document.getElementById('wmax'), cropSel=document.getElementById('crop');
let arrays=[], point=null;
for(const c of Object.keys(DATA)){const o=document.createElement('option');o.textContent=c;caseSel.appendChild(o);}
function decode(b64){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);
 const v=new Uint16Array(u.buffer),f=new Float32Array(v.length);for(let i=0;i<v.length;i++)f[i]=v[i]/Q;return f;}
function paint(cv,arr,N,x0,y0,w){const ctx=cv.getContext('2d'),im=ctx.createImageData(w,w),m=+wmax.value;
 for(let y=0;y<w;y++)for(let x=0;x<w;x++){const X=x0+x,Y=y0+y,i=(y*w+x)*4;
  const g=(X<0||Y<0||X>=N||Y>=N)?0:Math.max(0,Math.min(255,arr[Y*N+X]/m*255));im.data[i]=im.data[i+1]=im.data[i+2]=g;im.data[i+3]=255;}
 ctx.putImageData(im,0,0);}
function stats(arr,N,px,py,r){let s=0,s2=0,n=0;for(let y=py-r;y<=py+r;y++)for(let x=px-r;x<=px+r;x++){
 if(x<0||y<0||x>=N||y>=N)continue;const v=arr[y*N+x];s+=v;s2+=v*v;n++;}const m=s/n;return [m,Math.sqrt(Math.max(0,s2/n-m*m))];}
function build(){const d=DATA[caseSel.value],N=d.size;arrays=d.panels.map(p=>decode(p.img));point=null;
 const main=document.getElementById('main'),obs=document.getElementById('obs');main.innerHTML='';obs.innerHTML='';
 d.panels.forEach((p,k)=>{const cell=document.createElement('div');cell.className='cell';
  cell.innerHTML='<h3>'+p.name+'</h3>';const cv=document.createElement('canvas');cv.width=cv.height=N;cv.dataset.k=k;cell.appendChild(cv);
  const lab=document.createElement('div');lab.className='lab '+(p.ok===true?'ok':p.ok===false?'bad':'');lab.textContent=p.label;cell.appendChild(lab);
  main.appendChild(cell);
  cv.addEventListener('mousedown',e=>{if(e.button===0&&e.ctrlKey){e.preventDefault();const r=cv.getBoundingClientRect();
   point=[Math.floor((e.clientX-r.left)/r.width*N),Math.floor((e.clientY-r.top)/r.height*N)];render();}});
  cv.addEventListener('contextmenu',e=>{if(e.ctrlKey)e.preventDefault();});
  const oc=document.createElement('div');oc.className='cell';oc.innerHTML='<h3>'+p.name+'</h3>';
  const ov=document.createElement('canvas');oc.appendChild(ov);const val=document.createElement('div');val.className='val';oc.appendChild(val);obs.appendChild(oc);});
 render();}
function render(){const d=DATA[caseSel.value],N=d.size,w=+cropSel.value;document.getElementById('wv').textContent=(+wmax.value).toFixed(2);
 const mains=document.querySelectorAll('#main canvas'),obsC=document.querySelectorAll('#obs canvas'),vals=document.querySelectorAll('#obs .val');
 mains.forEach((cv,k)=>{paint(cv,arrays[k],N,0,0,N);if(point){const ctx=cv.getContext('2d');ctx.strokeStyle='#ffd166';ctx.lineWidth=1.5;
  ctx.strokeRect(point[0]-w/2+.5,point[1]-w/2+.5,w,w);}});
 if(!point){document.getElementById('obsTitle').textContent='观察区域：尚未选择（Ctrl + 左键点击上方任一图像）';return;}
 const [px,py]=point,x0=px-Math.floor(w/2),y0=py-Math.floor(w/2);
 document.getElementById('obsTitle').textContent='观察区域：中心 (x='+px+', y='+py+')，'+w+'×'+w+' 像素，最近邻放大；数值为相对输入 p99.5 的幅值';
 obsC.forEach((cv,k)=>{cv.width=cv.height=w;paint(cv,arrays[k],N,x0,y0,w);const ctx=cv.getContext('2d');
  ctx.strokeStyle='rgba(255,209,102,.8)';ctx.lineWidth=.5;ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,w);ctx.moveTo(0,w/2);ctx.lineTo(w,w/2);ctx.stroke();
  const [m,s]=stats(arrays[k],N,px,py,3);vals[k].textContent='中心值 '+arrays[k][py*N+px].toFixed(3)+'\n7×7 均值 '+m.toFixed(3)+'  标准差 '+s.toFixed(4);});}
caseSel.onchange=build;wmax.oninput=render;cropSel.onchange=render;build();
</script></body></html>
'''

if __name__ == '__main__':
    main()
