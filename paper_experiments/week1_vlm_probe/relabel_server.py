# -*- coding: utf-8 -*-
"""
重标注 · 第二步：人工确认界面 v2（画布直绘 + 自动保存 + 断点续传）
==================================================================
v2 修复：
  1) 画布直接绘制框（不再用 DOM 浮层，大图也不会卡）
  2) 每个操作自动保存到服务器（关页面/崩溃都不丢进度）
  3) 刷新后从服务器读回已保存结果，自动跳到第一张未确认的图
  4) 图片加载失败/超时有提示并可跳过，不会卡死

操作：
  - 点画布上的框选中（红色高亮）
  - 按钮：保留 / 删除 / 类别:破洞污渍褶皱 / 不确定
  - ＋补框：拖拽画新框，弹窗输入类别
  - 整图无瑕疵：删除本图全部候选
  - 下一张 / 上一张；"跳到未确认"自动定位

运行：E:\Miniconda\envs\label-studio\python.exe relabel_server.py
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe")
# 可通过环境变量切换数据集（test 重标时用）：
#   RELABEL_CAND=relabel/relabel_test_candidates.json
#   RELABEL_RESULTS=relabel/relabel_test_results.json
#   RELABEL_IMG_DIR=E:\...\Final_Merged_Dataset_v3\images\train
CAND = Path(os.environ.get("RELABEL_CAND", str(BASE / "relabel" / "relabel_candidates.json")))
RESULTS = Path(os.environ.get("RELABEL_RESULTS", str(BASE / "relabel" / "relabel_results.json")))
IMG_DIR = Path(os.environ.get("RELABEL_IMG_DIR",
                              r"E:\CodeBase_YangJunJie\ClothingInspection\Final_Merged_Dataset_v3\images\val"))
PORT = int(os.environ.get("RELABEL_PORT", "8766"))
LOCK = threading.Lock()

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>val 集重标注 - 人工确认 v2</title>
<style>
  body { font-family:"Microsoft YaHei",sans-serif; max-width:1000px; margin:20px auto; padding:0 16px; background:#f5f6fa; }
  h2 { color:#333; }
  #bar { height:8px; background:#ddd; border-radius:4px; margin:6px 0 14px; }
  #bar>div { height:100%; background:#4c8bf5; border-radius:4px; transition:width .2s; }
  #status { position:fixed; top:10px; right:16px; background:#2c3e50; color:#fff; padding:6px 12px; border-radius:6px; font-size:13px; }
  #meta { color:#555; font-size:13px; margin:4px 0; }
  #tip { color:#666; font-size:13px; min-height:18px; margin:6px 0; }
  canvas { border:1px solid #888; background:#fff; max-width:100%; cursor:crosshair; display:block; margin:0 auto; }
  .btns { margin:12px 0; display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
  .btns button { padding:9px 16px; font-size:14px; border:none; border-radius:6px; cursor:pointer; color:#fff; }
  #bKeep{background:#27ae60} #bDel{background:#e74c3c} #bHole{background:#c0392b} #bStain{background:#e67e22}
  #bWrinkle{background:#9b59b6} #bUn{background:#95a5a6} #bAdd{background:#2980b9} #bAllClean{background:#7f8c8d}
  #bPrev{background:#34495e} #bNext{background:#2c3e50; font-weight:bold;} #bJump{background:#16a085}
  #err { color:#e74c3c; font-weight:bold; margin:8px 0; }
</style>
</head>
<body>
<div id="status">加载中...</div>
<h2>val 集重标注（人工确认 v2）</h2>
<div id="bar"><div id="barfill"></div></div>
<div id="meta"></div>
<canvas id="cv"></canvas>
<div id="err"></div>
<div class="btns">
  <button id="bKeep">保留 ✓</button>
  <button id="bDel">删除 ✗</button>
  <button id="bHole">类别:破洞</button>
  <button id="bStain">类别:污渍</button>
  <button id="bWrinkle">类别:褶皱</button>
  <button id="bUn">不确定</button>
  <button id="bAdd">＋补框</button>
  <button id="bAllClean">整图无瑕疵</button>
  <button id="bPrev">← 上一张</button>
  <button id="bNext">下一张 →</button>
  <button id="bJump">跳到未确认</button>
</div>
<div id="tip"></div>
<script>
let data=[], cur=0, img=null, boxes=[], sel=-1, addMode=false, drawing=null;
let res = {};   // image -> {actions:{},added:[]}
let dirty = false;

function $id(x){ return document.getElementById(x); }

function loadImg(src){
  return new Promise((ok,fail)=>{
    const i=new Image();
    const t=setTimeout(()=>fail(new Error('图片加载超时')),25000);
    i.onload=()=>{ clearTimeout(t); ok(i); };
    i.onerror=()=>{ clearTimeout(t); fail(new Error('图片加载失败')); };
    i.src=src;
  });
}

async function show(){
  $id('err').textContent='';
  const d=data[cur];
  $id('status').textContent=(cur+1)+' / '+data.length+' 张';
  $id('barfill').style.width=(cur/data.length*100)+'%';
  const savedN=Object.keys(res).length;
  $id('meta').textContent='图片: '+d.image+' | 候选框 '+d.candidates.length+
    ' 个（虚线=自动保留） | 已保存 '+savedN+'/'+data.length+' 张 | 本图操作自动保存';
  try{
    img = await loadImg('/img/'+encodeURIComponent(d.image));
  }catch(e){
    $id('err').textContent='⚠ 图片加载失败: '+e.message+'（点"下一张"跳过）';
    img = new Image(); img.width=d.width; img.height=d.height;
  }
  const cv=$id('cv');
  cv.width=d.width; cv.height=d.height;
  boxes = d.candidates.map((c,i)=>({...c, keep: c.auto_keep!==false,
    category: c.vlm_category||null, idx:i}));
  const prev = res[d.image];
  if(prev){
    for(const [id,act] of Object.entries(prev.actions||{})){
      const b=boxes.find(x=>x.id===id);
      if(b){ b.keep=act.keep; b.category=act.category; }
    }
    for(const a of prev.added||[]){
      boxes.push({id:'add'+boxes.length, source:'add', box:a.box, keep:true,
        category:a.category, idx:boxes.length});
    }
  }
  sel=-1; render(); updateTip();
}

function render(){
  const cv=$id('cv'), ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  if(img && img.width>1){ ctx.drawImage(img,0,0); }
  boxes.forEach((b,i)=>{
    if(!b.keep) return;
    const isAuto=(b.source==='yolo'&&b.auto_keep);
    ctx.strokeStyle=(i===sel)?'#e74c3c':(isAuto?'#4c8bf5':'#f39c12');
    ctx.lineWidth=(i===sel)?4:2;
    ctx.setLineDash(isAuto?[8,6]:[]);
    const [x1,y1,x2,y2]=b.box;
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.setLineDash([]);
    ctx.font='bold 15px sans-serif';
    ctx.fillStyle='rgba(0,0,0,0.7)';
    const lbl=b.id+' '+b.category;
    const tw=ctx.measureText(lbl).width;
    ctx.fillStyle=(i===sel)?'#e74c3c':'#2980b9';
    ctx.fillRect(x1, Math.max(0,y1-20), tw+8, 18);
    ctx.fillStyle='#fff';
    ctx.fillText(lbl, x1+4, Math.max(14,y1-6));
  });
  if(drawing){
    ctx.strokeStyle='#e74c3c'; ctx.lineWidth=2;
    ctx.strokeRect(drawing[0],drawing[1],drawing[2]-drawing[0],drawing[3]-drawing[1]);
  }
}

function canvasPos(e){
  const cv=$id('cv'), r=cv.getBoundingClientRect();
  return [ (e.clientX-r.left)*cv.width/r.width, (e.clientY-r.top)*cv.height/r.height ];
}

function hitTest(x,y){
  let best=-1, bestD=1e9;
  boxes.forEach((b,i)=>{
    if(!b.keep) return;
    const [x1,y1,x2,y2]=b.box;
    const cx=(x1+x2)/2, cy=(y1+y2)/2;
    const inside = x>=x1&&x<=x2&&y>=y1&&y<=y2;
    const d=Math.hypot(x-cx,y-cy);
    if(inside){ best=i; return; }
    if(d<bestD){ bestD=d; best=i; }
  });
  return best;
}

function updateTip(){
  if(sel>=0&&boxes[sel]){
    const b=boxes[sel];
    $id('tip').textContent='已选: '+b.id+' | 来源:'+b.source+' | 千问:'+
      (b.vlm_defect===true?'有':(b.vlm_defect===false?'无':'?'))+'('+(b.vlm_category||'-')+')'+
      ' | 保留:'+b.keep+' | 类别:'+(b.category||'未定');
  } else {
    $id('tip').textContent=addMode?'补框模式：在图上拖拽画新框':'点击图上的框选中（或点"补框"新增）';
  }
}

function saveCur(quiet){
  if(!res[d_cur()]) res[d_cur()]={actions:{},added:[]};
  const r=res[d_cur()];
  boxes.forEach(b=>{ if(b.source!=='add') r.actions[b.id]={keep:b.keep, category:b.category}; });
  r.added=boxes.filter(b=>b.source==='add'&&b.keep).map(b=>({box:b.box, category:b.category}));
  if(!quiet) $id('status').textContent=(cur+1)+' / '+data.length+' 张（保存中…）';
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:d_cur(), record:r})})
    .catch(e=>{ $id('err').textContent='⚠ 保存失败: '+e.message; });
}

function d_cur(){ return data[cur].image; }

function act(fn){
  if(sel>=0){ fn(); render(); updateTip(); saveCur(); }
}
$id('bKeep').onclick=()=>act(()=>{boxes[sel].keep=true;});
$id('bDel').onclick=()=>act(()=>{boxes[sel].keep=false;});
$id('bHole').onclick=()=>act(()=>{boxes[sel].category='hole';});
$id('bStain').onclick=()=>act(()=>{boxes[sel].category='stain';});
$id('bWrinkle').onclick=()=>act(()=>{boxes[sel].category='wrinkle';});
$id('bUn').onclick=()=>act(()=>{boxes[sel].category=null;});
$id('bAllClean').onclick=()=>{ boxes.forEach(b=>{if(b.source!=='add') b.keep=false;}); render(); updateTip(); saveCur(); };
$id('bAdd').onclick=()=>{ addMode=!addMode; $id('bAdd').style.opacity=addMode?0.6:1; updateTip(); };
$id('bPrev').onclick=async()=>{ saveCur(); if(cur>0){ cur--; await show(); } };
$id('bNext').onclick=async()=>{ saveCur(); if(cur<data.length-1){ cur++; await show(); } else { alert('全部完成！'); } };
$id('bJump').onclick=async()=>{ saveCur(); const first=data.findIndex(d=>!res[d.image]); if(first>=0){ cur=first; await show(); } else { alert('全部已确认！'); } };

// 键盘快捷键
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  const k=e.key.toLowerCase();
  if(k==='1') $id('bHole').click();
  else if(k==='2') $id('bStain').click();
  else if(k==='3') $id('bWrinkle').click();
  else if(k==='4') $id('bDel').click();
  else if(k==='5') $id('bKeep').click();
  else if(k==='n') $id('bNext').click();
  else if(k==='p') $id('bPrev').click();
});

// 画布交互
const cv=$id('cv');
cv.addEventListener('mousedown',e=>{
  const [x,y]=canvasPos(e);
  if(addMode){ drawing=[x,y,x,y]; return; }
  sel=hitTest(x,y); render(); updateTip();
});
cv.addEventListener('mousemove',e=>{
  if(!drawing) return;
  const [x,y]=canvasPos(e);
  drawing[2]=x; drawing[3]=y; render();
});
cv.addEventListener('mouseup',e=>{
  if(!drawing) return;
  const [x,y]=canvasPos(e);
  const [x1,y1]=drawing, x2=x, y2=y;
  drawing=null; addMode=false; $id('bAdd').style.opacity=1;
  const bx1=Math.min(x1,x2),by1=Math.min(y1,y2),bx2=Math.max(x1,x2),by2=Math.max(y1,y2);
  if(bx2-bx1<10||by2-by1<10){ render(); return; }
  const cat=prompt('新框类别：hole / stain / wrinkle（留空=hole）','hole')||'hole';
  boxes.push({id:'add'+boxes.length, source:'add',
    box:[Math.round(bx1),Math.round(by1),Math.round(bx2),Math.round(by2)],
    keep:true, category:cat, idx:boxes.length});
  saveCur(); render(); updateTip();
});

async function init(){
  try{
    const [cr,rr]=await Promise.all([fetch('/candidates'), fetch('/results')]);
    data=await cr.json();
    const saved=await rr.json();
    if(Array.isArray(saved)) saved.forEach(r=>{ res[r.image]=r; });
    if(!data.length){ $id('status').textContent='无数据'; return; }
    // 断点续传：跳到第一张未确认的
    const first=data.findIndex(d=>!res[d.image]);
    cur = first>=0 ? first : 0;
    await show();
  }catch(e){
    $id('err').textContent='初始化失败: '+e.message;
  }
}
init();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/candidates":
            if not CAND.exists():
                self._send(404, "{\"error\":\"relabel_candidates.json 不存在\"}")
            else:
                self._send(200, CAND.read_text(encoding="utf-8"))
        elif path == "/results":
            if RESULTS.exists():
                self._send(200, RESULTS.read_text(encoding="utf-8"))
            else:
                self._send(200, "[]")
        elif path.startswith("/img/"):
            from urllib.parse import unquote
            name = unquote(os.path.basename(path))
            fp = IMG_DIR / name
            if fp.exists():
                ctype = "image/png" if fp.suffix.lower() == ".png" else "image/jpeg"
                self._send(200, fp.read_bytes(), ctype)
            else:
                self._send(404, "not found")
        else:
            self._send(404, "not found")

    def do_POST(self):
        if self.path != "/save":
            self._send(404, "not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._send(400, "{\"error\":\"bad json\"}")
            return
        with LOCK:
            results = []
            if RESULTS.exists():
                try:
                    results = json.loads(RESULTS.read_text(encoding="utf-8"))
                except Exception:
                    results = []
            results = [r for r in results if r.get("image") != data.get("image")]
            results.append(data)
            RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        self._send(200, "{\"ok\":true}")


if __name__ == "__main__":
    if not CAND.exists():
        sys.exit("[error] 先运行 relabel_pipeline.py")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"重标注界面 v2 已启动: http://127.0.0.1:{PORT}")
    print("提示：之前已保存的进度会自动恢复，并跳到第一张未确认的图片")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
