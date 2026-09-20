# -*- coding: utf-8 -*-
"""
标注审计 · 本地网页点选工具（零依赖，仅用 Python 标准库）
==========================================================
启动后打开 http://127.0.0.1:8765 ，逐张看图并点选：
    1 = 破洞   2 = 污渍   3 = 褶皱   4 = 无瑕疵   5 = 不确定
每点一下自动保存到 audit_results.json（增量保存，可随时关掉重开）。

运行：
    E:\Miniconda\envs\label-studio\python.exe audit_server.py
"""
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(r"E:\CodeBase_YangJunJie\Paper\paper_experiments\week1_vlm_probe")
SAMPLES = BASE / "hard_samples" / "audit_samples.json"
RESULTS = BASE / "hard_samples" / "audit_results.json"
CROPS = BASE / "hard_samples" / "crops"
PORT = int(os.environ.get("AUDIT_PORT", "8765"))

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>标注审计 - 布料瑕疵核实</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; background:#f5f6fa; }
  h2 { color:#333; }
  #progress { font-size: 15px; color:#555; margin: 8px 0; }
  .bar { height: 8px; background:#ddd; border-radius:4px; margin: 6px 0 16px; }
  .bar > div { height:100%; background:#4c8bf5; border-radius:4px; transition:width .2s; }
  img { max-width:100%; max-height:460px; border:1px solid #ccc; background:#fff; display:block; margin:0 auto 14px; }
  #meta { text-align:center; color:#666; font-size:13px; margin-bottom:10px; }
  .btns { display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }
  .btns button { padding:12px 22px; font-size:16px; border:none; border-radius:8px; cursor:pointer; color:#fff; }
  .btns button:active { transform:scale(.96); }
  #b1{background:#e74c3c} #b2{background:#e67e22} #b3{background:#9b59b6} #b4{background:#27ae60} #b5{background:#95a5a6}
  .hint { text-align:center; color:#999; font-size:12px; margin-top:12px; }
  #done { text-align:center; font-size:20px; color:#27ae60; padding:60px 0; }
</style>
</head>
<body>
<h2>标注审计：请判断图中布料是否有瑕疵</h2>
<p id="progress">加载中...</p>
<div class="bar"><div id="barfill"></div></div>
<div id="stage">
  <div id="meta"></div>
  <img id="img" alt="loading...">
  <div class="btns">
    <button id="b1">1 破洞</button>
    <button id="b2">2 污渍</button>
    <button id="b3">3 褶皱</button>
    <button id="b4">4 无瑕疵</button>
    <button id="b5">5 不确定</button>
  </div>
  <p class="hint">快捷键：1~5 直接按键盘。图片会自动切到下一张。</p>
</div>
<div id="done" style="display:none"></div>
<script>
let samples = [], cur = 0, answers = {};
const KEY = { '1':'hole','2':'stain','3':'wrinkle','4':'clean','5':'unsure' };

async function init() {
  const r = await fetch('/samples');
  samples = await r.json();
  if (!samples.length) { document.getElementById('done').style.display='block'; document.getElementById('done').textContent='没有待审计样本'; return; }
  show();
}
function show() {
  if (cur >= samples.length) { finish(); return; }
  const s = samples[cur];
  document.getElementById('progress').textContent = `第 ${cur+1} / ${samples.length} 张`;
  document.getElementById('barfill').style.width = (cur/samples.length*100) + '%';
  document.getElementById('meta').textContent =
    `类型: ${s.case_type} | 标注: ${s.gt_cls} | 千问判断: ${s.vlm_is_defect===true?'有缺陷':'无缺陷'}${s.vlm_category ? ' ('+s.vlm_category+')' : ''}`;
  document.getElementById('img').src = s.img;
}
function answer(v) {
  if (cur >= samples.length) return;
  const s = samples[cur];
  answers[s.id] = v;
  fetch('/answer', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ id: s.id, answer: v }) });
  cur++;
  show();
}
function finish() {
  document.getElementById('stage').style.display = 'none';
  document.getElementById('done').style.display = 'block';
  document.getElementById('done').innerHTML = '✅ 全部完成！共 ' + samples.length + ' 张。<br>结果已自动保存，可以关闭本页面了。';
}
document.addEventListener('keydown', e => { if (KEY[e.key]) answer(KEY[e.key]); });
document.getElementById('b1').onclick = () => answer('hole');
document.getElementById('b2').onclick = () => answer('stain');
document.getElementById('b3').onclick = () => answer('wrinkle');
document.getElementById('b4').onclick = () => answer('clean');
document.getElementById('b5').onclick = () => answer('unsure');
init();
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 安静模式
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
        elif path == "/samples":
            if not SAMPLES.exists():
                self._send(404, "{\"error\":\"audit_samples.json 不存在，先运行 build_audit_samples.py\"}")
            else:
                self._send(200, SAMPLES.read_text(encoding="utf-8"))
        elif path.startswith("/img/"):
            cid = os.path.basename(path)
            fp = CROPS / cid
            if fp.exists():
                ctype, _ = mimetypes.guess_type(str(fp)) or ("image/jpeg", None)
                self._send(200, fp.read_bytes(), ctype or "image/jpeg")
            else:
                self._send(404, "not found")
        else:
            self._send(404, "not found")

    def do_POST(self):
        if self.path != "/answer":
            self._send(404, "not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._send(400, "{\"error\":\"bad json\"}")
            return
        results = []
        if RESULTS.exists():
            try:
                results = json.loads(RESULTS.read_text(encoding="utf-8"))
            except Exception:
                results = []
        # 去重：同一 id 只保留最后一次
        results = [r for r in results if r.get("id") != data.get("id")]
        results.append(data)
        RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        self._send(200, "{\"ok\":true}")

if __name__ == "__main__":
    if not SAMPLES.exists():
        sys.exit("[error] 先运行: python build_audit_samples.py")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"审计工具已启动: http://127.0.0.1:{PORT}")
    print("浏览器打开后逐张看图点选（或按键盘 1~5），结果自动保存到 audit_results.json")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
