"""Loopback-only research demo. Offline by default; no legacy login/database."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from .pipeline import DialoguePipeline, FixtureBackend, VARIANTS

PAGE = r"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>宗教對話研究示範</title><style>
body{font:17px/1.7 system-ui,sans-serif;background:#f4f3ef;color:#20312e;margin:0}
main{max-width:760px;margin:50px auto;padding:28px}h1{font-size:30px;margin-bottom:8px}
.mode{background:#e3ece5;padding:12px 18px;border-radius:8px}textarea{box-sizing:border-box;width:100%;padding:14px;font:inherit;min-height:130px;border:1px solid #bac4bf;border-radius:8px}
button{font:inherit;background:#245e4c;color:white;border:0;border-radius:7px;padding:10px 22px;margin-top:14px;cursor:pointer}
button:disabled{opacity:.5}article{background:white;padding:24px;margin-top:28px;border-radius:10px;white-space:pre-wrap}
small{color:#51645c}label{display:block;margin:28px 0 8px}
</style><main><small>FGS · Revision 3</small><h1>宗教對話研究示範</h1>
<p class="mode" id="mode">載入模式中…</p><p>本系統是研究用學習助理，不代表星雲大師或佛光山發言。</p>
<form id="form"><label for="q">想討論的問題</label><textarea id="q" maxlength="10000" required>如何在與家人意見不合時保持尊重？</textarea>
<button id="send">送出問題</button></form><article id="answer" aria-live="polite">回答會顯示在這裡。</article><small id="status"></small>
<h2>JSON 轉 CSV</h2><p>支援 JSON 物件、物件陣列與 JSONL。檔案只傳至本機轉換，不會送到模型服務。</p>
<form id="convert"><input type="file" id="jsonfile" accept=".json,.jsonl,application/json" required>
<label for="recordskey">資料陣列欄位（例如 data；一般陣列或 JSONL 留空）</label><input id="recordskey">
<label><input type="checkbox" id="chatpairs">另加 question / answer 欄位（單輪 messages 格式）</label>
<button id="convertbtn">轉換並下載 CSV</button><p id="convertstatus" aria-live="polite"></p></form></main>
<script>
fetch('/mode').then(r=>r.json()).then(x=>document.getElementById('mode').textContent=x.description);
document.getElementById('form').addEventListener('submit',async e=>{e.preventDefault();const b=document.getElementById('send');b.disabled=true;
document.getElementById('answer').textContent='處理中…';document.getElementById('status').textContent='';
try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value})});
const x=await r.json();document.getElementById('answer').textContent=x.answer||x.error;document.getElementById('status').textContent=x.status||'';}
catch(e){document.getElementById('answer').textContent='連線失敗，請確認本機服務仍在執行。';}finally{b.disabled=false;}});
document.getElementById('convert').addEventListener('submit',async e=>{e.preventDefault();const b=document.getElementById('convertbtn');b.disabled=true;
const status=document.getElementById('convertstatus');try{const f=document.getElementById('jsonfile').files[0];if(!f||f.size>4000000)throw Error('請選擇 4 MB 以下的 JSON/JSONL；較大檔案請使用命令列。');
const r=await fetch('/convert',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:await f.text(),records_key:document.getElementById('recordskey').value||null,chat_pairs:document.getElementById('chatpairs').checked})});
const x=await r.json();if(!r.ok)throw Error(x.error);const u=URL.createObjectURL(new Blob(['\uFEFF',x.csv],{type:'text/csv;charset=utf-8'}));
const a=document.createElement('a');a.href=u;a.download=f.name.replace(/\.(jsonl?|ndjson)$/i,'')+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);status.textContent='完成：'+x.rows+' 筆資料。';
}catch(e){status.textContent=e.message;}finally{b.disabled=false;}});
</script></html>"""


def make_handler(pipeline, variant, live=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # do not log private user questions

        def send_json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/mode":
                self.send_json({"description": "即時 API 模式：會向設定的模型服務送出問題。" if live else
                                "離線示範模式：使用固定示例回答，不是即時 LLM，也不是研究結果。"})
            elif self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            if self.path not in {"/ask", "/convert"}:
                self.send_json({"error": "Not found"}, 404)
                return
            # Only local same-origin browser requests; no CORS, external binding or forms.
            expected = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if self.headers.get("Origin") not in expected or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_json({"error": "Same-origin JSON request required"}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= (25000000 if self.path == "/convert" else 50000):
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object")
                if self.path == "/convert":
                    from .conversion import parse_records, csv_text
                    if not isinstance(payload.get("text"), str):
                        raise ValueError("請提供 JSON 文字")
                    rows = parse_records(payload["text"], payload.get("records_key"), bool(payload.get("chat_pairs")))
                    content, fields = csv_text(rows)
                    self.send_json({"csv": content, "rows": len(rows), "columns": fields})
                    return
                question = payload["question"]
                result = pipeline.run(question, variant, sample_id="interactive")
                self.send_json({"answer": result["answer"], "status": result["status"]})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc) if self.path == "/convert" else "請提供有效的文字問題。"}, 400)
    return Handler


def serve(port=8765, config_path=None, question_model=None, answer_model=None, live=False, cache_dir="runs/web_cache"):
    if not 1 <= port <= 65535:
        raise ValueError("Invalid port")
    if live:
        if not config_path or not question_model or not answer_model:
            raise ValueError("Live web demo requires --config, --question-model and --answer-model")
        import joblib
        from .api import ChatClient, APIBackend, LLMGuard
        from .io import read_json
        config = read_json(config_path)
        client = ChatClient(config, cache_dir, live=True)
        backend = APIBackend(client, config["generator_model"], config["refiner_model"])
        guard = LLMGuard(client, config["guard_model"]) if config.get("guard_model") else None
        pipeline = DialoguePipeline(backend, joblib.load(question_model), joblib.load(answer_model), guard,
                                    config.get("max_generation_attempts", 3), config.get("max_refinements", 3))
        variant = VARIANTS["full"]
    else:
        if config_path or question_model or answer_model:
            raise ValueError("Offline web demo uses fixtures; enable --live to use configured models")
        pipeline, variant = DialoguePipeline(FixtureBackend()), VARIANTS["refinement_only"]
    server = HTTPServer(("127.0.0.1", port), make_handler(pipeline, variant, live))
    print(f"Local demo: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
