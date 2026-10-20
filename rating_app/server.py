"""Loopback HTTP app with SQLite storage and role-scoped in-memory sessions."""
import argparse
import json
import secrets
import time
import zipfile
import io
import tempfile
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .study import Study,create_study,csv_text,dump,AXES

ASSETS=Path(__file__).parent/'static'
class App(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,study,port):
        self.study=study;self.sessions={};self.failures=[]
        super().__init__(('127.0.0.1',port),Handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def valid_host(self):return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
    def send(self,data,status=200,mime='application/json; charset=utf-8',cookie=None,attachment=None):
        body=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if cookie:self.send_header('Set-Cookie',cookie)
        if attachment:self.send_header('Content-Disposition',f'attachment; filename="{attachment}"')
        self.end_headers();self.wfile.write(body)
    def session(self,role=None):
        cookie=SimpleCookie()
        try:cookie.load(self.headers.get('Cookie',''))
        except Exception: return None
        token=cookie.get('rating_session');s=self.server.sessions.get(token.value if token else '')
        if not s or s['expires']<time.time() or (role and s['role']!=role):return None
        s['expires']=time.time()+3600
        return s
    def do_GET(self):
        if not self.valid_host():return self.send({'error':'Local host required'},403)
        path=self.path.split('?',1)[0]
        if path in ['/','/admin','/app.js','/style.css']:
            file='index.html' if path in ['/','/admin'] else path[1:]
            return self.send((ASSETS/file).read_bytes(),mime={'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[file.split('.')[-1]])
        s=self.session()
        if not s:return self.send({'error':'請先登入。'},401)
        if path=='/api/state' and s['role']=='rater':return self.send(self.server.study.state(s['participant']))
        if path=='/api/dashboard' and s['role']=='admin':return self.send(self.server.study.dashboard())
        if path=='/api/export' and s['role']=='admin':
            data=self.server.study.export();return self.send(json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8'),attachment='ratings_export.json')
        if path=='/api/export.zip' and s['role']=='admin':
            data=self.server.study.export();buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('ratings_export.json',json.dumps(data,ensure_ascii=False,indent=2))
                z.writestr('ratings_all.csv',csv_text(data['ratings']))
                for axis in AXES:
                    rows=[{k:v for k,v in r.items() if k!='axis'} for r in data['completed_scored'] if r['axis']==axis]
                    z.writestr(axis+'_completed.csv',csv_text(rows))
                z.writestr('README.txt','含全部已存／未存狀態，統計CSV只含已送出評分者的有效整數評分；缺評、無法判斷、略過均未填零。請將整個ZIP交回研究者。')
            return self.send(buffer.getvalue(),mime='application/zip',attachment='human_ratings_export.zip')
        if path=='/api/backup' and s['role']=='admin':
            with tempfile.TemporaryDirectory(dir=self.server.study.folder) as tmp:
                file=Path(tmp)/'ratings.sqlite3';self.server.study.backup(file);body=file.read_bytes()
            return self.send(body,mime='application/octet-stream',attachment='PRIVATE_ratings_backup.sqlite3')
        return self.send({'error':'找不到頁面或權限不足。'},404)
    def do_POST(self):
        origin=self.headers.get('Origin')
        allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
        if not self.valid_host() or origin not in allowed or self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send({'error':'只接受本機同來源JSON操作。'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=16000:raise ValueError('資料大小不正確。')
            data=json.loads(self.rfile.read(size))
            if not isinstance(data,dict):raise ValueError('資料格式不正確。')
            if self.path=='/api/login':
                role=data.get('role');code=data.get('code')
                if role not in ['admin','rater'] or not isinstance(code,str) or len(code)>200:raise ValueError('登入資料不正確。')
                t=time.time();self.server.failures=[x for x in self.server.failures if t-x<60]
                if len(self.server.failures)>=10:return self.send({'error':'嘗試次數過多，請稍候一分鐘。'},429)
                person=self.server.study.login(code,role=='admin')
                if not person:self.server.failures.append(t);return self.send({'error':'代碼不正確，請向研究者確認。'},401)
                for key,s in list(self.server.sessions.items()):
                    if s['participant']==person or s['expires']<t:self.server.sessions.pop(key,None)
                token=secrets.token_urlsafe(32);self.server.sessions[token]=dict(role=role,participant=person,expires=t+3600)
                return self.send({'ok':True,'role':role},cookie=f'rating_session={token}; HttpOnly; SameSite=Strict; Path=/')
            s=self.session()
            if not s:return self.send({'error':'登入逾時，請重新輸入代碼。已儲存的評分仍保留。'},401)
            if self.path=='/api/logout':
                for key,v in list(self.server.sessions.items()):
                    if v is s:self.server.sessions.pop(key,None)
                return self.send({'ok':True},cookie='rating_session=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/')
            if self.path=='/api/activate' and s['role']=='admin':
                if data.get('confirmed') is not True:raise ValueError('請先確認評分設計。')
                self.server.study.activate();return self.send({'ok':True})
            if s['role']!='rater':raise PermissionError('評分者權限才可操作。')
            if self.path=='/api/consent':
                if data.get('agreed') is not True:raise ValueError('尚未同意參與。')
                self.server.study.consent(s['participant']);return self.send({'ok':True})
            if self.path=='/api/save':return self.send({'version':self.server.study.save(s['participant'],data.get('token'),data.get('rating'),data.get('version'))})
            if self.path=='/api/finish':self.server.study.finish(s['participant']);return self.send({'ok':True})
            return self.send({'error':'找不到操作。'},404)
        except (ValueError,TypeError,KeyError) as e:self.send({'error':str(e)},400)
        except PermissionError as e:self.send({'error':str(e)},403)
        except RuntimeError as e:self.send({'error':str(e)},409)
        except Exception:self.send({'error':'無法儲存，請勿關閉頁面，並請研究者檢查本機服務。'},500)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    make=sub.add_parser('prepare');make.add_argument('--responses',type=Path,required=True);make.add_argument('--out',type=Path,required=True)
    make.add_argument('--participants',type=int,default=24);make.add_argument('--items',type=int,default=10);make.add_argument('--pool',type=int,default=120)
    run=sub.add_parser('serve');run.add_argument('--study',type=Path,required=True);run.add_argument('--port',type=int,default=8766);run.add_argument('--open',action='store_true')
    export=sub.add_parser('export');export.add_argument('--study',type=Path,required=True);export.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':
        c=create_study(a.responses,a.out,a.participants,a.items,a.pool);print(f"Prepared draft: {c['participants']} participants, {c['items_per_person']} questions/person, {c['pool_size']} questions in pool. Activate in researcher panel after review.")
    elif a.command=='export':
        if a.out.exists():raise ValueError('匯出檔已存在。請使用新名稱。')
        dump(a.out,Study(a.study).export());print('Exported; missing scores not imputed.')
    else:
        if not (a.study/'ratings.sqlite3').exists():raise ValueError('找不到評分資料庫。')
        server=App(Study(a.study),a.port)
        print(f'Rating page: http://127.0.0.1:{server.server_port}/\nResearcher page: http://127.0.0.1:{server.server_port}/admin\nLocal only, no external API. Keep this process open.',flush=True)
        if a.open:webbrowser.open(f'http://127.0.0.1:{server.server_port}/admin')
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()

if __name__=='__main__':main()
