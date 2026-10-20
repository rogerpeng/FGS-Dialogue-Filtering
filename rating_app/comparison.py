"""Offline, free-question comparison; kept separate from the balanced rating study."""
import argparse,csv,hashlib,hmac,io,json,random,secrets,sqlite3,threading,unicodedata,re
from pathlib import Path
from http.server import ThreadingHTTPServer
from contextlib import contextmanager
from .server import Handler
from .study import Study,AXES,now,csv_text

@contextmanager
def connection(path):
    c=sqlite3.connect(path)
    try:
        with c:yield c
    finally:c.close()

def normalize(text):
    text=unicodedata.normalize('NFKC',text).strip()
    text=re.sub(r'^(請用簡明的文字說明|請說明以下問題|請簡要回答|我想了解)[:：]\s*','',text)
    return re.sub(r'\s+','',text).rstrip('?？。')

class Comparison:
    def __init__(self,folder,bank,responses):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.bank=json.loads(Path(bank).read_text(encoding='utf-8-sig'))
        raw=[json.loads(s) for s in Path(responses).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        manifest=dict(bank_sha256=hashlib.sha256(Path(bank).read_bytes()).hexdigest(),responses_sha256=hashlib.sha256(Path(responses).read_bytes()).hexdigest())
        m=self.folder/'input_manifest.json'
        if m.exists() and json.loads(m.read_text())!=manifest:raise ValueError('Input changed; use a new comparison folder.')
        m.write_text(json.dumps(manifest),encoding='utf-8')
        key=self.folder/'PRIVATE_token_secret.txt'
        if not key.exists():key.write_text(secrets.token_hex(32),encoding='ascii')
        self.secret=key.read_text().encode();self.groups={};self.tokens={};self.systems=sorted({r['system'] for r in raw})
        for r in raw:
            if r.get('replicate',0)!=0:continue
            token=hmac.new(self.secret,(r['id']+'\0'+r['system']).encode(),hashlib.sha256).hexdigest()
            self.tokens[token]=r
            self.groups.setdefault(normalize(r['question']),[]).append((token,r))
        self.db=self.folder/'comparison_ratings.sqlite3'
        with connection(self.db) as c:
            c.execute('CREATE TABLE IF NOT EXISTS consent (participant TEXT PRIMARY KEY, at TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS ratings (participant TEXT, token TEXT, payload TEXT, version INTEGER, updated_at TEXT, PRIMARY KEY(participant,token))')
            c.execute('CREATE TABLE IF NOT EXISTS audit (participant TEXT, token TEXT, payload TEXT, version INTEGER, at TEXT)')
    def bank_view(self):return [dict(id=r['id'],question=r['question'],available=len(self.groups.get(normalize(r['question']),[]))) for r in self.bank]
    def consent(self,person):
        with connection(self.db) as c:c.execute('INSERT OR IGNORE INTO consent VALUES (?,?)',(person,now()))
    def lookup(self,person,question):
        if not isinstance(question,str) or not 1<=len(question.strip())<=10000:raise ValueError('請輸入問題（最多10000字）。')
        rows=list(self.groups.get(normalize(question),[]))
        random.Random(hmac.new(self.secret,(person+normalize(question)).encode(),hashlib.sha256).hexdigest()).shuffle(rows)
        with connection(self.db) as c:
            scores={t:(json.loads(p),v) for t,p,v in c.execute('SELECT token,payload,version FROM ratings WHERE participant=?',(person,))}
        return dict(question=question,total_configurations=len(self.systems),available=len(rows),answers=[dict(token=t,slot=i+1,answer=r['answer'],saved=scores.get(t,(None,0))[0],version=scores.get(t,(None,0))[1],rateable=r.get('status')!='provider_or_filter_error') for i,(t,r) in enumerate(rows)])
    def save(self,person,data):
        t=data.get('token');p=data.get('rating');version=data.get('version')
        if t not in self.tokens or self.tokens[t].get('status')=='provider_or_filter_error':raise ValueError('這則沒有可評分的實驗回答。')
        if type(version) is not int or version<0:raise ValueError('版本不正確。')
        p=Study.validate(p)
        with connection(self.db) as c:
            c.execute('BEGIN IMMEDIATE')
            if not c.execute('SELECT 1 FROM consent WHERE participant=?',(person,)).fetchone():raise ValueError('請先閱讀說明並同意參與。')
            old=c.execute('SELECT version FROM ratings WHERE participant=? AND token=?',(person,t)).fetchone()
            if (old[0] if old else 0)!=version:raise RuntimeError('資料已在另一頁更新；請重新查詢。')
            at=now();payload=json.dumps(p,ensure_ascii=False)
            c.execute('INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)',(person,t,payload,version+1,at))
            c.execute('INSERT INTO audit VALUES (?,?,?,?,?)',(person,t,payload,version+1,at))
        return dict(version=version+1)
    def export(self):
        with connection(self.db) as c:
            scores=list(c.execute('SELECT participant,token,payload,version,updated_at FROM ratings'))
            consents=[dict(participant=p,at=t) for p,t in c.execute('SELECT * FROM consent')]
        return dict(design='free_question_exploratory_21_configurations',note='Self-selected questions; unequal exposure; do not merge with balanced study as if randomized.',exported_at=now(),consents=consents,ratings=[dict(participant_id=p,item_id=self.tokens[t]['id'],system=self.tokens[t]['system'],question=self.tokens[t]['question'],answer_sha256=hashlib.sha256(self.tokens[t]['answer'].encode()).hexdigest(),rating=json.loads(v),version=n,updated_at=d) for p,t,v,n,d in scores],coverage=self.bank_view())

class ComparisonHandler(Handler):
    def do_GET(self):
        if not self.valid_host():return self.send({'error':'Local host required'},403)
        path=self.path.split('?',1)[0]
        if path in ['/','/admin','/comparison.js']:
            file='comparison.js' if path.endswith('.js') else 'comparison.html'
            return self.send((Path(__file__).parent/'static'/file).read_bytes(),mime='text/javascript; charset=utf-8' if file.endswith('.js') else 'text/html; charset=utf-8')
        if path=='/api/comparison-bank':
            if not self.session():return self.send({'error':'請先登入。'},401)
            return self.send(dict(questions=self.server.comparison.bank_view(),systems=len(self.server.comparison.systems)))
        if path=='/api/comparison-export':
            if not self.session('admin'):return self.send({'error':'僅研究者可匯出。'},403)
            return self.send(json.dumps(self.server.comparison.export(),ensure_ascii=False,indent=2).encode(),attachment='comparison_ratings_export.json')
        return super().do_GET()
    def do_POST(self):
        if self.path not in ['/api/comparison-lookup','/api/comparison-save','/api/comparison-consent']:return super().do_POST()
        if not self.valid_host() or self.headers.get('Origin') not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'} or self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send({'error':'僅接受本機JSON操作。'},403)
        s=self.session('rater')
        if not s:return self.send({'error':'請用評分者代碼登入。'},401)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=50000:raise ValueError('資料大小不正確。')
            data=json.loads(self.rfile.read(size))
            if not isinstance(data,dict):raise ValueError('格式不正確。')
            if self.path.endswith('lookup'):result=self.server.comparison.lookup(s['participant'],data.get('question'))
            elif self.path.endswith('save'):result=self.server.comparison.save(s['participant'],data)
            else:
                if data.get('agreed') is not True:raise ValueError('尚未同意參與。')
                self.server.comparison.consent(s['participant']);result={'ok':True}
            return self.send(result)
        except RuntimeError as e:return self.send({'error':str(e)},409)
        except (TypeError,ValueError,KeyError) as e:return self.send({'error':str(e)},400)
        except Exception:return self.send({'error':'儲存失敗，請保留頁面並聯絡研究者。'},500)

class ComparisonApp(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,study,comparison,port):
        self.study=study;self.comparison=comparison;self.sessions={};self.failures=[]
        super().__init__(('127.0.0.1',port),ComparisonHandler)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--responses',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,default=8768);a=p.parse_args()
    app=ComparisonApp(Study(a.study),Comparison(a.out,a.bank,a.responses),a.port)
    print(f'http://127.0.0.1:{app.server_port}/',flush=True)
    try:app.serve_forever()
    except KeyboardInterrupt:pass
    finally:app.server_close()
if __name__=='__main__':main()
