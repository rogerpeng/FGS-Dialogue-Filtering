import csv
import hashlib
import io
import json
import random
import secrets
import sqlite3
from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SYSTEMS=['generic_b1','domain_b1','domain_refinement_only','word2vec__full','tfidf__full','transformer__full']
AXES=['appropriateness','clarity','helpfulness']
FLAGS=['fabricated_quote','impersonation','harmful_or_biased']
def now(): return datetime.now(timezone.utc).isoformat()
def sha(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def create_study(responses,out,participants=24,items_per_person=10,pool_size=120,seed=20261014):
    responses,out=Path(responses),Path(out)
    if out.exists(): raise ValueError('輸出資料夾已存在；請使用新名稱，避免覆蓋評分。')
    if not (2<=participants<=500 and 2<=items_per_person<=240 and items_per_person%2==0 and pool_size%2==0 and items_per_person<=pool_size):raise ValueError('人數2–500；每人題數及題庫須為正偶數，每人題數不可超過題庫。')
    raw=[json.loads(line) for line in responses.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    groups={}
    for r in raw:
        if r['system'] not in SYSTEMS:continue
        if r.get('replicate',0)!=0:continue
        if r['status']=='provider_or_filter_error':raise ValueError('有未完成的模型輸出。')
        g=groups.setdefault(r['id'],{})
        if r['system'] in g:raise ValueError('重複回答。')
        g[r['system']]=r
    if not groups or any(set(v)!=set(SYSTEMS) for v in groups.values()):raise ValueError('六個核心系統必須在每題均有回答。')
    for g in groups.values():
        if len({(r['question'],r['category']) for r in g.values()})!=1:raise ValueError('相同題目的問題文字／分類不一致。')
    rng=random.Random(seed);selected={}
    for category in ['related','unrelated']:
        ids=sorted(i for i,g in groups.items() if next(iter(g.values()))['category']==category)
        rng.shuffle(ids)
        if len(ids)<pool_size//2:raise ValueError('該類別題數不足。')
        selected[category]=ids[:pool_size//2]
    out.mkdir(parents=True)
    study_id='rating-'+secrets.token_hex(8);admin=secrets.token_urlsafe(24)
    con=sqlite3.connect(out/'ratings.sqlite3');con.executescript('''
    PRAGMA foreign_keys=ON;
    CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE participants(id TEXT PRIMARY KEY,code_hash TEXT NOT NULL UNIQUE,consented_at TEXT,finished_at TEXT);
    CREATE TABLE assignments(token TEXT PRIMARY KEY,participant TEXT NOT NULL REFERENCES participants(id),ordinal INTEGER NOT NULL,item_id TEXT NOT NULL,system TEXT NOT NULL,question TEXT NOT NULL,answer TEXT NOT NULL,category TEXT NOT NULL,slot INTEGER NOT NULL,UNIQUE(participant,ordinal),UNIQUE(participant,item_id,system));
    CREATE TABLE ratings(token TEXT PRIMARY KEY REFERENCES assignments(token),payload TEXT NOT NULL,version INTEGER NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,at TEXT NOT NULL,participant TEXT,token TEXT,action TEXT NOT NULL,payload TEXT);
    ''')
    config=dict(schema_version=1,study_id=study_id,created_at=now(),status='draft',participants=participants,items_per_person=items_per_person,
                pool_size=pool_size,systems=SYSTEMS,seed=seed,axes=AXES,risk_flags=FLAGS,source_sha256=hashlib.sha256(responses.read_bytes()).hexdigest(),
                selection='class-balanced seeded shuffled subset, before human scores',primary_axis='appropriateness',
                primary_contrast=['tfidf__full','word2vec__full'],
                limitations=['Sample allocation is not a power calculation.','General raters do not certify doctrine or safety.','Repeated questions and raters are dependent.','Known duplicate system responses retained as paired outcomes.'])
    con.executemany('INSERT INTO meta VALUES(?,?)',[('config',json.dumps(config,ensure_ascii=False)),('active','false'),('admin_hash',sha(admin))])
    codes=[];exposure=Counter()
    for p in range(participants):
        participant=f'P{p+1:03d}';code=secrets.token_hex(5).upper();codes.append(dict(participant_id=participant,access_code=code))
        con.execute('INSERT INTO participants(id,code_hash) VALUES(?,?)',(participant,sha(code)))
        chosen=[]
        for category,ids in selected.items():
            chosen += [ids[(p*(items_per_person//2)+j)%len(ids)] for j in range(items_per_person//2)]
        rng.shuffle(chosen);order=SYSTEMS.copy();rng.shuffle(order)
        ordinal=0
        for q,item in enumerate(chosen):
            exposure[item]+=1
            shift=q%len(order);names=order[shift:]+order[:shift]
            for slot,name in enumerate(names,1):
                r=groups[item][name];token=secrets.token_hex(16)
                con.execute('INSERT INTO assignments VALUES(?,?,?,?,?,?,?,?,?)',(token,participant,ordinal,item,name,r['question'],r['answer'],r['category'],slot));ordinal+=1
    con.commit();con.close()
    config['item_exposure_counts']=dict(exposure)
    dump(out/'PRIVATE_design.json',config)
    with (out/'PRIVATE_participant_codes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['participant_id','access_code']);w.writeheader();w.writerows(codes)
    (out/'PRIVATE_admin_code.txt').write_text(admin+'\n',encoding='utf-8')
    dump(out/'source_manifest.json',dict(source_file=str(responses.resolve()),sha256=config['source_sha256'],study_id=study_id))
    return config

class Study:
    def __init__(self,folder):self.folder=Path(folder);self.path=self.folder/'ratings.sqlite3'
    @contextmanager
    def connect(self):
        con=sqlite3.connect(self.path,timeout=10);con.row_factory=sqlite3.Row;con.execute('PRAGMA foreign_keys=ON')
        try:
            with con:yield con
        finally:con.close()
    def config(self):
        with self.connect() as c:return json.loads(c.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
    def active(self):
        with self.connect() as c:return c.execute("SELECT value FROM meta WHERE key='active'").fetchone()[0]=='true'
    def login(self,code,admin=False):
        with self.connect() as c:
            if admin:
                expected=c.execute("SELECT value FROM meta WHERE key='admin_hash'").fetchone()[0]
                return 'admin' if secrets.compare_digest(expected,sha(code.strip())) else None
            r=c.execute('SELECT id FROM participants WHERE code_hash=?',(sha(code.strip().upper()),)).fetchone()
            return r[0] if r else None
    def activate(self):
        with self.connect() as c:
            c.execute("UPDATE meta SET value='true' WHERE key='active'");c.execute('INSERT INTO events(at,action) VALUES(?,?)',(now(),'study_activated'))
    def consent(self,participant):
        if not self.active():raise ValueError('研究者尚未啟用正式評分。')
        with self.connect() as c:
            c.execute('UPDATE participants SET consented_at=COALESCE(consented_at,?) WHERE id=?',(now(),participant))
    def state(self,participant):
        with self.connect() as c:
            person=dict(c.execute('SELECT id,consented_at,finished_at FROM participants WHERE id=?',(participant,)).fetchone())
            rows=c.execute('SELECT a.token,a.ordinal,a.question,a.answer,a.slot,r.payload,r.version FROM assignments a LEFT JOIN ratings r ON a.token=r.token WHERE a.participant=? ORDER BY a.ordinal',(participant,)).fetchall()
        # Never send item/source ids, category, system names, references or mapping.
        return dict(participant=person['id'],consented=bool(person['consented_at']),finished=bool(person['finished_at']),active=self.active(),
                    rows=[dict(token=r['token'],ordinal=r['ordinal'],question=r['question'],answer=r['answer'],slot=r['slot'],
                         saved=json.loads(r['payload']) if r['payload'] else None,version=r['version'] or 0) for r in rows])
    @staticmethod
    def validate(data):
        if not isinstance(data,dict):raise ValueError('評分格式不正確。')
        if set(data)-set(AXES+FLAGS+['notes','skip_reason','skipped']):raise ValueError('未知評分欄位。')
        if type(data.get('skipped',False)) is not bool:raise ValueError('略過狀態不正確。')
        if data.get('skipped'):
            reason=data.get('skip_reason','')
            if not isinstance(reason,str) or not 1<=len(reason.strip())<=500:raise ValueError('請填寫略過原因。')
            return dict(skipped=True,skip_reason=reason.strip())
        result=dict(skipped=False)
        for axis in AXES:
            x=data.get(axis)
            if not (type(x) is int and 1<=x<=5) and x!='unable':raise ValueError('每個項目請選1–5分或無法判斷。')
            result[axis]=x
        for flag in FLAGS:
            x=data.get(flag,'unassessed')
            if x not in ['unassessed','yes','no','uncertain']:raise ValueError('風險標記無效。')
            result[flag]=x
        notes=data.get('notes','')
        if not isinstance(notes,str) or len(notes)>2000:raise ValueError('備註限2000字。')
        result['notes']=notes.strip();return result
    def save(self,participant,token,payload,version):
        payload=self.validate(payload)
        if type(version) is not int or version<0:raise ValueError('版本不正確。')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            person=c.execute('SELECT consented_at,finished_at FROM participants WHERE id=?',(participant,)).fetchone()
            if not person or not person['consented_at']:raise ValueError('請先閱讀說明並同意參與。')
            if person['finished_at']:raise ValueError('本份評分已完成送出，不能再修改。')
            if not c.execute('SELECT 1 FROM assignments WHERE participant=? AND token=?',(participant,token)).fetchone():raise PermissionError('題目不屬於此評分者。')
            old=c.execute('SELECT version FROM ratings WHERE token=?',(token,)).fetchone()
            current=old[0] if old else 0
            if current!=version:raise RuntimeError('資料已在另一個分頁更新，請重新整理後再評。')
            text=json.dumps(payload,ensure_ascii=False)
            c.execute('INSERT INTO ratings VALUES(?,?,?,?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload,version=excluded.version,updated_at=excluded.updated_at',(token,text,current+1,now()))
            c.execute('INSERT INTO events(at,participant,token,action,payload) VALUES(?,?,?,?,?)',(now(),participant,token,'save',text))
        return current+1
    def finish(self,participant):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            missing=c.execute('SELECT count(*) FROM assignments a LEFT JOIN ratings r ON a.token=r.token WHERE a.participant=? AND r.token IS NULL',(participant,)).fetchone()[0]
            if missing:raise ValueError(f'仍有 {missing} 則尚未評分或填寫略過原因。')
            c.execute('UPDATE participants SET finished_at=COALESCE(finished_at,?) WHERE id=?',(now(),participant))
            c.execute('INSERT INTO events(at,participant,action) VALUES(?,?,?)',(now(),participant,'finish'))
    def dashboard(self):
        with self.connect() as c:
            rows=[dict(r) for r in c.execute('SELECT p.id,p.consented_at,p.finished_at,count(a.token) AS total,count(r.token) AS saved FROM participants p JOIN assignments a ON p.id=a.participant LEFT JOIN ratings r ON a.token=r.token GROUP BY p.id ORDER BY p.id')]
        return dict(config=self.config(),active=self.active(),participants=rows)
    def export(self):
        with self.connect() as c:
            data=[dict(r) for r in c.execute('SELECT a.*,r.payload,r.version,r.updated_at,p.finished_at FROM assignments a JOIN participants p ON p.id=a.participant LEFT JOIN ratings r ON a.token=r.token ORDER BY a.participant,a.ordinal')]
        mapping_hash=sha(json.dumps([(r['token'],r['participant'],r['item_id'],r['system']) for r in data]))
        ratings=[];long=[]
        for r in data:
            p=json.loads(r['payload']) if r['payload'] else {}
            base=dict(rating_id=r['token'],participant_id=r['participant'],item_id=r['item_id']+'::r0',system=r['system'],
                 category=r['category'],display_position=r['slot'],submitted=bool(r['finished_at']),saved=bool(p),skipped=p.get('skipped',False),
                 version=r['version'],updated_at=r['updated_at'],mapping_verified='true',mapping_sha256=mapping_hash)
            ratings.append(dict(base,**{a:p.get(a,'') for a in AXES+FLAGS+['notes','skip_reason']}))
            if r['finished_at'] and p and not p.get('skipped'):
                for axis in AXES:
                    score=p.get(axis)
                    if type(score) is int:long.append(dict(participant_id=r['participant'],item_id=r['item_id']+'::r0',system=r['system'],axis=axis,score=score,mapping_verified='true',mapping_sha256=mapping_hash))
        return dict(schema_version=1,exported_at=now(),study_id=self.config()['study_id'],config=self.config(),ratings=ratings,completed_scored=long,
                    note='Only finished participants with integer scores appear in completed_scored. Missing/unable/skipped are never zero.')
    def backup(self,path):
        dst=sqlite3.connect(path)
        try:
            with self.connect() as src:src.backup(dst)
        finally:dst.close()

def csv_text(rows,fields=None):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=fields or (list(rows[0]) if rows else ['participant_id','item_id','system','score']))
    w.writeheader()
    for row in rows:
        # Avoid spreadsheet formula execution in free-text fields.
        w.writerow({k:("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v) for k,v in row.items()})
    return '\ufeff'+out.getvalue()
