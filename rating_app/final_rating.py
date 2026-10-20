"""Frozen, rating-only subset. No generation or API activation endpoints."""
import hashlib
import sqlite3
from pathlib import Path
from http.server import ThreadingHTTPServer
from .live_comparison import LiveComparison, LiveHandler, read
from .comparison import normalize
from .study import Study

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/short_seven_rating_20261014'

class FinalComparison(LiveComparison):
    def __init__(self):
        super().__init__(OUT,ROOT.parent/'星雲大師相關提問_500題_20261014/questions.json',ROOT/'data/final_short_seven_20261014',ROOT)
        self.scope=read(OUT/'final_scope_308.json')
        for name,expected in self.scope['answer_sha256'].items():
            if Path(name).name!=name or hashlib.sha256((self.ready/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Frozen answer changed: '+name)
        ids=set(self.scope['question_ids'])
        self.bank=[q for q in self.bank if q['id'] in ids]
        if len(self.bank)!=308 or any(len(self.groups.get(normalize(q['question']),[]))!=7 for q in self.bank):
            raise ValueError('Expected 308 complete seven-system questions')
        self.by_question={normalize(q['question']):q for q in self.bank}
        self.by_id={q['id']:q for q in self.bank}
        self.groups={k:v for k,v in self.groups.items() if k in self.by_question}
        self.tokens={t:r for t,r in self.tokens.items() if r['id'] in ids}

    def configure(self,data):
        raise RuntimeError('本研究已固定308題，僅供評分，不再啟用API。')

    def start(self,person,question):
        raise RuntimeError('本研究已固定308題，請從題庫選題，不再生成新回答。')

    def export(self):
        data=super().export()
        data.update(design='final_308_questions_seven_systems_rating_only',final_scope=self.scope)
        return data

class FinalHandler(LiveHandler):
    def do_GET(self):
        if not self.valid_host():return self.send({'error':'Local host required'},403)
        path=self.path.split('?',1)[0]
        if path == '/api/comparison-bank':
            session = self.session()
            if not session:return self.send({'error':'請先登入。'},401)
            comparison = self.server.comparison
            rows = comparison.bank_view()
            with sqlite3.connect(comparison.db) as con:
                consented = bool(con.execute('SELECT 1 FROM consent WHERE participant=?',(session['participant'],)).fetchone())
            return self.send(dict(questions=rows,systems=len(comparison.systems),consented=consented))
        if path in ('/','/admin','/live.js'):
            file='final308.js' if path.endswith('.js') else 'final308.html'
            return self.send((ROOT/'rating_app/static'/file).read_bytes(),mime='text/javascript; charset=utf-8' if file.endswith('.js') else 'text/html; charset=utf-8')
        return super().do_GET()

def main():
    app=ThreadingHTTPServer(('127.0.0.1',8770),FinalHandler)
    app.daemon_threads=True
    app.study=Study(ROOT/'runs/human_rating_20261014')
    app.comparison=FinalComparison()
    app.sessions={};app.failures=[]
    print('Rating only: http://127.0.0.1:8770/ ; 308 questions, 7 systems; API disabled',flush=True)
    try:app.serve_forever()
    finally:app.server_close()

if __name__=='__main__':main()
