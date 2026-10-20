"""On-demand short-answer study. No API call until an admin sets a session budget."""
import argparse
import hashlib
import hmac
import json
import math
import os
import threading
import time
from http.cookies import SimpleCookie
from pathlib import Path
from http.server import ThreadingHTTPServer

from .comparison import Comparison, ComparisonHandler, connection, normalize
from .server import Handler
from .study import Study, now
from .short_answer import load_protocol, ShortAnswerBackend, length_metadata
from fgs.api import ChatClient
from fgs.io import digest
from fgs.pipeline import DialoguePipeline, VARIANTS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


class BudgetClient(ChatClient):
    """Persist reservations BEFORE sending; unknown outcomes cannot be retried.

    Accounting intentionally overestimates input ($12.50/M including cache writes).
    Output includes reasoning ($50/M). This is a local guard, not the provider bill.
    """
    def __init__(self, study):
        config = dict(study.settings, api_key_env='FGS_SHORT_SESSION_KEY')
        super().__init__(config, study.folder / 'api_cache', live=True)
        self.study = study
        self.stop_reason = None
        self.cache_dir.mkdir(exist_ok=True)

    def stop(self, message):
        self.stop_reason = message
        raise RuntimeError(message)

    def complete(self, model, messages, nonce):
        payload = dict(model=model, messages=messages, store=False,
                       max_completion_tokens=int(self.config['max_completion_tokens']))
        if self.config.get('reasoning_effort') is not None:
            payload['reasoning_effort'] = self.config['reasoning_effort']
        if self.config.get('temperature') is not None:
            payload['temperature'] = float(self.config['temperature'])
        key = digest(json.dumps(dict(url=self.url, payload=payload, nonce=nonce), sort_keys=True, ensure_ascii=False))
        path = self.cache_dir / (key + '.json')
        if not path.exists():
            with self.study.control:
                if not self.study.enabled or not self.study.api_key:
                    self.stop('API 尚未啟用或已暫停。請研究者設定費用上限。')
                # Byte bound plus generous message overhead; no hidden tools/images.
                reserve = ((len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 2048) * 12.5
                           + int(self.config['max_completion_tokens']) * 50) / 1e6
                with connection(self.study.db) as c:
                    c.execute('BEGIN IMMEDIATE')
                    if c.execute('SELECT 1 FROM api_ledger WHERE cache_key=?', (key,)).fetchone():
                        self.stop('先前呼叫結果未確認或快取遺失；為避免重複收費，已停止。請保留紀錄交研究者檢查。')
                    spent = c.execute('SELECT COALESCE(SUM(charge),0) FROM api_ledger').fetchone()[0]
                    if spent + reserve > self.study.cap:
                        self.stop('剩餘上限不足以預留下一次呼叫。已保留進度；研究者調整上限後可續跑。')
                    c.execute('INSERT INTO api_ledger VALUES (?,?,?,?,?)', (key, reserve, 'reserved', '{}', now()))
                session_key = self.study.api_key
            old = os.environ.get('FGS_SHORT_SESSION_KEY')
            os.environ['FGS_SHORT_SESSION_KEY'] = session_key
            try:
                return super().complete(model, messages, nonce)
            finally:
                if old is None:
                    os.environ.pop('FGS_SHORT_SESSION_KEY', None)
                else:
                    os.environ['FGS_SHORT_SESSION_KEY'] = old
                self.reconcile(key, path)
        self.reconcile(key, path)
        return super().complete(model, messages, nonce)

    def reconcile(self, key, path):
        if not path.exists():
            return
        usage = read(path).get('usage', {})
        if not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
            return  # retain reservation when provider usage is unavailable
        charge = (usage['prompt_tokens'] * 12.5 + usage['completion_tokens'] * 50) / 1e6
        with connection(self.study.db) as c:
            c.execute('UPDATE api_ledger SET charge=?,state=?,usage=? WHERE cache_key=?',
                      (charge, 'reported_usage_conservative', json.dumps(usage), key))


def load_filters(root, cache):
    import joblib
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    filters = {}
    encoder = None
    for kind in ('word2vec', 'tfidf', 'transformer'):
        pair = []
        for role in ('question', 'answer'):
            path = (root / f'runs/supplement_20261013/transformer/random/{role}/seed42/model.joblib'
                    if kind == 'transformer' else root / f'runs/random_80_10_10_20261010/{role}_train/{kind}_seed42/model.joblib')
            model = joblib.load(path)
            if model.kind != kind or model.seed != 42 or model.threshold != .5:
                raise ValueError('Filter settings differ from frozen protocol')
            model.cache_dir = str(cache)
            if kind == 'transformer':
                if encoder is None:
                    encoder = SentenceTransformer(model.embedding_model, revision=model.embedding_revision,
                        device='cpu', local_files_only=True, trust_remote_code=False)
                model.encoder = encoder
            model.accepts('如何以慈悲心與家人相處？')  # offline preflight before any paid request
            pair.append(model)
        filters[kind] = tuple(pair)
    return filters


class LiveComparison(Comparison):
    def __init__(self, folder, bank, protocol, root):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        seed = folder / 'empty_responses.jsonl'
        if not seed.exists():
            seed.write_text('', encoding='utf-8')
        self.protocol_folder = Path(protocol)
        self.protocol, self.prompts = load_protocol(protocol)
        self.settings = read(self.protocol_folder / 'api_settings.json')
        self.root = Path(root)
        super().__init__(folder, bank, seed)
        lock = read(self.protocol_folder / 'lock.json')
        if (folder / 'protocol_lock.json').exists() and read(folder / 'protocol_lock.json') != lock:
            raise ValueError('Protocol changed; use a new study folder')
        atomic(folder / 'protocol_lock.json', lock)
        self.systems = self.protocol['systems']
        self.by_question = {normalize(r['question']): r for r in self.bank}
        self.by_id = {r['id']: r for r in self.bank}
        if len(self.by_question) != 500 or len(self.by_id) != 500:
            raise ValueError('Expected 500 distinct questions')
        with connection(self.db) as c:
            c.execute('CREATE TABLE IF NOT EXISTS api_ledger (cache_key TEXT PRIMARY KEY,charge REAL,state TEXT,usage TEXT,at TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS authorization (at TEXT,cap REAL,enabled INTEGER)')
        self.control = threading.RLock()
        self.enabled = False
        self.api_key = None
        self.cap = 0.
        self.job = dict(state='idle')
        self.filters = None
        self.ready = folder / 'completed'
        self.ready.mkdir(exist_ok=True)
        for path in sorted(self.ready.glob('*.json')):
            self.register(read(path))

    def register(self, rows):
        if len(rows) != len(self.systems) or {r['system'] for r in rows} != set(self.systems):
            raise ValueError('Incomplete configuration set')
        qid = rows[0]['id']
        expected = self.by_id[qid]['question']
        if any(r['id'] != qid or r['question'] != expected or r['status'] == 'provider_or_filter_error' for r in rows):
            raise ValueError('Invalid completed answer set')
        group = []
        for r in rows:
            token = hmac.new(self.secret, (r['id'] + '\0' + r['system']).encode(), hashlib.sha256).hexdigest()
            self.tokens[token] = r
            group.append((token, r))
        self.groups[normalize(expected)] = group

    def configure(self, data):
        with self.control:
            if data.get('enabled') is False:
                self.enabled = False
                self.api_key = None
            else:
                key = data.get('key')
                cap = data.get('cap')
                if data.get('confirmed') is not True or not isinstance(key, str) or not 20 <= len(key.strip()) <= 1000:
                    raise ValueError('請輸入 API Key 並確認啟用付費生成。')
                if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 < cap <= 10000:
                    raise ValueError('請輸入 0 至 10000 美元內的正數累計上限。')
                self.api_key = key.strip()
                self.cap = float(cap)
                self.enabled = True
            with connection(self.db) as c:
                c.execute('INSERT INTO authorization VALUES (?,?,?)', (now(), self.cap, int(self.enabled)))
            return self.status()

    def status(self):
        with self.control, connection(self.db) as c:
            spent = c.execute('SELECT COALESCE(SUM(charge),0) FROM api_ledger').fetchone()[0]
            unresolved = c.execute("SELECT COUNT(*) FROM api_ledger WHERE state='reserved'").fetchone()[0]
            ratings = c.execute('SELECT COUNT(*) FROM ratings').fetchone()[0]
            return dict(enabled=self.enabled, cap_usd=self.cap, conservative_spend_usd=spent,
                        unresolved_reservations=unresolved, completed_questions=len(self.groups),
                        saved_ratings=ratings, job=dict(self.job))

    def start(self, person, question):
        if not isinstance(question, str):
            raise ValueError('請輸入題目。')
        q = self.by_id.get(question.strip()) or self.by_question.get(normalize(question))
        if q is None:
            raise ValueError('請選擇500題題庫中的問題，或貼上完整題目文字。')
        with connection(self.db) as c:
            if not c.execute('SELECT 1 FROM consent WHERE participant=?', (person,)).fetchone():
                raise ValueError('請先同意參與。')
        with self.control:
            if normalize(q['question']) in self.groups:
                return dict(state='complete', question=q['question'])
            if self.job['state'] == 'running':
                if self.job['id'] != q['id']:
                    raise RuntimeError('系統正在處理另一題，請稍後再提交。')
                return dict(self.job)
            if not self.enabled:
                raise RuntimeError('尚未啟用 API。請研究者到管理頁輸入 Key、費用上限並啟用。')
            self.job = dict(state='running', id=q['id'], question=q['question'], progress=0, message='載入本機分類器')
            threading.Thread(target=self.generate, args=(q,), daemon=True).start()
            return dict(self.job)

    def generate(self, q):
        client = None
        try:
            load_protocol(self.protocol_folder)
            for name, expected in self.protocol['protected_original_files'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                    raise ValueError('Protected experiment input changed')
            if self.filters is None:
                self.filters = load_filters(self.root, self.folder / 'tokenizer_cache')
            client = BudgetClient(self)
            domain = ShortAnswerBackend(client, self.prompts, self.protocol['version'])
            generic = ShortAnswerBackend(client, self.prompts, self.protocol['version'], generic=True)
            rows = []
            for system in self.systems:
                if '__' in system:
                    profile, variant = system.split('__')
                    query_filter, answer_filter = self.filters[profile]
                else:
                    variant = 'refinement_only' if system == 'domain_refinement_only' else 'b1'
                    query_filter = answer_filter = None
                pipeline = DialoguePipeline(generic if system == 'generic_b1' else domain,
                    query_filter, answer_filter, max_generation_attempts=1, max_refinements=1)
                row = pipeline.run(q['question'], VARIANTS[variant], sample_id=q['id'])
                if row['status'] == 'provider_or_filter_error':
                    raise RuntimeError('生成或分類失敗。已保留快取與費用紀錄，未發布不完整回答。')
                length = length_metadata(row['answer'])
                low, high = self.protocol.get('target_characters', [20, 80])
                length.update(target_min=low, target_max=high, below_target=length['characters'] < low, above_target=length['characters'] > high)
                row.update(system=system, length=length, protocol_version=self.protocol['version'])
                rows.append(row)
                with self.control:
                    self.job.update(progress=len(rows), message='生成與過濾回答')
            atomic(self.folder / ('events_' + q['id'] + '.json'), client.events)
            atomic(self.ready / (q['id'] + '.json'), rows)
            with self.control:
                self.register(rows)
                self.job.update(state='complete', message=f'{len(self.systems)}種配置已完成')
        except Exception as exc:
            # Never serialize exception text from a provider or arbitrary library.
            atomic(self.folder / ('failure_' + q['id'] + '.json'), dict(at=now(), error_type=type(exc).__name__,
                   events=client.events if client else [], note='No automatic retry; inspect ledger and cache before resuming.'))
            with self.control:
                self.job.update(state='failed', message=(client.stop_reason if client else None) or '本題未完成；進度已保留。請研究者檢查費用上限、API或本機模型。')

    def export(self):
        with self.control:
            data = super().export()
            data.update(design='500_question_short_answer_on_demand_7_systems', protocol=self.protocol,
                        responses=list(self.tokens.values()), runtime=self.status())
            with connection(self.db) as c:
                data['api_ledger'] = [dict(cache_key=k, conservative_usd=v, state=s, usage=json.loads(u), at=t)
                    for k,v,s,u,t in c.execute('SELECT * FROM api_ledger')]
            return data


class LiveHandler(ComparisonHandler):
    # Unique cookie name prevents sessions on other localhost ports interfering.
    def session(self, role=None):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie', ''))
        except Exception:
            return None
        token = cookie.get('short_rating_session')
        s = self.server.sessions.get(token.value if token else '')
        if not s or s['expires'] < time.time() or (role and s['role'] != role):
            return None
        s['expires'] = time.time() + 3600
        return s

    def send(self, *args, **kwargs):
        if kwargs.get('cookie'):
            kwargs['cookie'] = kwargs['cookie'].replace('rating_session=', 'short_rating_session=')
        return super().send(*args, **kwargs)

    def do_GET(self):
        if not self.valid_host():
            return self.send({'error':'Local host required'}, 403)
        path = self.path.split('?', 1)[0]
        if path in ('/', '/admin', '/live.js'):
            file = 'live.js' if path.endswith('.js') else 'live.html'
            return self.send((Path(__file__).parent/'static'/file).read_bytes(),
                mime='text/javascript; charset=utf-8' if file.endswith('.js') else 'text/html; charset=utf-8')
        if path == '/api/live-status':
            s = self.session()
            if not s:
                return self.send({'error':'請先登入。'}, 401)
            data = self.server.comparison.status()
            return self.send(data if s['role'] == 'admin' else dict(job=data['job'], enabled=data['enabled']))
        if path == '/api/comparison-bank':
            s = self.session()
            if not s:
                return self.send({'error':'請先登入。'}, 401)
            comparison = self.server.comparison
            rows = comparison.bank_view()
            if s['role'] == 'rater':
                with connection(comparison.db) as c:
                    counts = {}
                    for (token,) in c.execute('SELECT token FROM ratings WHERE participant=?', (s['participant'],)):
                        qid = comparison.tokens[token]['id']
                        counts[qid] = counts.get(qid, 0) + 1
                for row in rows:
                    row['rated'] = counts.get(row['id'], 0)
            return self.send(dict(questions=rows, systems=len(comparison.systems)))
        if path not in ('/style.css', '/api/comparison-bank', '/api/comparison-export'):
            return self.send({'error':'找不到頁面。'}, 404)
        return super().do_GET()

    def do_POST(self):
        if self.path in ('/api/login', '/api/logout', '/api/comparison-lookup', '/api/comparison-save', '/api/comparison-consent'):
            return super().do_POST()
        if self.path not in ('/api/live-configure', '/api/live-generate'):
            return self.send({'error':'找不到操作。'}, 404)
        if not self.valid_host() or self.headers.get('Origin') not in {
            f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'} or self.headers.get('Content-Type','').split(';')[0] != 'application/json':
            return self.send({'error':'僅接受本機同來源JSON操作。'}, 403)
        role = 'admin' if self.path.endswith('configure') else 'rater'
        s = self.session(role)
        if not s:
            return self.send({'error':'權限不足或登入逾時。'}, 403)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16000:
                raise ValueError('資料大小不正確。')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('資料格式不正確。')
            result = (self.server.comparison.configure(data) if role == 'admin'
                      else self.server.comparison.start(s['participant'], data.get('question')))
            return self.send(result)
        except (ValueError, TypeError) as exc:
            return self.send({'error':str(exc)}, 400)
        except RuntimeError as exc:
            return self.send({'error':str(exc)}, 409)
        except Exception:
            return self.send({'error':'操作失敗，請保留資料並聯絡研究者。'}, 500)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=8770)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    app = ThreadingHTTPServer(('127.0.0.1', a.port), LiveHandler)
    app.daemon_threads = True
    app.study = Study(a.study)
    app.comparison = LiveComparison(a.out, a.bank, a.protocol, root)
    app.sessions = {}
    app.failures = []
    print(f'Short-answer rating: http://127.0.0.1:{a.port}/admin ; live API disabled until admin opt-in', flush=True)
    try:
        app.serve_forever()
    finally:
        app.server_close()


if __name__ == '__main__':
    main()
