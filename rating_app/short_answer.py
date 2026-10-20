"""Versioned short-answer backend; prompts supplied from a separately locked plan.

Uses the existing client's cache and transport, without modifying frozen fgs code.
No live calls are enabled by this module. A budget-approved runner must supply client.
"""
import json
from pathlib import Path
import hashlib

def load_protocol(folder):
    folder=Path(folder)
    locks=json.loads((folder/'lock.json').read_text(encoding='utf-8'))
    for name,expected in locks.items():
        if Path(name).name!=name:raise ValueError('Invalid manifest filename')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:raise ValueError('Short-answer protocol changed: '+name)
    p=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    prompts=json.loads((folder/'prompts.json').read_text(encoding='utf-8'))
    return p,prompts

def length_metadata(answer):
    n=sum(not c.isspace() for c in answer)
    return dict(characters=n,target_min=20,target_max=80,below_target=n<20,above_target=n>80,modified=False)

class ShortAnswerBackend:
    def __init__(self,client,prompts,version,generic=False):
        self.client=client;self.prompts=prompts;self.version=version;self.generic=generic
    def generate(self,question,sample_id,replicate,attempt):
        role='generic_generation' if self.generic else 'domain_generation'
        return self.client.complete('gpt-6-astra',[{'role':'system','content':self.prompts[role]},{'role':'user','content':question}],[self.version,role,sample_id,replicate,attempt])
    def refine(self,question,answer,sample_id,replicate,attempt):
        return self.client.complete('gpt-6-astra',[{'role':'system','content':self.prompts['refinement']},{'role':'user','content':json.dumps({'question':question,'draft':answer},ensure_ascii=False)}],[self.version,'domain_refinement',sample_id,replicate,attempt])
