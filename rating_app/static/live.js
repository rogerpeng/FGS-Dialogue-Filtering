'use strict';
const $=id=>document.getElementById(id),admin=location.pathname==='/admin';let bank=[],dirty=false,systemCount=7;
const axes=[['appropriateness','整體適切性'],['clarity','表達清楚度'],['helpfulness','有用程度']];
function message(t){$('message').textContent=t;$('message').classList.toggle('hidden',!t);}
async function api(path,data){const r=await fetch(path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'Content-Type':'application/json'},body:data===undefined?undefined:JSON.stringify(data),cache:'no-store'});const x=await r.json();if(!r.ok)throw Error(x.error||'操作失敗');return x;}
const wrap=fn=>async e=>{try{message('');await fn(e);}catch(err){message(err.message);}};
if(admin){$('entry-title').textContent='研究者管理';$('role').href='/';$('role').textContent='返回評分者登入';}
function renderBank(){const term=$('search').value.trim();$('bank').replaceChildren();for(const r of bank.filter(r=>!term||r.question.includes(term)||r.id.includes(term))){const o=document.createElement('option');o.value=r.id;o.textContent=`${r.id} · 回答${r.available}/${systemCount} · 已評${r.rated||0}/${systemCount} · ${r.question}`;$('bank').append(o);}}
$('login-form').onsubmit=wrap(async e=>{e.preventDefault();await api('/api/login',{role:admin?'admin':'rater',code:$('access').value});$('access').value='';const data=await api('/api/comparison-bank');bank=data.questions;systemCount=data.systems;$('entry').classList.add('hidden');$(admin?'management':'workspace').classList.remove('hidden');$('leave').classList.remove('hidden');renderBank();$('bank-status').textContent=`${bank.length}題，${data.systems}種配置；${bank.filter(r=>r.available===data.systems).length}題已有全部回答。`;});
$('search').oninput=renderBank;
$('use-question').onclick=()=>{const q=bank.find(r=>r.id===$('bank').value);if(q)$('question').value=q.question;};
async function requireSaved(){if(document.querySelector('form[data-dirty="true"]'))throw Error('有尚未儲存的評分，請先儲存後再換題或換人。');}
$('ask-form').onsubmit=wrap(async e=>{e.preventDefault();await requireSaved();if(!$('agree').checked)throw Error('請先閱讀評分說明並勾選願意參與。');await api('/api/comparison-consent',{agreed:true});let q=$('question').value.trim().replace(/^\d{1,3}[.．]\s*/,'');const data=await api('/api/comparison-lookup',{question:q});$('answers').replaceChildren();$('result-title').textContent=q;$('coverage').textContent=data.available?`已有 ${data.available} / ${data.total_configurations} 種配置回答。請分別保存每則評分；相同回答仍屬不同配置。`:'這題尚無既有實驗回答，暫時不能進行回答評分。這不代表問題被filter拒絕。';for(const r of data.answers)renderAnswer(r);});
function renderAnswer(r){const form=document.createElement('form');form.className='panel';form.style.marginBottom='20px';const h=document.createElement('h2');h.textContent=`候選回答 ${r.slot}`;const answer=document.createElement('p');answer.style.whiteSpace='pre-wrap';answer.textContent=r.answer;form.append(h,answer);const controls={};for(const [key,title] of axes){const label=document.createElement('label');label.textContent=title;const select=document.createElement('select');select.required=true;select.setAttribute('aria-label',`候選回答 ${r.slot} ${title}`);for(const [v,t] of [['','請選擇'],['1','1 · 最低'],['2','2'],['3','3'],['4','4'],['5','5 · 最高'],['unable','無法判斷']]){const o=document.createElement('option');o.value=v;o.textContent=t;select.append(o);}select.value=r.saved?String(r.saved[key]??''):'';label.append(select);form.append(label);controls[key]=select;}
const label=document.createElement('label');label.textContent='備註（選填）';const notes=document.createElement('textarea');notes.maxLength=2000;notes.rows=2;notes.value=r.saved?.notes||'';label.append(notes);form.append(label);const save=document.createElement('button');save.className='primary';save.textContent='儲存這則評分';save.disabled=!r.rateable;const status=document.createElement('p');status.className='fine';status.textContent=r.saved?'已儲存，可修改':'尚未儲存';form.append(save,status);form.oninput=()=>{form.dataset.dirty='true';dirty=true;status.textContent='尚有未儲存修改';};form.onsubmit=wrap(async e=>{e.preventDefault();save.disabled=true;try{const rating={skipped:false,notes:notes.value};for(const [key] of axes)rating[key]=controls[key].value==='unable'?'unable':Number(controls[key].value);const x=await api('/api/comparison-save',{token:r.token,version:r.version,rating});if(!r.saved){const item=bank.find(x=>x.question===$('result-title').textContent);if(item)item.rated=(item.rated||0)+1;}r.saved=rating;renderBank();r.version=x.version;form.dataset.dirty='false';dirty=!!document.querySelector('form[data-dirty="true"]');status.textContent='已儲存';}finally{save.disabled=false;}});$('answers').append(form);}
$('leave').onclick=wrap(async()=>{await requireSaved();await api('/api/logout',{});dirty=false;location.reload();});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});

let generating=false;
const originalLogin=$('login-form').onsubmit;
$('login-form').onsubmit=async e=>{await originalLogin(e);if(admin&&!$('management').classList.contains('hidden'))await wrap(refreshStatus)();};
async function refreshStatus(){const s=await api('/api/live-status');if(admin)$('live-status').textContent=`${s.enabled?'API已啟用':'API未啟用'} · 回答完成 ${s.completed_questions}/500題 · 已存 ${s.saved_ratings}則評分 · 保守已用及預留 US$${s.conservative_spend_usd.toFixed(4)} / 上限 US$${s.cap_usd.toFixed(2)} · 未確認呼叫 ${s.unresolved_reservations}`;return s;}
$('api-form').onsubmit=wrap(async e=>{e.preventDefault();const key=$('api-key').value;$('api-key').value='';await api('/api/live-configure',{key,cap:Number($('budget').value),confirmed:$('paid-consent').checked,enabled:true});await refreshStatus();message('已啟用逐題生成。請登出後用評分者代碼登入。');});
$('pause-api').onclick=wrap(async()=>{await api('/api/live-configure',{enabled:false});await refreshStatus();message('已暫停新呼叫並清除記憶體中的Key。');});
$('refresh-status').onclick=wrap(refreshStatus);
$('ask-form').onsubmit=wrap(async e=>{
 e.preventDefault();if(generating)return;await requireSaved();if(!$('agree').checked)throw Error('請先閱讀說明並勾選願意參與。');
 await api('/api/comparison-consent',{agreed:true});let q=$('question').value.trim().replace(/^\d{1,3}[.．]\s*/,'');q=bank.find(r=>r.id===q)?.question||q;
 let data=await api('/api/comparison-lookup',{question:q});
 const button=$('ask-form').querySelector('button');generating=true;button.disabled=true;
 try{
  if(!data.available){let job=await api('/api/live-generate',{question:q});q=job.question||q;
   while(job.state==='running'){$('coverage').textContent=`${job.message}（${job.progress||0}/${systemCount}）…已送出的呼叫即使關閉頁面仍會繼續處理本題。`;await new Promise(resolve=>setTimeout(resolve,2000));job=(await api('/api/live-status')).job;}
   if(job.state!=='complete')throw Error(job.message||'生成未完成，請聯絡研究者。');
   data=await api('/api/comparison-lookup',{question:q});
  }
  $('answers').replaceChildren();$('result-title').textContent=q;$('coverage').textContent=`已有 ${data.available}/${data.total_configurations} 種配置短回答。請逐則儲存評分；相同文字仍可能來自不同配置。`;
  for(const r of data.answers)renderAnswer(r);
  bank=(await api('/api/comparison-bank')).questions;renderBank();
 }finally{generating=false;button.disabled=false;}
});
