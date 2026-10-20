'use strict';
const $=id=>document.getElementById(id),admin=location.pathname==='/admin';let bank=[],dirty=false,systemCount=7,savingAll=false;let answerEditors=[];let consentRecorded=false;
const axes=[['appropriateness','整體適切性'],['clarity','表達清楚度'],['helpfulness','有用程度']];
function message(t){$('message').textContent=t;$('message').classList.toggle('hidden',!t);}
async function api(path,data){const r=await fetch(path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'Content-Type':'application/json'},body:data===undefined?undefined:JSON.stringify(data),cache:'no-store'});const x=await r.json();if(!r.ok)throw Error(x.error||'操作失敗');return x;}
const wrap=fn=>async e=>{try{message('');await fn(e);}catch(err){message(err.message);}};
if(admin){$('entry-title').textContent='研究者管理';$('role').href='/';$('role').textContent='返回評分者登入';}
$('login-form').onsubmit=wrap(async e=>{e.preventDefault();await api('/api/login',{role:admin?'admin':'rater',code:$('access').value});$('access').value='';const data=await api('/api/comparison-bank');bank=data.questions;systemCount=data.systems;consentRecorded=!!data.consented;$('agree').checked=consentRecorded;$('entry').classList.add('hidden');$(admin?'management':'workspace').classList.remove('hidden');$('leave').classList.remove('hidden');$('bank-status').textContent=`${bank.length}題，${data.systems}種配置；${bank.filter(r=>r.available===data.systems).length}題已有全部回答。`;});
async function requireSaved(){if(savingAll)throw Error('正在儲存，請稍候。');if(document.querySelector('form[data-dirty="true"]'))throw Error('有尚未儲存的評分，請先儲存後再換題或換人。');}
function renderAnswer(r){const form=document.createElement('form');form.className='panel';form.style.marginBottom='20px';const h=document.createElement('h2');h.textContent=`候選回答 ${r.slot}`;const answer=document.createElement('p');answer.style.whiteSpace='pre-wrap';answer.textContent=r.answer;form.append(h,answer);const controls={};for(const [key,title] of axes){
 const group=document.createElement('fieldset');
 const legend=document.createElement('legend');legend.textContent=title;group.append(legend);
 const scale=document.createElement('div');scale.className='scale';group.append(scale);
 for(const [value,text] of [['1','1 · 最低'],['2','2'],['3','3'],['4','4'],['5','5 · 最高'],['unable','無法判斷']]){
  const label=document.createElement('label');
  const input=document.createElement('input');input.type='radio';input.name=`score-${r.token}-${key}`;
  input.value=value;input.required=true;
  input.setAttribute('aria-label',`候選回答 ${r.slot} ${title} ${text}`);
  input.checked=!!r.saved&&String(r.saved[key])===value;
  const caption=document.createElement('span');caption.textContent=text;
  label.append(input,caption);
  if(value==='unable'){label.className='unable';group.append(label);}else scale.append(label);
 }
 form.append(group);controls[key]=group;
}

const label=document.createElement('label');label.textContent='備註（選填）';const notes=document.createElement('textarea');notes.maxLength=2000;notes.rows=2;notes.value=r.saved?.notes||'';label.append(notes);form.append(label);const status=document.createElement('p');status.className='fine';
status.textContent=r.saved?'已儲存，可修改':'尚未儲存';form.append(status);
form.oninput=()=>{form.dataset.dirty='true';dirty=true;status.textContent='尚有未儲存修改';const hint=$('save-all-status');if(hint)hint.textContent='完成本題7組評分後，按下方按鈕一次儲存。';};
form.onsubmit=e=>e.preventDefault();
answerEditors.push({r,form,controls,notes,status});$('answers').append(form);
}

function renderSaveAll(){
 const panel=document.createElement('section');panel.className='panel';
 const button=document.createElement('button');button.id='save-all';button.type='button';button.className='primary wide';button.textContent='儲存本題全部7組評分';
 const hint=document.createElement('p');hint.id='save-all-status';hint.className='fine';hint.setAttribute('role','status');hint.textContent='請完成每組的三項評分；無法判斷可選「無法判斷」。';
 panel.append(button,hint);$('answers').append(panel);
 button.onclick=async()=>{
  if(savingAll)return;
  message('');
  const pending=[];
  for(const editor of answerEditors){
   const rating={skipped:false,notes:editor.notes.value};
   for(const [key,title] of axes){
    const selected=editor.controls[key].querySelector('input:checked');
    if(!selected){hint.textContent=`候選回答 ${editor.r.slot} 的「${title}」尚未填寫，尚未開始儲存。`;editor.controls[key].scrollIntoView({behavior:'smooth',block:'center'});editor.controls[key].querySelector('input').focus({preventScroll:true});return;}
    rating[key]=selected.value==='unable'?'unable':Number(selected.value);
   }
   if(!editor.form.reportValidity())return;
   if(!editor.r.saved||editor.form.dataset.dirty==='true')pending.push({editor,rating});
  }
  if(!pending.length){hint.textContent='本題7組評分皆已儲存，沒有未存修改。';return;}
  savingAll=true;button.disabled=true;
  const inputs=[...$('answers').querySelectorAll('input,textarea')];inputs.forEach(input=>input.disabled=true);
  let saved=0;
  try{
   for(const {editor,rating} of pending){
    hint.textContent=`儲存中 ${saved}/${pending.length}，請勿關閉頁面…`;
    const result=await api('/api/comparison-save',{token:editor.r.token,version:editor.r.version,rating});
    if(!editor.r.saved){const item=bank.find(q=>q.question===$('result-title').textContent);if(item)item.rated=(item.rated||0)+1;}
    editor.r.saved=rating;editor.r.version=result.version;editor.form.dataset.dirty='false';editor.status.textContent='已儲存';saved++;
   }
   hint.textContent='本題7組評分已全部儲存，可以換下一題。';
  }catch(error){
   hint.textContent=`本次已儲存 ${saved}/${pending.length} 組，尚有評分未存：${error.message}。請保留頁面；已成功的分數不會遺失。`;
  }finally{
   dirty=answerEditors.some(e=>e.form.dataset.dirty==='true'||!e.r.saved);
   inputs.forEach(input=>input.disabled=false);savingAll=false;button.disabled=false;
  }
 };
}
$('leave').onclick=wrap(async()=>{await requireSaved();await api('/api/logout',{});dirty=false;location.reload();});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});

let generating=false;
const originalLogin=$('login-form').onsubmit;
$('login-form').onsubmit=async e=>{await originalLogin(e);if(admin&&!$('management').classList.contains('hidden'))await wrap(refreshStatus)();};
async function refreshStatus(){const s=await api('/api/live-status');if(admin)$('live-status').textContent=`只評分模式（不需API Key） · 回答完成 ${s.completed_questions}/308題 · 已存 ${s.saved_ratings}則評分 · 保守已用及預留 US$${s.conservative_spend_usd.toFixed(4)} / 上限 US$${s.cap_usd.toFixed(2)} · 未確認呼叫 ${s.unresolved_reservations}`;return s;}
$('api-form').onsubmit=wrap(async e=>{e.preventDefault();const key=$('api-key').value;$('api-key').value='';await api('/api/live-configure',{key,cap:Number($('budget').value),confirmed:$('paid-consent').checked,enabled:true});await refreshStatus();message('已啟用逐題生成。請登出後用評分者代碼登入。');});
$('pause-api').onclick=wrap(async()=>{await api('/api/live-configure',{enabled:false});await refreshStatus();message('已暫停新呼叫並清除記憶體中的Key。');});
$('refresh-status').onclick=wrap(refreshStatus);
$('ask-form').onsubmit=wrap(async e=>{
 e.preventDefault();if(generating)return;await requireSaved();if(!$('agree').checked)throw Error('請先閱讀說明並勾選願意參與。');
 if(!consentRecorded){await api('/api/comparison-consent',{agreed:true});consentRecorded=true;}let q=$('question').value.trim().replace(/^\d{1,3}[.．]\s*/,'');q=bank.find(r=>r.id===q)?.question||q;
 let data=await api('/api/comparison-lookup',{question:q});
 const button=$('ask-form').querySelector('button');generating=true;button.disabled=true;
 try{
  if(!data.available)throw Error('目前沒有這個問題的已保存回答，請確認問題文字或聯絡研究者。');
  $('answers').replaceChildren();answerEditors=[];$('result-title').textContent=q;$('coverage').textContent=`已有 ${data.available}/${data.total_configurations} 種配置短回答。填完後到頁面最下面一次儲存7組評分；相同文字仍可能來自不同配置。`;
  for(const r of data.answers)renderAnswer(r);if(data.answers.length)renderSaveAll();
  bank=(await api('/api/comparison-bank')).questions;
 }finally{generating=false;button.disabled=false;}
});
