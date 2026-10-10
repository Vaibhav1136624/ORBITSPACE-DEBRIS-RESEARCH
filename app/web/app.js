'use strict';
const $ = id => document.getElementById(id);
let queue=null, selected=null, filter='all', reviews=[], generation=0, detailGeneration=0, busy=false, pending=null, viewReady=false, identity=null;
function syncReviewButton(){ $('review-button').disabled=!viewReady||busy||!identity||!['operator','admin'].includes(identity.role); }
const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
const badge=(text,type)=>el('span',text,'badge '+type);
async function api(path,options){const response=await fetch(path,options);const data=await response.json();if(!response.ok)throw new Error(data.error||'Request failed');return data;}
function error(message){$('error').textContent=message;$('error').hidden=!message;}
const formatRisk=value=>value===null?'Unavailable':value.toFixed(3);
const priority=r=>r.escalation?badge('Escalation','danger'):r.manual_review_required?badge('Review required','warn'):badge('Baseline','neutral');
function renderQueue(){
 if(!queue)return;
 $('total').textContent=queue.encounters;$('slots').textContent=`${queue.selected} / ${queue.policy.capacity}`;$('overflow').textContent=queue.mandatory_overflow;$('ack-count').textContent=queue.acknowledged;
 const query=$('search').value.trim();const rows=queue.rows.filter(r=>(!query||String(r.event_id).includes(query))&&(filter==='all'||filter==='selected'&&r.selected||filter==='overflow'&&r.mandatory_overflow||filter==='unacknowledged'&&r.review_status==='unacknowledged'));
 $('queue-body').replaceChildren();$('empty').hidden=!!rows.length;
 for(const r of rows){const tr=el('tr');if(r.event_id===selected)tr.className='focused';
  const identity=el('td');identity.append(el('span',String(r.rank).padStart(2,'0'),'rank'));const button=el('button',`#${r.event_id}`);button.setAttribute('aria-label',`Inspect encounter ${r.event_id}`);button.onclick=()=>selectEncounter(r.event_id);identity.append(button);tr.append(identity);
  tr.append(el('td',formatRisk(r.reported_log_risk),'number'));const state=el('td');state.append(priority(r));tr.append(state);
  const slot=el('td');slot.append(r.selected?badge('Assigned','calm'):r.mandatory_overflow?badge('Mandatory overflow','danger'):badge('Outside capacity','neutral'));tr.append(slot);
  tr.append(el('td',r.review_status==='acknowledged'?'Acknowledged':'Unacknowledged'));$('queue-body').append(tr);
 }
 $('queue-caption').textContent=`${rows.length} of ${queue.encounters} visible · ${queue.checkpoint_days*24} hours before approach`;
}
function renderReviews(){
 const actions=reviews.filter(a=>a.event_id===selected&&a.checkpoint_days===queue.checkpoint_days).reverse();$('review-history').replaceChildren();
 if(!actions.length){$('review-history').append(el('p','No actions at this checkpoint.','muted'));return;}
 for(const a of actions){const div=el('div',undefined,'reviewitem');div.append(el('strong',`${a.action==='acknowledge'?'Acknowledged':'Reopened'} · ${a.operator}`),el('p',a.note||'No note'),el('small',new Date(a.created_utc).toLocaleString()));$('review-history').append(div);}
}
function drawHistory(reports){
 const svg=$('chart');svg.replaceChildren();const valid=reports.filter(r=>r.risk!==null);if(!valid.length){$('chart-caption').textContent='No valid reported risk available.';return;}
 const ns='http://www.w3.org/2000/svg';const node=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;svg.append(e);return e;};
 const times=valid.map(r=>r.time_to_tca*24), values=valid.map(r=>r.risk);const minX=Math.min(...times),maxX=Math.max(...times),minY=Math.min(...values,-6)-.5,maxY=Math.max(...values,-6)+.5;
 const x=t=>maxX===minX?190:45+(maxX-t)/(maxX-minX)*295;const y=r=>108-(r-minY)/(maxY-minY)*88;
 node('line',{x1:45,x2:340,y1:y(-6),y2:y(-6),stroke:'#c09369','stroke-dasharray':'4 4'});
 node('text',{x:8,y:y(-6)+3,fill:'#8d541c','font-size':10},'−6');
 node('polyline',{points:valid.map(r=>`${x(r.time_to_tca*24)},${y(r.risk)}`).join(' '),fill:'none',stroke:'#176655','stroke-width':2});
 for(const r of valid)node('circle',{cx:x(r.time_to_tca*24),cy:y(r.risk),r:3,fill:'#176655'});
 node('text',{x:45,y:130,fill:'#627476','font-size':10},`${maxX.toFixed(1)}h`);node('text',{x:340,y:130,fill:'#627476','font-size':10,'text-anchor':'end'},`${minX.toFixed(1)}h`);
 $('chart-caption').textContent=`${reports.length} available reports · log₁₀ risk ${Math.min(...values).toFixed(2)} to ${Math.max(...values).toFixed(2)} · Dashed line: escalation threshold`;
}
async function selectEncounter(id){
 if(busy)return;selected=id;renderQueue();const token=++detailGeneration;const r=queue.rows.find(row=>row.event_id===id);if(!r){$('detail-content').hidden=true;$('detail-title').textContent='No encounter available';return;}
 $('detail-title').textContent=`Encounter #${id}`;$('detail-content').hidden=false;$('detail-badges').replaceChildren(priority(r),badge(r.review_status==='acknowledged'?'Acknowledged':'Unacknowledged','neutral'));
 $('detail-summary').textContent=r.reasons.length?`Review required: ${r.reasons.map(x=>x.replaceAll('_',' ')).join(', ')}.`:'Reported risk is available for baseline ranking.';
 $('detail-risk').textContent=formatRisk(r.reported_log_risk);$('detail-age').textContent=`${(r.report_age_days*24).toFixed(1)}h`;
 $('review-button').textContent=r.review_status==='acknowledged'?'Reopen review':'Acknowledge encounter';$('action-status').textContent='';$('note').value='';
 $('provenance').replaceChildren();for(const [key,value] of [['Policy',queue.policy.version],['Latest report hash',r.latest_report_hash],['History hash',r.history_hash]])$('provenance').append(el('dt',key),el('dd',value));
 renderReviews();$('chart').replaceChildren();$('chart-caption').textContent='Loading history…';
 try{const data=await api(`/api/history/${queue.checkpoint_days}/${id}`);if(token===detailGeneration)drawHistory(data.reports);}catch(e){if(token===detailGeneration)$('chart-caption').textContent=`History unavailable: ${e.message}`;}
}
function renderNotifications(data){const visible=data.notifications.filter(n=>n.checkpoint_days>=queue.checkpoint_days);$('notice-count').textContent=`${visible.length} transitions through this checkpoint`;$('notifications').replaceChildren();for(const n of visible){const d=el('div',undefined,'event');d.append(el('strong',`#${n.event_id} · ${n.kind.replaceAll('_',' ')}`),el('small',`${n.checkpoint_days*24}h before approach · ${n.prior_state.replaceAll('_',' ')} → ${n.state.replaceAll('_',' ')}`));$('notifications').append(d);}if(!visible.length)$('notifications').append(el('p','No transitions yet.','event'));}
async function load(checkpoint){
 const token=++generation;let loaded=false;viewReady=false;++detailGeneration;error('');$('refresh').disabled=true;$('checkpoint').disabled=true;$('review-button').disabled=true;$('connection').textContent='Loading checkpoint…';
 try{const [next,actions,notices]=await Promise.all([api(`/api/queue/${checkpoint}`),api('/api/reviews'),api('/api/notifications')]);if(token!==generation)return;queue=next;reviews=actions.reviews;renderQueue();renderNotifications(notices);await selectEncounter(queue.rows.some(r=>r.event_id===selected)?selected:queue.rows[0]?.event_id);$('connection').textContent='Connected · persistent local replay';loaded=true;}
 catch(e){error(e.message);$('connection').textContent='Refresh failed · previous view may be stale';}
 finally{if(token===generation){$('refresh').disabled=false;$('checkpoint').disabled=false;viewReady=loaded&&!!queue?.rows.length;syncReviewButton();}}
}
$('review-form').onsubmit=async event=>{
 event.preventDefault();if(busy||!queue||selected===null)return;const row=queue.rows.find(r=>r.event_id===selected);const request={checkpoint_days:queue.checkpoint_days,event_id:selected,action:row.review_status==='acknowledged'?'reopen':'acknowledge',operator:$('operator').value.trim(),note:$('note').value};if(!request.operator){$('operator').focus();return;}
 const signature=JSON.stringify(request);if(!pending||pending.signature!==signature)pending={signature,request:{...request,action_id:crypto.randomUUID()}};
 busy=true;$('review-button').disabled=true;$('checkpoint').disabled=true;$('refresh').disabled=true;error('');$('action-status').textContent='Saving review action…';
 try{await api('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(pending.request)});pending=null;busy=false;await load(String(queue.checkpoint_days));$('action-status').textContent='Review action saved. Risk and queue assignment are unchanged.';}
 catch(e){error(`Could not confirm save: ${e.message}. Retry uses the same request ID.`);$('action-status').textContent='Save not confirmed.';}
 finally{busy=false;syncReviewButton();$('checkpoint').disabled=false;$('refresh').disabled=false;}
};
$('search').oninput=renderQueue;document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===button));renderQueue();});
$('checkpoint').onchange=()=>load($('checkpoint').value);$('refresh').onclick=()=>load($('checkpoint').value);
async function init(){try{const data=await api('/api/snapshots');$('checkpoint').replaceChildren();for(const day of data.snapshots){const option=el('option',`${day*24} hours before approach`);option.value=day;$('checkpoint').append(option);}if(!data.snapshots.length)throw new Error('No replay snapshots. Run the replay demo first.');$('checkpoint').value=Math.min(...data.snapshots);await load($('checkpoint').value);}catch(e){error(e.message);$('connection').textContent='Unable to load replay';}}
init();
