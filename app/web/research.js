'use strict';
// All provider and feedback text is rendered as text, never provider-supplied HTML.
const researchPanel=el('section',undefined,'activity');
researchPanel.id='research-panel';
const researchHead=el('div',undefined,'panelhead');
researchHead.append(el('h2','Sources & research extensions'));
const researchRefresh=el('button','Refresh source status','secondary');
researchHead.append(researchRefresh);researchPanel.append(researchHead);
const researchContent=el('div');researchContent.style.padding='20px';researchPanel.append(researchContent);
document.querySelector('footer').before(researchPanel);

const loginForm=el('form');loginForm.id='login-form';
loginForm.append(el('h3','Operator sign-in'));
const username=el('input');username.autocomplete='username';username.required=true;username.id='login-username';
const password=el('input');password.type='password';password.autocomplete='current-password';password.required=true;password.id='login-password';
for(const [text,input] of [['Username',username],['Password',password]]){const label=el('label',text);label.append(input);loginForm.append(label);}
const signIn=el('button','Sign in');signIn.type='submit';loginForm.append(signIn);
const authStatus=el('p','Read-only until signed in. Create a local account using manage_orbit.py add-user.','muted');
const signOut=el('button','Sign out','secondary');signOut.type='button';signOut.hidden=true;
$('review-form').before(loginForm,authStatus,signOut);
$('operator').readOnly=true;$('operator').placeholder='Authenticated operator';
async function session(){
 const data=await api('/api/session');identity=data.identity;
 $('operator').value=identity?.username||'';loginForm.hidden=!!identity;signOut.hidden=!identity;
 authStatus.textContent=identity?`Signed in as ${identity.username} · ${identity.role}`:'Read-only. Create a local account using manage_orbit.py add-user, then sign in.';
 syncReviewButton();
}
loginForm.onsubmit=async event=>{event.preventDefault();signIn.disabled=true;try{await api('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:username.value,password:password.value})});password.value='';await session();}catch(e){authStatus.textContent=e.message;}finally{signIn.disabled=false;}};
signOut.onclick=async()=>{try{await api('/api/logout',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await session();}catch(e){authStatus.textContent=e.message;}};
async function loadResearch(){
 researchRefresh.disabled=true;
 try{
  const data=await api('/api/research');researchContent.replaceChildren();
  researchContent.append(el('p','UTC ingestion is separate from the historical replay above. Configured credentials do not establish successful live retrieval.'));
  for(const source of data.sources){const history=data.poll_history.find(p=>p.source===source.source);const item=el('p');item.append(el('strong',source.source.replaceAll('_',' ')+' · '),el('span',source.state.replaceAll('_',' ')));if(history)item.append(el('small',` · Latest attempt: ${history.status} · ${history.successes} successful / ${history.failures} failed`));researchContent.append(item);}
  researchContent.append(el('h3',`${data.decision.rows.length} current UTC encounters · ${data.decision.mandatory_overflow} mandatory overflow`));
  if(!data.decision.rows.length)researchContent.append(el('p','No eligible UTC conjunction messages have been ingested. Historical benchmark records are not displayed as live data.'));
  else{const wrap=el('div',undefined,'tablewrap'),table=el('table');const header=el('tr');for(const name of ['Encounter','Risk','Time remaining','Review','Forecast proxy'])header.append(el('th',name));table.append(header);for(const row of data.decision.rows){const tr=el('tr');for(const value of [row.encounter_key,formatRisk(row.log_risk),row.hours_to_tca.toFixed(1)+'h',row.review_reasons.join(', ')||'Baseline',formatRisk(row.forecast_log_risk_proxy)])tr.append(el('td',value));table.append(tr);}wrap.append(table);researchContent.append(wrap);}
  const details=el('details');details.open=true;details.append(el('summary','Ten research features and their evidence status'));
  for(const feature of data.decision.features){const item=el('div',undefined,'reviewitem');item.append(el('strong',feature.name.replaceAll('_',' ')),badge(feature.status.replaceAll('_',' '),'neutral'),el('p',feature.description));details.append(item);}researchContent.append(details);
  researchContent.append(el('p','Open limitations: '+data.limitations.join(' · '),'muted'));
 }catch(e){researchContent.replaceChildren(el('p','Source status unavailable: '+e.message));}
 finally{researchRefresh.disabled=false;}
}
researchRefresh.onclick=loadResearch;
session().catch(e=>{authStatus.textContent='Sign-in status unavailable: '+e.message;});
loadResearch();
