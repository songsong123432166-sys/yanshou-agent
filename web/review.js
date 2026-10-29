'use strict';
const reviewUI = new Map();
const reviewNames = {};
function reviewState(id) {
  if (!reviewUI.has(id)) reviewUI.set(id,{open:false,attachmentsOpen:false,data:null,busy:false,error:'',feedback:null,saving:null,recent:new Set(),drafts:new Map(),page:0,filter:'pending'});
  return reviewUI.get(id);
}
async function reviewRequest(job,path,body,rawOptions) {
  const view=reviewState(job.id);
  if(view.busy)return;
  view.busy=true;view.error='';view.feedback=null;
  view.saving=body?.issue_id||body?.field_id ? body : null;
  renderJobs();
  try {
    const options=rawOptions || (body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
    const result=await api(`/api/jobs/${job.id}/review${path}`,options);
    if(result.review) view.data=result;
    if(result.projection)Object.assign(reviewNames,result.projection.decision_labels);
    if(body?.issue_id){
      const issue=result.review.issues.find(item=>item.issue_id===body.issue_id);
      view.recent.add(body.issue_id);
      const text=path==='/propose'?'AI 改稿已生成，尚未执行。请核对修改前后，再点“确认按建议改”。':`已保存：题目 ${issue.sample_id||'ID 缺失'} · ${reviewNames[issue.decision]}。${body.decision==='approve'?'该修改将在导出审核后版本时应用。':body.decision==='keep_highlight'?'仅保留当前问题，其余修改照常；导出审核后版本时 Caption 单元格标黄。':''}`;
      view.feedback={issue_id:body.issue_id,kind:'success',text};
    }else if(body?.field_id){
      for(const issue of result.review.issues.filter(item=>item.field_id===body.field_id))view.recent.add(issue.issue_id);
      view.feedback={field_id:body.field_id,kind:'success',text:body.reset?'已撤销本字段人工稿。':'本字段人工最终稿已保存；导出审核后版本时应用。'};
    }else if(path.startsWith('/import')){
      view.feedback={kind:'success',text:'审核表已导入，审核结论已保存。'};
    }
    if(path==='/apply'){
      job.review_output=result;
      const text=`审核后版本已导出：${result.changed_captions} 条 Caption${result.highlighted_captions?`；${result.highlighted_path?'标黄':'待标黄'} ${result.highlighted_captions} 条`:''}；剩余 ${result.remaining_review_items} 项待处理。${(result.export_warnings||[]).join(' ')||result.highlight_warning||''}`;
      message(text);view.feedback={kind:'success',text};
    }
    if(path==='/export') message('审核表已导出，可编辑审核结论和人工改稿后回传。');
    await refreshJobs();
  } catch(error) {
    view.error=path==='/apply'?`审核后版本未导出：${error.message} 已保存的审核结论仍保留。`:`保存或操作未完成：${error.message}`;
    view.feedback={issue_id:body?.issue_id,field_id:body?.field_id,kind:'error',text:view.error};
  }
  finally {view.busy=false;view.saving=null;renderJobs();}
}
function sourceExcerpt(text,snippet,ranges) {
  const pre=element('pre','review-source');
  if(Array.isArray(ranges)){
    let cursor=0;
    for(const [a,b] of ranges){
      pre.append(document.createTextNode(text.slice(cursor,a)),element('mark','review-error-point',text.slice(a,b)));cursor=b;
    }
    pre.append(document.createTextNode(text.slice(cursor)));return pre;
  }
  const target=snippet?.trim();
  if(!target || !text.includes(target)) {pre.textContent=text;return pre;}
  const parts=text.split(target);
  parts.forEach((part,index)=>{
    pre.append(document.createTextNode(part));
    if(index<parts.length-1) pre.append(element('mark','review-error-point',target));
  });
  return pre;
}
function renderReviewPanel(job) {
  const view=reviewState(job.id),panel=element('section','review-panel');
  const open=element('button','secondary-button',view.open?'收起审核':'审核与返修');
  open.disabled=view.busy||job.repair_exporting;
  open.onclick=()=>{
    view.open=!view.open;
    if(view.open && !view.data) reviewRequest(job,''); else renderJobs();
  };
  panel.append(open);
  if(!view.open) return panel;
  panel.append(element('p','partial-note',job.workflow_version?.startsWith('safe-auto-review-')?'基础标点、符号宽度、既有编号标签格式及词表错字默认自动修正；其余问题逐项审核。人工定稿按完整字段保存，原文件保留。':'识别只提出疑点。按错误项审核；人工定稿按完整字段保存。仅已确认的修改会回填到新副本，原文件保留。'));
  if(view.error){const alert=element('div','job-error',view.error);alert.setAttribute('role','alert');panel.append(alert);}
  const feedback=element('div',`review-feedback${view.busy?' is-saving':''}`);
  feedback.setAttribute('role','status');feedback.setAttribute('aria-live','polite');
  feedback.textContent=view.busy?(view.saving?.issue_id&&!view.saving.decision?'AI 正在按建议生成改稿…':view.saving?'正在保存审核结论…':'正在读取或处理，请稍候…'):view.feedback?.kind==='success'?view.feedback.text:'';
  panel.append(feedback);
  if(!view.data) {panel.append(element('p','partial-note',view.busy?'正在读取审核记录…':'尚未读取审核记录。'));return panel;}
  const documentData=view.data.review,summary=view.data.summary;
  const projection=view.data.projection;
  if(!projection||projection.version!==1||projection.review_revision!==documentData.revision){
    const warning=element('div','job-error','审核预览未就绪，请刷新页面后重试。');warning.setAttribute('role','alert');panel.append(warning);return panel;
  }
  Object.assign(reviewNames,projection.decision_labels);
  const automaticIds=new Set(projection.automatic_issue_ids);
  const visibleIds=new Set(projection.visible_issue_ids);
  const humanIssues=documentData.issues.filter(i=>visibleIds.has(i.issue_id));
  const blockedFields=new Set(projection.blockers.map(item=>item.field_id));
  if(blockedFields.size){
    const warning=element('div','job-error',`${blockedFields.size} 个字段的改稿暂不能安全应用，已保存的审核结论仍保留。`);
    warning.setAttribute('role','alert');const locate=element('button','secondary-button','定位无法应用的字段');
    locate.onclick=()=>{view.filter='blocked';view.page=0;view.recent.clear();renderJobs();};warning.append(locate);panel.append(warning);
  }
  panel.append(element('strong','',`共 ${humanIssues.length} 项人工处理 · 待处理 ${summary.pending} 项 · 人工定稿 ${summary.manual_fields} 个字段`));
  if(summary.highlighted_captions)panel.append(element('p','review-highlight-note',`已选择 ${summary.highlighted_captions} 条 Caption 标黄；仅保留对应问题，其余修改照常。导出后生效。`));
  const toolbar=element('div','review-toolbar');
  const filter=document.createElement('select');
  filter.setAttribute('aria-label','筛选审核结论');
  for(const [value,label] of [['pending','待处理'],['confirm','人工确认改正'],['rewrite','人工判断 / 改写'],['all','全部人工项'],['approved','已确认修改'],['rejected','误报 / 保留'],['blocked','无法应用的字段']]) {
    const option=element('option','',label);option.value=value;filter.append(option);
  }
  filter.value=view.filter;filter.onchange=()=>{view.filter=filter.value;view.page=0;view.recent.clear();renderJobs();};
  toolbar.append(filter);
  const exportButton=element('button','secondary-button','导出可回传审核表');
  exportButton.onclick=()=>reviewRequest(job,'/export',{});
  const input=document.createElement('input');input.type='file';input.accept='.csv,.xlsx';input.hidden=true;
  input.setAttribute('aria-label','上传审核完成的表格');
  input.onchange=()=>{
    const file=input.files[0];if(!file)return;
    reviewRequest(job,`/import?name=${encodeURIComponent(file.name)}&revision=${documentData.revision}`,null,{method:'POST',body:file,headers:{'Content-Type':'application/octet-stream'}});
  };
  const importButton=element('button','secondary-button','回传已审核表格');importButton.onclick=()=>input.click();
  const apply=element('button','primary-button review-apply','导出审核后版本');
  apply.onclick=()=>reviewRequest(job,'/apply',{revision:documentData.revision,projection_fingerprint:projection.fingerprint});
  for(const button of [exportButton,importButton,apply])button.disabled=view.busy||job.repair_exporting;
  apply.disabled=apply.disabled||!projection.export_ready;
  toolbar.append(apply,exportButton,importButton,input);panel.append(toolbar);
  panel.append(element('p','partial-note','上传原文件作为未审核版本保留。导出审核后版本会合并自动修正、已确认修改和人工稿；保留或待处理项按原文输出。没有文字改动也能导出。'));
  const byId=new Map(documentData.issues.map(issue=>[issue.issue_id,issue]));
  const accepted=issue=>!automaticIds.has(issue.issue_id)&&(view.filter==='all'||
    (view.filter==='blocked'&&blockedFields.has(issue.field_id))||(view.filter==='confirm'&&issue.operations.length>0)||(view.filter==='rewrite'&&!issue.operations.length)||
    (view.filter==='pending'&&(['pending','defer'].includes(issue.decision)||view.recent.has(issue.issue_id)))||
    (view.filter==='approved'&&['approve','manual'].includes(issue.decision))||(view.filter==='rejected'&&['reject','keep','keep_highlight'].includes(issue.decision)));
  if(view.filter==='pending'&&view.recent.size)panel.append(element('p','partial-note','本次刚审核的项目暂留原位，便于核对；切换筛选后仅显示对应项目。'));
  const fields=documentData.fields.filter(field=>field.issue_ids.some(id=>accepted(byId.get(id)))||(['all','blocked'].includes(view.filter)&&blockedFields.has(field.field_id)));
  const pages=Math.max(1,Math.ceil(fields.length/10));view.page=Math.min(view.page,pages-1);
  const paging=element('div','review-toolbar');
  for(const [label,delta] of [['上一页',-1],['下一页',1]]){
    const button=element('button','secondary-button',label);button.disabled=view.busy||view.page+delta<0||view.page+delta>=pages;
    button.onclick=()=>{view.page+=delta;renderJobs();};paging.append(button);
  }
  paging.append(element('span','',`${view.page+1} / ${pages} 页 · ${fields.length} 个字段`));panel.append(paging);
  if(!fields.length)panel.append(element('p','partial-note','当前筛选下没有待审核项目。'));
  const yellowRows=new Set(humanIssues.filter(i=>i.decision==='keep_highlight').map(i=>i.row_index));
  for(const field of fields.slice(view.page*10,view.page*10+10)){
    const resolution=projection.fields[field.field_id];
    const group=element('article',`review-field${yellowRows.has(field.row_index)?' is-highlighted-caption':''}`);
    group.append(element('h3','',`题目 ID：${field.sample_id||'原表缺失，须先补齐'} · ${field.name}`));
    if(resolution.error){const warning=element('div','job-error',`此字段暂不能应用：${resolution.error}`);warning.setAttribute('role','alert');group.append(warning);}
    const issues=field.issue_ids.map(id=>byId.get(id));
    for(const issue of issues.filter(accepted)){
      const item=element('div',`review-issue review-status-${issue.decision}`);
      item.dataset.issueId=issue.issue_id;
      item.append(element('strong','',`【${issue.label}】${reviewNames[issue.decision]}`),element('p','',issue.reason));
      item.append(element('p','partial-note',issue.operations.length?'处理方式：人工确认改正':'处理方式：可先让 AI 按建议生成改稿，再人工确认'));
      issue.excerpts.forEach((excerpt,index)=>item.append(sourceExcerpt(excerpt,issue.snippet,issue.excerpt_highlights?.[index])));
      const buttons=element('div','review-toolbar');
      for(const decision of ['approve','reject','keep','keep_highlight','defer','pending']){
        const label=decision==='pending'?'重置':reviewNames[decision];
        const button=element('button','secondary-button',label);
        const selected=issue.decision===decision&&decision!=='pending';
        button.classList.add('review-decision');button.dataset.decision=decision;
        button.setAttribute('aria-pressed',String(selected));
        if(selected){button.classList.add('is-selected');button.textContent=`✓ ${label}`;}
        if(view.busy&&view.saving?.issue_id===issue.issue_id&&view.saving.decision===decision)button.textContent='正在保存…';
        button.disabled=view.busy||Object.hasOwn(documentData.field_edits,field.field_id)||(decision==='approve'&&(!issue.operations.length||!field.sample_id));
        if(decision==='approve'&&!issue.operations.length){button.textContent='待生成改稿';button.title='先点“让 AI 按建议生成改稿”，核对修改前后即可确认。也可人工填写。';}
        else if(decision==='approve'&&!field.sample_id){button.textContent='缺少题目 ID';button.title='原表缺少真实题目 ID，暂不能确认修改。';}
        if(decision==='keep_highlight'){button.title='仅保留当前问题，其余修改照常；导出审核后版本时整条 Caption 单元格标黄。CSV 另生成标黄 Excel。';if(!field.sample_id)button.disabled=true;}
        if(selected)button.disabled=true;
        button.onclick=()=>reviewRequest(job,'',{revision:documentData.revision,issue_id:issue.issue_id,decision});buttons.append(button);
      }
      if(issue.operations.length){
        const candidateApproved=issue.decision==='approve';
        if(issue.requires_confirmation)item.append(element('p','partial-note',candidateApproved?'候选已确认；导出审核后版本时应用下方替换，无需手写整段。':issue.decision==='manual'?'本字段已有人工定稿，以下保留原候选供核对。':'AI 已给出明确候选，尚未执行。核对下方替换内容后可点“确认按建议改”，无需手写整段。'));
        if(issue.operations.some(op=>op.candidate_scope==='音频正文'))item.append(element('p','partial-note','此候选涉及音频正文，请核对原始音频后再确认。未经确认不会修改。'));
        if(issue.suggestion_candidate?.explanation)item.append(element('p','partial-note',`改稿说明：${issue.suggestion_candidate.explanation}`));
        const details=element('details','review-proposal');details.open=Boolean(issue.requires_confirmation);details.append(element('summary','',issue.requires_confirmation?`候选替换（${candidateApproved?'已确认':['reject','keep','keep_highlight','manual'].includes(issue.decision)?'未采用':'待人工确认'}）`:'指定修改内容'));
        const proposals=[...new Set(issue.operations.map(op=>`${op.candidate_old||op.old||'（补入位置）'} → ${op.candidate_new||op.new||'（删除）'}`))];
        for(const proposal of proposals)details.append(element('pre','review-source',proposal));
        item.append(details);
      }else{
        item.append(element('p','partial-note','报告尚未转成具体改稿。可让 AI 按这条建议生成修改前后，核对并确认；缺少事实依据时仍需人工判断。'));
        if(['pending','defer'].includes(issue.decision)){
          const propose=element('button','secondary-button','让 AI 按建议生成改稿');
          propose.disabled=view.busy||!field.sample_id||Object.hasOwn(documentData.field_edits,field.field_id);
          propose.title='调用 MiMo 处理本字段和该条建议；生成候选后仍须确认，不直接改原文。';
          if(view.busy&&view.saving?.issue_id===issue.issue_id&&!view.saving.decision)propose.textContent='AI 正在生成改稿…';
          propose.onclick=()=>reviewRequest(job,'/propose',{revision:documentData.revision,issue_id:issue.issue_id});
          item.append(propose);
        }
      }
      item.append(buttons);group.append(item);
      const localFeedback=view.feedback&&(view.feedback.issue_id===issue.issue_id||view.feedback.field_id===field.field_id);
      if(view.busy&&(view.saving?.issue_id===issue.issue_id||view.saving?.field_id===field.field_id))item.append(element('p','review-decision-feedback is-saving','正在保存…'));
      else if(localFeedback)item.append(element('p',`review-decision-feedback ${view.feedback.kind==='error'?'is-error':''}`,view.feedback.kind==='success'?'✓ 已保存':view.feedback.text));
    }
    if(!resolution.error){
      const finalView=element('details','review-final-text');
      finalView.append(element('summary','',resolution.changed?'当前最终字段（导出使用此文本）':'当前最终字段（保留原文）'),element('pre','review-source',resolution.final_text.trim()));
      group.append(finalView);
    }
    const subject=issues.find(issue=>issue.subject)?.subject;
    if(subject){const context=element('details','review-context');context.append(element('summary','','完整主体字段'),element('pre','review-source',subject));group.append(context);}
    const editor=element('details','review-editor');editor.append(element('summary','','人工填写最终完整字段'));
    editor.append(element('p','partial-note','修改整段字段正文，保留所需信息。保存时确认本字段所有剩余疑点；已标为误报或保留的结论继续保留。'));
    editor.append(element('pre','review-source',field.original));
    const area=document.createElement('textarea');area.setAttribute('aria-label',`题目 ${field.sample_id} ${field.name} 的人工最终稿`);
    area.value=view.drafts.get(field.field_id)??resolution.editor_text;
    editor.append(area);
    const save=element('button','secondary-button review-decision','确认本字段最终稿');
    const updateSave=()=>{
      const saved=Object.hasOwn(documentData.field_edits,field.field_id)&&area.value===documentData.field_edits[field.field_id];
      save.classList.toggle('is-selected',saved);save.setAttribute('aria-pressed',String(saved));
      save.textContent=saved?'✓ 最终稿已保存':view.busy&&view.saving?.field_id===field.field_id?'正在保存…':'确认本字段最终稿';
      save.disabled=view.busy||!field.sample_id||saved;
    };
    area.oninput=()=>{view.drafts.set(field.field_id,area.value);updateSave();};updateSave();
    save.onclick=()=>reviewRequest(job,'',{revision:documentData.revision,field_id:field.field_id,final_text:area.value});editor.append(save);
    if(Object.hasOwn(documentData.field_edits,field.field_id)){
      const reset=element('button','secondary-button','撤销本字段人工稿');reset.disabled=view.busy;
      reset.onclick=()=>{view.drafts.delete(field.field_id);reviewRequest(job,'',{revision:documentData.revision,field_id:field.field_id,reset:true});};editor.append(reset);
    }
    group.append(editor);panel.append(group);
  }
  if(job.review_output?.corrected_path){
    const output=job.review_output;
    const automatic=output.export_kind==='automatic'||(!output.export_kind&&output.corrected_path.includes('_自动修正_'));
    const current=!automatic&&output.review_revision===documentData.revision&&(!output.plan_fingerprint||output.plan_fingerprint===projection.fingerprint);
    panel.append(element('strong','',automatic?'自动修正副本（尚未人工审核）':current?'当前审核后版本已导出':'上次审核后版本：尚未包含最新审核，请重新导出'));
    panel.append(element('p','partial-note',`${output.changed_captions} 条 Caption、${output.changed_fields} 个字段有修改；${output.remaining_review_items} 项待处理。${!output.changed_captions?'没有文字改动，主文件按原文完整导出。':''}`));
    panel.append(element('code','review-path',output.corrected_path));
    const primary=element('button','secondary-button',automatic?'查看自动修正副本（未审核）':'查看审核后版本');
    primary.onclick=()=>action(`/api/jobs/${job.id}/reveal?file=corrected`);panel.append(primary);
    const attachments=element('details','review-attachments');attachments.open=view.attachmentsOpen;
    const heading=element('summary','','标黄名单与导出说明');
    heading.onclick=event=>{event.preventDefault();attachments.open=!attachments.open;view.attachmentsOpen=attachments.open;};attachments.append(heading);
    const warnings=output.export_warnings?.length?output.export_warnings:output.highlight_warning?[output.highlight_warning]:[];
    for(const warning of warnings)panel.append(element('p','partial-note',warning));
    const links=element('div','review-toolbar');
    for(const [kind,label,path] of [['highlighted','显示标黄 Excel',output.highlighted_path],['highlight_ids','显示标黄 ID 名单',output.highlight_ids_path],['changes','显示修改前后记录',output.change_log_path],['recheck','显示复检说明',output.recheck_path]]){
      if(!path||(kind==='highlighted'&&path===output.corrected_path))continue;
      const button=element('button','secondary-button',label);button.onclick=()=>action(`/api/jobs/${job.id}/reveal?file=${kind}`);links.append(button);
    }
    attachments.append(element('p','partial-note',`按审核选择标黄 ${output.highlighted_captions||0} 条；复检仅使用本地规则，不表示语义全部合格。`),links);
    panel.append(attachments);
  }
  return panel;
}
