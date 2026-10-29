'use strict';
const fieldState={file:null,busy:false,jobs:[],signature:'',expanded:new Map()};
function setFieldPanelExpanded(expanded){
  $('#field-panel-body').hidden=!expanded;
  $('#field-panel-toggle').setAttribute('aria-expanded',String(expanded));
  $('#field-panel-toggle').textContent=expanded?'▴ 收起':'▾ 展开';
  try{localStorage.setItem('captionqa.field-panel-expanded',String(expanded));}catch{}
}
$('#field-panel-toggle').onclick=()=>setFieldPanelExpanded($('#field-panel-body').hidden);
try{if(localStorage.getItem('captionqa.field-panel-expanded')==='false')setFieldPanelExpanded(false);}catch{}
function fieldMessage(text){const node=$('#field-message');node.textContent=text;node.hidden=!text;}
function fieldOptions(){
  const concurrency=Number($('#field-concurrency').value);
  if(!Number.isInteger(concurrency)||concurrency<1||concurrency>10)throw new Error('并行条数请填写1–10。');
  const raw=$('#field-task-map').value.trim();const task_mapping=raw?JSON.parse(raw):{};
  if(!task_mapping||Array.isArray(task_mapping)||typeof task_mapping!=='object')throw new Error('任务映射须填写JSON对象。');
  return {column:$('#field-column').value.trim()||null,id_column:$('#field-id-column').value.trim()||null,
    sheet:$('#field-sheet').value.trim()||null,concurrency,task_mapping};
}
function fieldSelect(files){
  const file=files[0];if(!file)return;
  if(!/\.(csv|xlsx|jsonl|json)$/i.test(file.name)||!file.size||file.size>100*1024*1024){fieldMessage('请选择不超过100 MB的CSV、XLSX、JSONL或JSON文件。');return;}
  fieldState.file=file;$('#field-selected').textContent=file.name;$('#field-start').disabled=fieldState.busy;fieldMessage('');
}
async function refreshFieldJobs(){
  try{
    const data=await api('/api/field-jobs');const signature=JSON.stringify(data.jobs);
    if(signature===fieldState.signature)return;
    fieldState.jobs=data.jobs;fieldState.signature=signature;renderFieldJobs();
  }catch(error){if(fieldState.jobs.some(j=>['queued','running'].includes(j.status)))fieldMessage('字段任务连接暂时中断，将自动重连。');}
}
async function fieldAction(job,path,body){
  try{await api(`/api/field-jobs/${job.id}/${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});fieldMessage('');await refreshFieldJobs();}
  catch(error){fieldMessage(error.message);}
}
function renderFieldJobs(){
  const active=fieldState.jobs.filter(job=>['queued','running'].includes(job.status)).length;
  $('#field-panel-status').textContent=active?`${active} 个任务处理中`:fieldState.jobs.length?`${fieldState.jobs.length} 条记录`:'独立于内容质检';
  $('#field-jobs').replaceChildren(...fieldState.jobs.map(job=>{
    const card=element('article','field-job');card.dataset.fieldJobId=job.id;
    const details=element('details','field-job-details');details.open=fieldState.expanded.get(job.id)??false;
    const heading=element('summary','field-job-summary');
    const overview=element('div');overview.append(element('h3','',job.name));
    const status={queued:'等待处理',running:'正在整理字段',done:'字段规范完成',partial:'已输出，部分题目需确认',failed:'处理未完成'}[job.status]||job.status;
    overview.append(element('p','field-job-meta',`${status} · ${job.processed} / ${job.total} 条`));
    const counts=job.counts||{};
    overview.append(element('p','field-job-stat',`${job.outdated_schema?'旧规则 · ':''}已规范 ${counts.already_standard||0} · 已调整 ${counts.converted||0} · 需确认 ${counts.needs_confirmation||0}${counts.failed?' · 失败 '+counts.failed:''}${job.missing_field_count?' · 缺字段 '+job.missing_field_count:''}`));
    const toggle=element('span','field-job-toggle',details.open?'▴ 收起详情':'▾ 展开详情');
    heading.append(overview,toggle);details.append(heading);
    heading.addEventListener('click',event=>{
      event.preventDefault();details.open=!details.open;
      fieldState.expanded.set(job.id,details.open);
      toggle.textContent=details.open?'▴ 收起详情':'▾ 展开详情';
    });
    const body=element('div','field-job-body');
    body.append(element('p','',`${job.outdated_schema?'旧规则统计：':''}字段已规范 ${counts.already_standard||0} · 字段已调整 ${counts.converted||0} · 需确认 ${counts.needs_confirmation||0} · 失败 ${counts.failed||0} · 跳过 ${counts.skipped||0}`));
    if(job.outdated_schema)body.append(element('p','partial-note','此记录使用旧字段规则，统计及导出文件尚未更新。当前标准保留 video_similarity_tag，## 后须有一个半角空格，每个字段内容结束后留一个空行；新版还可处理 task_type 已写子任务标签、subtask_type 错放生成要求的情况。可按当前设置重试。'));
    if(job.legacy_reference_count)body.append(element('p','partial-note',`旧任务类型为“参考”：${job.legacy_reference_count} 条，已单独标记 ID。`));
    if(job.missing_field_count)body.append(element('p','partial-note',`原始缺字段：${job.missing_field_count} 条，已标记 ID；已补齐的记录也保留标记。`));
    if(job.selected_column)body.append(element('p','partial-note',`仅处理 ${job.selected_sheet?job.selected_sheet+' / ':''}${job.selected_column}；其他列保留。`));
    if(job.error)body.append(element('p','job-error',job.error));
    if(['failed','partial'].includes(job.status)||job.outdated_schema){
      const retry=element('button','secondary-button','按当前字段设置重试');
      retry.onclick=()=>{try{fieldAction(job,'retry',fieldOptions());}catch(error){fieldMessage(error.message);}};body.append(retry);
    }
    const buttons=element('div','review-toolbar');
    for(const [kind,label] of [['corrected','显示修正副本'],['report','显示 ID 报告'],['fixed_ids','显示已修复 ID'],['review_ids','显示需确认 ID'],['legacy_reference_ids','显示旧“参考”ID名单'],['missing_field_ids','显示缺字段 ID'],['missing_field_report','显示缺字段明细']]){
      if(!job.outputs?.[kind])continue;
      const button=element('button','secondary-button',label);button.onclick=()=>fieldAction(job,'reveal',{kind});buttons.append(button);
    }
    if(buttons.childNodes.length){body.append(buttons,element('code','',job.outputs.corrected));}
    details.append(body);card.append(details);
    return card;
  }));
}
const fieldDrop=$('#field-dropzone');
fieldDrop.onclick=()=>{if(!fieldState.busy)$('#field-file').click();};
fieldDrop.onkeydown=event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();fieldDrop.click();}};
for(const type of ['dragenter','dragover'])fieldDrop.addEventListener(type,event=>{event.preventDefault();event.stopPropagation();fieldDrop.classList.add('dragging');});
for(const type of ['dragleave','drop'])fieldDrop.addEventListener(type,event=>{event.preventDefault();event.stopPropagation();fieldDrop.classList.remove('dragging');});
fieldDrop.addEventListener('drop',event=>{if(!fieldState.busy)fieldSelect(event.dataTransfer.files);});
$('#field-file').onchange=event=>{fieldSelect(event.target.files);event.target.value='';};
$('#field-start').onclick=async()=>{
  if(fieldState.busy||!fieldState.file)return;
  let options;try{options=fieldOptions();}catch(error){fieldMessage('设置有误：'+error.message);return;}
  fieldState.busy=true;$('#field-start').disabled=true;fieldMessage('正在上传字段规范文件…');
  try{
    const file=fieldState.file;
    await api(`/api/field-jobs?name=${encodeURIComponent(file.name)}`,{method:'POST',body:file,
      headers:{'Content-Type':'application/octet-stream','X-Job-Options':encodeURIComponent(JSON.stringify(options))}});
    fieldState.file=null;$('#field-selected').textContent='或点击选择 CSV / XLSX / JSONL / JSON';
    fieldMessage('文件已添加。完成后自动把修正副本和 ID 报告保存到下载目录；需确认的题目保留原文。');await refreshFieldJobs();
  }catch(error){fieldMessage(error.message);}
  finally{fieldState.busy=false;$('#field-start').disabled=!fieldState.file;}
};
refreshFieldJobs();setInterval(refreshFieldJobs,1500);
