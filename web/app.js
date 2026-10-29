'use strict';
const $ = (selector) => document.querySelector(selector);
const token = $('meta[name="app-token"]').content;
const allowed = new Set(['xlsx', 'csv', 'jsonl', 'json']);
const state = { files: [], uploading: false, config: null, jobs: [], jobsSignature: '', online: true, reportExpanded: new Map(), peopleExpanded: new Map(), rechecking: new Set() };
const statusNames = { queued: '等待处理', preparing: '正在读取文件', running: '正在质检', done: '已完成', partial: '部分未完成', failed: '处理失败' };

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function message(text) { $('#message').textContent = text || ''; $('#message').hidden = !text; }
function formatSize(size) { return size < 1024 * 1024 ? `${Math.max(1, Math.round(size / 1024))} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`; }
function updateButton() {
  const full = $('input[name="mode"]:checked').value === 'full';
  const credentialMissing = full && state.config && !state.config.key_ready;
  $('#start-button').disabled = state.uploading || !state.files.length || !state.config || !state.online || credentialMissing;
  $('#start-button span').textContent = state.uploading ? '正在上传…' : '开始质检';
  $('#submit-note').textContent = credentialMissing ? 'MiMo 密钥未就绪，可先选择底线检查' : state.files.length ? `将处理 ${state.files.length} 个文件，问题 ID 名单保存到下载文件夹` : '添加文件后即可开始';
}
function renderSelection() {
  $('#selected-area').hidden = !state.files.length;
  $('#selected-count').textContent = `已选择 ${state.files.length} 个文件`;
  $('#selected-files').replaceChildren(...state.files.map((file, index) => {
    const li = element('li');
    const info = element('div', 'file-info');
    info.append(element('strong', '', file.name), element('small', '', formatSize(file.size)));
    const remove = element('button', 'remove-file', '×');
    remove.setAttribute('aria-label', `移除 ${file.name}`);
    remove.disabled = state.uploading;
    remove.onclick = () => { state.files.splice(index, 1); renderSelection(); };
    li.append(element('span', 'file-mini', file.name.split('.').pop().toUpperCase()), info, remove);
    return li;
  }));
  $('#clear-files').disabled = state.uploading;
  updateButton();
}
function addFiles(files) {
  if (state.uploading) return;
  const errors = [];
  for (const file of files) {
    if (!allowed.has(file.name.split('.').pop().toLowerCase())) { errors.push(`${file.name}：格式不支持`); continue; }
    if (!file.size || file.size > 100 * 1024 * 1024) { errors.push(`${file.name}：文件为空或超过 100 MB`); continue; }
    if (!state.files.some(old => old.name === file.name && old.size === file.size && old.lastModified === file.lastModified)) state.files.push(file);
  }
  message(errors.join('\n'));
  renderSelection();
}
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'X-App-Token': token, ...(options.headers || {}) } });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '操作失败，请重试');
  return result;
}
async function action(path, body) {
  try { await api(path, { method: 'POST', ...(body ? { body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } } : {}) }); await refreshJobs(); }
  catch (error) { message(error.message); }
}
async function recheckJob(job) {
  if(state.rechecking.has(job.id))return;
  state.rechecking.add(job.id);renderJobs();
  try {
    await api(`/api/jobs/${job.id}/recheck`,{method:'POST'});
    message(`已新建重新检测记录：${job.name}。沿用原检测模式，原文件和旧审核记录保留。`);
    await refreshJobs();
  }catch(error){message(error.message);}
  finally{state.rechecking.delete(job.id);renderJobs();}
}
function renderCheckReport(job) {
  const report = job.check_report;
  if (!report?.checked || job.stale_results || job.outdated_instruction_parser) return null;
  const panel = element('details', 'check-report');
  panel.open = state.reportExpanded.get(job.id) ?? true;
  panel.setAttribute('aria-label', '检查项简报');
  const heading = element('summary', 'check-report-heading');
  heading.append(element('strong', '', job.dry_run ? '底线检查项简报' : '辅助筛查简报'), element('span', '', `${report.checked} 条 Caption${job.status === 'running' ? ' · 持续更新' : ''}`));
  const toggle = element('span', 'check-report-toggle', panel.open ? '▴ 收起报告' : '▾ 展开报告');
  heading.append(toggle);
  panel.addEventListener('toggle', () => {
    if (!panel.isConnected) return;
    state.reportExpanded.set(job.id, panel.open);
    toggle.textContent = panel.open ? '▴ 收起报告' : '▾ 展开报告';
  });
  panel.append(heading);
  const table = element('table', 'check-report-table');
  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  const reviewFirst=job.workflow_version==='review-first-v1';
  for (const label of reviewFirst?['检查项','涉及','待核对']:['检查项', '涉及', '已修复', '需复核']) {
    const th = element('th', '', label); th.scope = 'col'; headRow.append(th);
  }
  head.append(headRow); table.append(head);
  const body = document.createElement('tbody');
  for (const item of report.items) {
    if (job.dry_run && item.requires_semantic) continue;
    const unavailable = item.requires_semantic && !report.semantic_checked;
    const row = element('tr', !item.affected ? 'check-zero' : '');
    const label = element('th', '', item.label); label.scope = 'row'; row.append(label);
    row.append(element('td', '', unavailable ? '未检查' : item.affected));
    if(!reviewFirst)row.append(element('td', item.fixed ? 'check-fixed' : '', unavailable ? '—' : item.fixed));
    row.append(element('td', item.review ? 'check-review' : '', unavailable ? '—' : item.review));
    body.append(row);
  }
  table.append(body); panel.append(table);
  panel.append(element('p', 'check-report-note', reviewFirst?'按 Caption 计数；同条可含多项，各项不可直接相加。这是识别疑点统计，人工结论见审核面板。':'按 Caption 计数；同条可含多项，各项不可直接相加。已修复指该项已全部修复。'));
  if (!job.dry_run && report.semantic_checked < report.checked) panel.append(element('p', 'check-report-note', `语义检查已完成 ${report.semantic_checked} / ${report.checked} 条；其余仅有静态结果。`));
  const extra = [];
  if (report.correction_review_rows) extra.push(`修订未应用，需人工确认：${report.correction_review_rows} 条`);
  if (report.other_review_rows) extra.push(`其他人工复核说明：${report.other_review_rows} 条`);
  if (extra.length) panel.append(element('p', 'check-report-extra', extra.join('；')));
  return panel;
}
function renderPeopleReport(job) {
  const report = job.people_report;
  if (!report?.available || job.stale_results) return null;
  const panel = element('details', 'check-report people-report');
  panel.open = state.peopleExpanded.get(job.id) ?? false;
  panel.setAttribute('aria-label', '人员问题排名');
  const heading = element('summary', 'check-report-heading');
  heading.append(element('strong', '', report.basis === 'baseline' ? '人员底线问题排名' : '人员辅助筛查排名'), element('span', '', `${report.people.length} 人 · 按问题涉及率排序`));
  const toggle = element('span', 'check-report-toggle', panel.open ? '▴ 收起排名' : '▾ 展开排名');
  heading.append(toggle); panel.append(heading);
  panel.addEventListener('toggle', () => {
    if (!panel.isConnected) return;
    state.peopleExpanded.set(job.id, panel.open);
    toggle.textContent = panel.open ? '▴ 收起排名' : '▾ 展开排名';
  });
  panel.append(element('p', 'check-report-note', report.note));
  const wrap = element('div', 'people-table-wrap');
  const table = element('table', 'check-report-table people-table');
  const head = document.createElement('thead');
  const headings = document.createElement('tr');
  for (const label of ['排名', '标注员', '已检 Caption', '涉及问题', '涉及率', '已修复', '需复核', '未完成']) {
    const th = element('th', '', label); th.scope = 'col'; headings.append(th);
  }
  head.append(headings); table.append(head);
  const body = document.createElement('tbody');
  for (const person of report.people) {
    const row = document.createElement('tr');
    for (const value of [person.rank ?? '—', person.name, person.checked, person.problem_rows,
      person.problem_rate === null ? '—' : `${person.problem_rate.toFixed(1)}%`, person.fixed_rows, person.review_rows, person.not_checked]) row.append(element('td', '', value));
    body.append(row);
  }
  table.append(body); wrap.append(table); panel.append(wrap);
  if (report.unattributed_records) {
    panel.append(element('p', 'check-report-extra', `${report.unattributed_records} 条人员归属待确认，未纳入个人排名，其中 ${report.unattributed_problem_rows} 条涉及问题。`));
    for (const [reason, count] of Object.entries(report.unattributed_reasons)) panel.append(element('p', 'check-report-note', `${reason}：${count} 条`));
  }
  panel.append(element('p', 'check-report-note', '人员仅取自“标注员”列；只在本地汇总，不将姓名发送给模型。多人记录不推断责任人；按已确认规则去掉姓名末尾日期和 +，合并同名人员。'));
  return panel;
}
function renderJobs() {
  $('#job-count').textContent = state.jobs.length;
  $('#empty-state').hidden = !!state.jobs.length;
  $('#jobs').replaceChildren(...state.jobs.map(job => {
    const card = element('article', 'job-card');
    const top = element('div', 'job-top');
    const title = element('div', 'job-title');
    title.append(element('strong', '', job.name), element('small', '', (job.dry_run ? '底线检查 · 确定性规则' : '语义辅助筛查 · MiMo（非底线结论）') + (job.sheet ? ` · ${job.sheet}` : '') + ` · 并行 ${job.concurrency} 条`));
    top.append(element('span', 'file-mini', job.name.split('.').pop().toUpperCase()), title, element('span', `status-tag ${job.status}`, statusNames[job.status] || job.status));
    const progressLine = element('div', 'job-progress');
    const progress = document.createElement('progress');
    progress.max = Math.max(job.total, 1);
    progress.value = Math.min(job.processed, progress.max);
    progress.setAttribute('aria-label', '质检进度');
    progressLine.append(progress, element('span', '', job.total ? `${job.processed} / ${job.total} 条` : job.status === 'failed' ? '未开始' : '准备中'));
    const stats = element('div', 'job-stats');
    for (const [key, label] of [['PASS',job.dry_run ? '未发现底线问题' : '筛查未发现问题'], ['AUTO_FIXED',job.workflow_version==='review-first-v1'?'历史建议状态':job.dry_run ? '可生成修正' : '已修复'], ['NEED_HUMAN_REVIEW',job.workflow_version==='review-first-v1'?'待人工审核':job.dry_run ? '底线需处理' : '疑似需复核'], ['SKIPPED_NOT_HUMAN_CAPTION','已跳过'], ['NOT_CHECKED','检测未完成']]) {
      if(key==='AUTO_FIXED'&&job.workflow_version==='review-first-v1')continue;
      const item = element('span', '', label); item.append(element('b', '', job.counts[key] || 0)); stats.append(item);
    }
    card.append(top, progressLine, stats);
    if(job.status==='running'&&!job.dry_run){
      const waiting=element('div','scope-note');
      waiting.setAttribute('aria-label','AI 请求进度');
      waiting.append(element('p','',`单次请求最多 ${job.request_timeout_seconds || 60} 秒；每条 AI 检查含重试最多 ${job.caption_timeout_seconds || 180} 秒。超时项保留本地结果，可单独重试。`));
      for(const item of job.active_requests || []){
        const name=item.sample_id ? `ID ${item.sample_id}` : `数据第 ${item.row_index} 条`;
        const attempt=item.attempt ? ` · 第 ${item.attempt}/${item.max_attempts} 次请求` : '';
        const stage=item.state==='retry_wait' ? `请求失败（${item.error_type}），等待重试` : item.state==='timed_out' ? '已达总时限，正在保存' : '等待 AI 返回';
        waiting.append(element('div','',`${name} · 已用 ${item.elapsed_seconds} 秒${attempt} · ${stage}`));
      }
      card.append(waiting);
    }
    if(job.timed_out_count)card.append(element('div','partial-note',`${job.timed_out_count} 条 AI 检查达到总时限；已保存本地结果，不代表通过，可点击“重试未完成项”。`));
    if(job.workflow_version==='review-first-v1')card.append(element('p','partial-note','识别阶段保留原文，仅生成疑点与建议；人工审核后才回填。'));
    if(job.workflow_version?.startsWith('safe-auto-review-'))card.append(element('p','partial-note','白名单格式、符号宽度和词表错字默认自动修正；病句、主体和事实问题仍需人工。修正副本另存，原文件保留。'));
    if (job.caption_location) card.append(element('div', 'partial-note', `仅检查 ${job.caption_location} 的内容；不检查其他列。`));
    if (job.counts.SKIPPED_NOT_HUMAN_CAPTION) {
      const skip = element('details', 'partial-note');
      skip.append(element('summary', '', '跳过是什么意思？'));
      skip.append(element('p', '', '该行未进入质检，例如 Caption 为空或格式不受支持。跳过不等于通过，也不计入问题 ID 名单。'));
      for (const [reason, count] of Object.entries(job.id_summary?.skipped_reasons || {})) skip.append(element('div', '', `${reason}：${count} 行`));
      card.append(skip);
    }
    if (job.error) card.append(element('div', 'job-error', job.error));
    if (job.policy_note) card.append(element('div', 'partial-note', job.policy_note));
    if(job.outdated_instruction_parser)card.append(element('div','job-error','旧版字段识别或模型上下文不完整，新旧字段优先级、主体引用或画面元素引用可能存在误报及遗漏。请点“重新检测”生成新结果，沿用本条记录的检测模式；也可按新标题规则仅做本地复检。旧文件和审核记录保留。'));
    if(job.field_resolution_notes?.length){
      const notes=element('details','scope-note');
      notes.append(element('summary','',`字段版本说明：${job.field_resolution_notes.length} 条（新版优先）`));
      for(const row of job.field_resolution_notes){
        for(const note of row.notes)notes.append(element('p','',`${row.sample_id==null?'第 '+row.row_index+' 条':'ID '+row.sample_id} · ${note.field}：${note.message}`));
      }
      card.append(notes);
    }
    const subjectAlerts = job.outdated_instruction_parser?[]:job.id_summary?.instruction_subject_alerts || [];
    if (subjectAlerts.length) {
      const alert = element('details', 'job-error');
      alert.append(element('summary', '', `重点复核：${subjectAlerts.length} 条自然指令漏写已定义主体`));
      alert.append(element('p', '', '总结中的引用不能替代自然指令；仅提示人工核对，不自动补标签或动作。'));
      for (const item of subjectAlerts) alert.append(element('div', '', `ID ${item.id ?? '未提供'}（原表第 ${item.excel_row} 行）：缺少 ${item.missing_subjects.join('、')}`));
      card.append(alert);
    }
    if (job.status === 'partial') card.append(element('div', 'partial-note', job.pending_retry ? `${job.pending_retry} 条尚未完成语义检查，可重试继续；当前名单不代表全量结论。` : '存在未完成检查或缺失 ID，请查看检测说明；当前名单不代表全量结论。'));
    const checkReport = renderCheckReport(job);
    if (checkReport) card.append(checkReport);
    const peopleReport = job.outdated_instruction_parser?null:renderPeopleReport(job);
    if (peopleReport) card.append(peopleReport);
    const output = element('div', 'job-output');
    if(job.id_summary?.auto_correction_error)output.append(element('span','partial-note',`自动修正副本未保存：${job.id_summary.auto_correction_error} 请展开审核面板重试生成。`));
    if(job.review_output?.corrected_path){
      const reviewed=job.review_output.export_kind==='reviewed'||(!job.review_output.export_kind&&!job.review_output.corrected_path.includes('_自动修正_'));
      const corrected=element('button','secondary-button',reviewed?'查看审核后版本':'查看自动修正副本（未审核）');
      corrected.onclick=()=>action(`/api/jobs/${job.id}/reveal?file=corrected`);
      if(job.review_output.highlighted_captions){
        output.append(element('span','review-highlight-note',`${job.review_output.highlighted_path?'Caption 标黄':'待标黄 Caption'} ${job.review_output.highlighted_captions} 条`));
        if(job.review_output.highlighted_path&&job.review_output.highlighted_path!==job.review_output.corrected_path){
          const highlighted=element('button','secondary-button','显示标黄 Excel');
          highlighted.onclick=()=>action(`/api/jobs/${job.id}/reveal?file=highlighted`);output.append(highlighted);
        }
      }
      for(const warning of job.review_output.export_warnings?.length?job.review_output.export_warnings:job.review_output.highlight_warning?[job.review_output.highlight_warning]:[])output.append(element('span','partial-note',warning));
      output.append(corrected,element('span','partial-note',`${reviewed?'审核后版本已导出':'已完成自动格式修正，尚未人工审核'}；${job.review_output.changed_captions} 条 Caption 有修改，其余行完整保留。${reviewed&&job.review_summary?.revision!==undefined&&job.review_summary.revision!==job.review_output.review_revision?' 审核已有更新，请重新导出。':''}`));
    }
    if (job.output_ready) {
      output.append(element('strong', '', `问题 ID：${job.id_summary?.problem_ids ?? '—'} 个`));
      output.append(element('code', '', job.repair_report_path || job.output_path));
      if (job.repair_report_path) {
        const repair = element('button', 'secondary-button', '显示返修表');
        repair.onclick = () => action(`/api/jobs/${job.id}/reveal?file=repair`);
        output.append(repair);
      } else if (!job.outdated_policy && !job.stale_results) {
        const repair = element('button', 'secondary-button', job.repair_exporting ? '正在生成返修表…' : '导出返修表');
        repair.disabled = job.repair_exporting;
        repair.onclick = () => action(`/api/jobs/${job.id}/repair-report`);
        output.append(repair);
      }
      if (job.repair_report_error) output.append(element('span', 'partial-note', job.repair_report_error));
      const reveal = element('button', 'secondary-button', '显示 ID 名单');
      reveal.onclick = () => action(`/api/jobs/${job.id}/reveal`);
      output.append(reveal);
      if (job.explanation_path) {
        const explain = element('button', 'secondary-button', '显示检测说明');
        explain.onclick = () => action(`/api/jobs/${job.id}/reveal?file=explanation`);
        output.append(explain);
      }
      if (job.people_report_path) {
        const people = element('button', 'secondary-button', '显示人员排名');
        people.onclick = () => action(`/api/jobs/${job.id}/reveal?file=people`);
        output.append(people);
      }
    }
    if(['done','partial','failed'].includes(job.status)&&!job.repair_exporting){
      const activeRecheck=state.jobs.some(item=>item.recheck_of===job.id&&['queued','preparing','running'].includes(item.status));
      const recheck=element('button','secondary-button',state.rechecking.has(job.id)?'正在创建重新检测…':activeRecheck?'重新检测中…':'重新检测');
      recheck.disabled=state.rechecking.has(job.id)||activeRecheck;
      recheck.title='使用上传原文件和本条记录的检测模式，生成新记录；旧报告及审核结果保留。';
      recheck.onclick=()=>recheckJob(job);output.append(recheck);
    }
    if (['partial', 'failed'].includes(job.status) || (job.status === 'done' && (job.outdated_policy||job.outdated_instruction_parser))) {
      const concurrencyLabel = element('label', 'sheet-picker', '并行条数');
      const concurrencySelect = document.createElement('select');
      concurrencySelect.setAttribute('aria-label', `设置 ${job.name} 重试并行条数`);
      for (let count = 1; count <= 10; count++) {
        const option = element('option', '', `${count} 条`); option.value = String(count); concurrencySelect.append(option);
      }
      concurrencySelect.value = String(job.concurrency);
      concurrencyLabel.append(concurrencySelect); output.append(concurrencyLabel);
      let sheetSelect;
      if (job.status === 'failed' && !job.processed && job.sheet_choices && job.sheet_choices.length > 1) {
        const label = element('label', 'sheet-picker', '工作表');
        sheetSelect = document.createElement('select');
        sheetSelect.setAttribute('aria-label', `选择 ${job.name} 的工作表`);
        const placeholder = element('option', '', '请选择工作表'); placeholder.value = ''; sheetSelect.append(placeholder);
        for (const item of job.sheet_choices) {
          const option = element('option', '', item.name + (job.stale_results ? '' : `（${item.caption_rows} 条人标）`));
          option.value = item.name; sheetSelect.append(option);
        }
        sheetSelect.value = job.sheet || '';
        label.append(sheetSelect); output.append(label);
      }
      const retry = element('button', 'secondary-button', job.outdated_instruction_parser?'按新标题规则本地复检':job.outdated_policy && !job.stale_results ? '按底线规则重新检测' : '重试未完成项');
      retry.onclick = () => {
        if (sheetSelect && !sheetSelect.value) { message('请选择要处理的工作表。'); return; }
        action(`/api/jobs/${job.id}/retry`, { concurrency: Number(concurrencySelect.value), ...(sheetSelect ? { sheet: sheetSelect.value } : {}) });
      };
      output.append(retry);
    }
    if (output.childNodes.length) card.append(output);
    if(['done','partial'].includes(job.status)&&!job.stale_results&&!job.outdated_instruction_parser)card.append(renderReviewPanel(job));
    return card;
  }));
}
async function refreshJobs() {
  try {
    const data = await api('/api/jobs');
    state.online = true;
    const signature = JSON.stringify(data.jobs);
    if (signature !== state.jobsSignature) {
      state.jobs = data.jobs; state.jobsSignature = signature; renderJobs();
    }
    if (state.config) { $('#connection').textContent = state.config.key_ready ? 'MiMo 已就绪' : '未配置密钥'; $('#connection').classList.toggle('warning', !state.config.key_ready); }
  } catch (error) {
    state.online = false; $('#connection').textContent = '连接已断开'; $('#connection').classList.add('warning');
  }
  updateButton();
}
const dropzone = $('#dropzone');
dropzone.onclick = () => { if (!state.uploading) $('#file-input').click(); };
dropzone.onkeydown = event => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); dropzone.click(); } };
$('#file-input').onchange = event => { addFiles(event.target.files); event.target.value = ''; };
for (const name of ['dragenter', 'dragover']) dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('dragging'); });
dropzone.addEventListener('drop', event => addFiles(event.dataTransfer.files));
document.addEventListener('dragover', event => event.preventDefault());
document.addEventListener('drop', event => event.preventDefault());
$('#clear-files').onclick = () => { state.files = []; renderSelection(); };
document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change', updateButton));
$('#start-button').onclick = async () => {
  if (state.uploading || !state.files.length) return;
  const concurrency = Number($('#concurrency').value);
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 10) { message('并行条数请填写 1–10 的整数。'); return; }
  const options = { dry_run: $('input[name="mode"]:checked').value === 'static', concurrency,
                    column: $('#column').value.trim() || null, sheet: $('#sheet').value.trim() || null,
                    id_column: $('#id-column').value.trim() || null, review_first: true };
  const pending = [...state.files];
  state.uploading = true; renderSelection(); message('');
  try {
    for (const file of pending) {
      await api(`/api/jobs?name=${encodeURIComponent(file.name)}`, { method: 'POST', body: file,
        headers: { 'Content-Type': 'application/octet-stream', 'X-Job-Options': encodeURIComponent(JSON.stringify(options)) } });
      state.files = state.files.filter(old => old !== file);
      await refreshJobs();
    }
    $('#history-title').scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch (error) { message(error.message); }
  finally { state.uploading = false; renderSelection(); }
};
async function boot() {
  try { state.config = await api('/api/config'); $('#output-dir').textContent = state.config.output_dir; }
  catch (error) { message('无法连接工作台，请重新启动本地服务并刷新页面。'); }
  await refreshJobs();
  setInterval(refreshJobs, 1500);
}
boot();
