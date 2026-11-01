import {esc, fmt, table} from './charts.js';
import {
  MiningRequestError,
  consumeMiningStream,
  formatTime,
  miningJSON,
  miningRequest,
  statusLabel,
} from './mining_api.js';
import {renderCandidate} from './mining_candidate.js';
import {renderSearchRun, searchBudgetFields} from './mining_search.js';

let options = null;
let activeController = null;

const activeStatuses = new Set(['queued', 'generating', 'searching', 'selecting', 'screening', 'evaluating', 'holding_out']);

export function disposeMining() {
  activeController?.abort();
  activeController = null;
}

async function miningOptions() {
  if (!options) options = await miningRequest('/api/mining/options');
  return options;
}

function errorText(error) {
  return `${error.message}${error.code ? ` [${error.code}]` : ''}${error.requestId ? ` 请求 ${error.requestId}` : ''}`;
}

function statusTag(status) {
  return `<span class="tag mining-status mining-status-${esc(status)}">${esc(statusLabel(status))}</span>`;
}

async function listPage(app, current) {
  const runs = await miningRequest('/api/mining/runs?limit=50');
  if (!current()) return;
  app.innerHTML = `<div class="page-heading">
    <div><h1>自动因子挖掘</h1><p class="intro muted">提交研究假设后，系统调用受控 Worker 生成候选，完成技术门禁、快速筛选、完整评估和一次性模块测试段，再由人工审核。</p></div>
    <a class="button-link" href="#/mining/new">新建挖掘运行</a>
  </div>
  <section class="card">
    <h2>运行记录</h2>
    ${runs.length ? '<div id="mining-runs"></div>' : '<p class="muted">还没有挖掘运行。</p>'}
  </section>
  <p class="note">批准只记录人工决定，不会自动修改正式因子库、模型、策略或回测。</p>`;
  if (!runs.length) return;
  const wrap = app.querySelector('#mining-runs');
  wrap.innerHTML = `<div class="table-wrap"><table><thead><tr><th>研究假设</th><th>状态</th><th>阶段</th><th>创建时间</th><th>错误</th></tr></thead><tbody>
    ${runs.map(run => `<tr><td><a href="#/mining/${encodeURIComponent(run.run_id)}">${esc(run.hypothesis)}</a><div class="mini muted">${esc(run.run_id)} · ${esc(run.engine || 'v1')}</div></td><td>${statusTag(run.status)}</td><td>${esc(statusLabel(run.last_completed_stage || run.status))}</td><td>${esc(formatTime(run.created_at))}</td><td>${esc(run.error_code || '—')}${run.error_message ? `<div class="mini muted">${esc(run.error_message)}</div>` : ''}</td></tr>`).join('')}
  </tbody></table></div>`;
}

function numericField(name, label, value, step, min, max, hint = '') {
  return `<label>${esc(label)}<input type="number" name="${name}" value="${esc(value)}" step="${esc(step)}" min="${esc(min)}" max="${esc(max)}" required>${hint ? `<span class="mini muted">${esc(hint)}</span>` : ''}</label>`;
}

async function newPage(app, current) {
  const opts = await miningOptions();
  if (!current()) return;
  const v2 = Boolean(opts.search_v2?.enabled);
  const defaults = v2 ? opts.search_v2.defaults : opts.defaults;
  const v2Options = opts.search_v2 || {};
  const research = opts.search_v2?.research || {};
  const budgetFields = v2
    ? searchBudgetFields(v2Options.defaults, v2Options.limits)
    : `
        ${numericField('max_rounds', '最大轮次', defaults.max_rounds, '1', 1, opts.limits.max_rounds)}
        ${numericField('max_candidates', '最大候选数', defaults.max_candidates, '1', 1, opts.limits.max_candidates)}
        ${numericField('full_eval_top_k', '完整评估数', defaults.full_eval_top_k, '1', 1, opts.limits.full_eval_top_k)}
        ${numericField('max_llm_calls', 'LLM 调用上限', defaults.max_llm_calls, '1', 1, opts.limits.max_llm_calls)}
        ${numericField('max_expression_depth', '表达式最大深度', defaults.max_expression_depth, '1', 2, opts.limits.max_expression_depth)}
        <label>方向<select name="direction"><option value="either" ${defaults.direction === 'either' ? 'selected' : ''}>不限</option><option value="positive" ${defaults.direction === 'positive' ? 'selected' : ''}>正向</option><option value="negative" ${defaults.direction === 'negative' ? 'selected' : ''}>负向</option></select></label>
        ${numericField('min_coverage', '最低覆盖率（0～1）', defaults.min_coverage, '0.01', 0, 1)}
        ${numericField('max_failure_rate', '失败率停止阈值（0～1）', defaults.max_failure_rate, '0.01', 0.01, 1)}
        ${numericField('min_quality_checks_for_failure_stop', '失败率最少检查数', defaults.min_quality_checks_for_failure_stop, '1', 1, opts.limits.max_candidates)}
        ${numericField('seed', '随机种子', defaults.seed, '1', -2147483648, 2147483647)}
        <label>排除字段<input name="excluded_fields" value="${esc((defaults.excluded_fields || []).join(', '))}" placeholder="用逗号分隔"><span class="mini muted">仅排除已登记字段，服务端会再次校验。</span></label>`;
  const windowsText = v2
    ? `开发期 ${esc(v2Options.windows.development_start)}～${esc(v2Options.windows.development_end)} · 选择期仅供控制器 · 最终研究期和模块测试段不进入 Worker`
    : `研究期 ${esc(opts.windows.research_start)}～${esc(opts.windows.research_end)} · 完整评估 ${esc(opts.windows.full_eval_start)}～${esc(opts.windows.full_eval_end)} · 测试段 ${esc(opts.windows.holdout_start)}～${esc(opts.windows.holdout_end)}`;
  app.innerHTML = `<div class="page-heading">
    <div><h1>新建挖掘运行</h1><p class="intro muted">${v2 ? '可从手工假设或服务端研报资料启动。选择期、最终研究期和测试段不会反馈给 Worker。' : '假设会与预算、数据快照、字段目录和算子目录一起冻结。Worker 不可用时任务会明确失败，不会回退到 Mock。'}</p></div>
    <a class="button-link secondary-link" href="#/mining">返回运行列表</a>
  </div>
  <form id="mining-form">
    <section class="card">
      <h2>研究输入</h2>
      ${v2 ? `<label>输入模式<select name="source_mode" id="source-mode"><option value="hypothesis">手工研究假设</option><option value="research">资料驱动</option></select></label>` : ''}
      <div data-mode-section="hypothesis">
        <label>研究假设<textarea name="hypothesis" minlength="8" maxlength="4000" rows="7" required placeholder="例如：短期换手率下降可能预示未来五日超额收益。"></textarea></label>
      </div>
      ${v2 ? `<div data-mode-section="research" hidden>
        <label>资料相对路径<input name="material_path" value="." maxlength="1000" placeholder="例如：2026-10 或 report.pdf"><span class="mini muted">只接受 ${esc(research.material_root || 'data/source/reports')} 下的 PDF 文件或目录；默认递归扫描。</span></label>
        <p class="mini muted">系统先解析文本型 PDF，再由 Worker v5/v6 抽取和转换候选。扫描件 OCR 尚未支持。</p>
      </div>` : ''}
      <p class="mini muted">固定口径：${esc(opts.fixed.universe)} · ${esc(opts.fixed.label)} · ${esc(opts.fixed.price_channel)}。研究与测试窗口在运行时冻结。</p>
    </section>
    <details class="card">
      <summary><strong>高级预算</strong> · 默认使用已验收预算</summary>
      <div class="form-grid budget-grid">
        ${budgetFields}
      </div>
      <div class="budget-windows">${windowsText}</div>
    </details>
    <div id="mining-form-status" class="status-box muted">${opts.worker_configured ? 'Worker 已配置；提交时仍会检查健康状态。' : '主服务尚未配置 MINING_WORKER_URL；提交会明确失败。'}</div>
    <div class="form-actions"><button type="submit">创建并开始运行</button><a class="button-link secondary-link" href="#/mining">取消</a></div>
  </form>`;
  const form = app.querySelector('#mining-form');
  const status = app.querySelector('#mining-form-status');
  const sourceMode = form.querySelector('#source-mode');
  const syncMode = () => {
    const mode = sourceMode?.value || (v2 ? 'hypothesis' : 'hypothesis');
    form.querySelectorAll('[data-mode-section]').forEach(section => {
      section.hidden = section.dataset.modeSection !== mode;
    });
    const hypothesis = form.querySelector('[name=hypothesis]');
    const materialPath = form.querySelector('[name=material_path]');
    if (hypothesis) hypothesis.required = mode === 'hypothesis';
    if (materialPath) materialPath.required = mode === 'research';
  };
  sourceMode?.addEventListener('change', syncMode);
  syncMode();
  let idempotencyKey = `web-${crypto.randomUUID()}`;
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const data = new FormData(form);
    const numeric = name => Number(data.get(name));
    const constraints = v2 ? null : {
      max_rounds: numeric('max_rounds'),
      max_candidates: numeric('max_candidates'),
      full_eval_top_k: numeric('full_eval_top_k'),
      max_llm_calls: numeric('max_llm_calls'),
      max_expression_depth: numeric('max_expression_depth'),
      direction: data.get('direction'),
      min_coverage: numeric('min_coverage'),
      max_failure_rate: numeric('max_failure_rate'),
      min_quality_checks_for_failure_stop: numeric('min_quality_checks_for_failure_stop'),
      seed: numeric('seed'),
      excluded_fields: String(data.get('excluded_fields') || '').split(',').map(item => item.trim()).filter(Boolean),
    };
    const submit = form.querySelector('[type=submit]');
    submit.disabled = true;
    status.className = 'status-box';
    status.textContent = '正在冻结上下文并提交 Worker 任务…';
    try {
      const researchMode = sourceMode?.value === 'research';
      const body = researchMode
        ? {
            material_path: data.get('material_path'),
            idempotency_key: idempotencyKey,
          }
        : {
            hypothesis: data.get('hypothesis'),
            idempotency_key: idempotencyKey,
          };
      if (v2) {
        body.budget = {
          ...v2Options.defaults,
          max_search_turns: numeric('search_turns'),
          max_hypothesis_branches: numeric('hypothesis_branches'),
          max_refines_per_branch: numeric('refines_per_branch'),
          max_candidates_per_turn: numeric('candidates_per_turn'),
          max_candidates_per_run: numeric('candidates_per_run'),
          max_full_evaluations: numeric('full_evaluations'),
        };
      } else {
        body.constraints = constraints;
      }
      const detail = await miningJSON(
        researchMode ? '/api/mining/research-runs' : '/api/mining/runs',
        body,
      );
      location.hash = `#/mining/${encodeURIComponent(detail.run.run_id)}`;
    } catch (error) {
      submit.disabled = false;
      status.className = 'status-box error';
      status.textContent = `${errorText(error)}。本次幂等键已保留，可在修复后重试。`;
    }
  });
}

function eventNode(event) {
  const node = document.createElement('article');
  node.className = 'mining-event';
  node.dataset.sequence = event.sequence;
  node.innerHTML = `<div class="event-line"><span class="event-sequence">#${esc(event.sequence)}</span><span class="tag">${esc(event.stage)}</span><strong>${esc(event.event_type)}</strong><span class="muted">${esc(formatTime(event.created_at))}</span></div>
    <details><summary>事件载荷</summary><pre>${esc(JSON.stringify(event.payload, null, 2))}</pre></details>`;
  return node;
}

function appendEvents(container, events, seen) {
  for (const event of events) {
    if (seen.has(event.sequence)) continue;
    seen.add(event.sequence);
    container.append(eventNode(event));
  }
}

function renderRun(app, detail, opts, current) {
  const run = detail.run;
  async function mutateRun(action) {
    const status = app.querySelector('#mining-run-status');
    const button = app.querySelector(action === 'cancel' ? '#cancel-mining' : '#resume-mining');
    if (action === 'cancel' && !confirm('确定在下一个安全边界取消该运行？')) return;
    button.disabled = true;
    status.className = 'status-box';
    status.textContent = action === 'cancel' ? '正在请求取消…' : '正在恢复运行…';
    try {
      await miningJSON(`/api/mining/runs/${encodeURIComponent(run.run_id)}/${action}`, {});
      if (!current()) return;
      await showRun(app, run.run_id, current);
    } catch (error) {
      button.disabled = false;
      status.className = 'status-box error';
      status.textContent = errorText(error);
    }
  }
  if (run.engine === 'v2') {
    renderSearchRun(app, detail, mutateRun);
    return;
  }
  const progress = detail.progress;
  const budget = run.context.budget || {};
  const candidates = detail.candidates;
  const canCancel = activeStatuses.has(run.status);
  const canResume = run.status === 'interrupted';
  app.innerHTML = `<div class="page-heading">
    <div><h1>运行 ${esc(run.run_id)}</h1><p class="intro muted">${esc(run.hypothesis)}</p></div>
    <a class="button-link secondary-link" href="#/mining">返回运行列表</a>
  </div>
  <section class="card">
    <div class="card-header"><h2>运行状态</h2><div>${statusTag(run.status)}</div></div>
    <div id="mining-run-status" class="status-box">${run.error_message ? `${esc(run.error_code || '')} ${esc(run.error_message)}` : `当前阶段：${esc(statusLabel(progress.current_stage || run.last_completed_stage))}`}</div>
    <div class="run-action-row">
      ${canCancel ? '<button type="button" id="cancel-mining" class="secondary">取消运行</button>' : ''}
      ${canResume ? '<button type="button" id="resume-mining">恢复运行</button>' : ''}
    </div>
  </section>
  <section class="card">
    <h2>冻结口径与预算</h2>
    <div id="run-context"></div>
  </section>
  <section class="card">
    <h2>预算消耗</h2>
    <div class="budget-progress">
      ${progressBar('轮次', progress.rounds_completed, budget.max_rounds)}
      ${progressBar('LLM 调用', progress.llm_calls_used, budget.max_llm_calls)}
      ${progressBar('候选', progress.candidates_stored, budget.max_candidates)}
      ${progressBar('质量检查', progress.quality_checks_used, budget.max_candidates)}
      ${progressBar('完整评估', progress.full_evaluations_used, budget.full_eval_top_k)}
    </div>
    <p class="mini muted">失败候选 ${esc(progress.failed_candidates)} · 测试段${progress.holdout_claimed ? '已领取' : '未领取'} · 最后完成阶段 ${esc(statusLabel(run.last_completed_stage))}</p>
  </section>
  <section class="card">
    <h2>候选</h2>
    ${candidates.length ? '<div id="run-candidates"></div>' : '<p class="muted">尚无可展示候选。</p>'}
  </section>
  <section class="card">
    <h2>Agent 运行轨迹</h2>
    <p class="mini muted">仅展示已保存的业务事件和安全轮次摘要，不展示模型隐藏思维链、原始提示或密钥。</p>
    <div id="mining-events" class="timeline" aria-live="polite"></div>
  </section>`;

  const contextRows = [
    {项目: '数据版本', 值: run.context.data_version},
    {项目: '数据指纹', 值: run.context.data_fingerprint},
    {项目: '标签', 值: run.context.label},
    {项目: '股票池', 值: run.context.universe},
    {项目: '价格通道', 值: run.context.price_channel},
    {项目: '研究期', 值: `${run.context.research_start}～${run.context.research_end}`},
    {项目: '完整评估', 值: `${run.context.full_eval_start}～${run.context.full_eval_end}`},
    {项目: '模块测试段', 值: `${run.context.holdout_start}～${run.context.holdout_end}`},
    {项目: '上下文哈希', 值: run.context.context_hash},
    {项目: '创建时间', 值: formatTime(run.created_at)},
  ];
  table(app.querySelector('#run-context'), contextRows);
  if (candidates.length) {
    table(app.querySelector('#run-candidates'), candidates.map(item => ({
      候选: `${item.name} · ${String(item.candidate_id).slice(-8)}`,
      状态: statusLabel(item.status),
      轮次: item.discovery_round,
      序号: item.discovery_index,
      错误: item.error_code || '—',
    })));
    app.querySelectorAll('#run-candidates tbody tr').forEach((row, index) => {
      row.querySelector('td').innerHTML = `<a href="#/mining/${encodeURIComponent(run.run_id)}/candidate/${encodeURIComponent(candidates[index].candidate_id)}">${esc(candidates[index].name)}</a><div class="mini muted">${esc(candidates[index].candidate_id)}</div>`;
    });
  }
  app.querySelector('#cancel-mining')?.addEventListener('click', () => mutateRun('cancel'));
  app.querySelector('#resume-mining')?.addEventListener('click', () => mutateRun('resume'));

}

function progressBar(label, used, total) {
  const maximum = Number(total) || 0;
  const current = Number(used) || 0;
  const percent = maximum > 0 ? Math.min(100, current / maximum * 100) : 0;
  return `<div class="budget-item"><div><span>${esc(label)}</span><strong>${esc(fmt(current))} / ${esc(fmt(maximum))}</strong></div><div class="budget-track"><span style="width:${percent.toFixed(1)}%"></span></div></div>`;
}

async function showRun(app, runId, current) {
  const opts = await miningOptions();
  const detail = await miningRequest(`/api/mining/runs/${encodeURIComponent(runId)}`);
  if (!current()) return;
  renderRun(app, detail, opts, current);
  const eventPage = await miningRequest(`/api/mining/runs/${encodeURIComponent(runId)}/events?after=0&limit=1000`);
  if (!current()) return;
  const timeline = app.querySelector('#mining-events');
  const seen = new Set();
  appendEvents(timeline, eventPage.events, seen);
  let cursor = eventPage.last_sequence;
  if (!activeStatuses.has(detail.run.status)) return;
  const local = new AbortController();
  activeController = local;
  const status = app.querySelector('#mining-run-status');
  let retries = 0;
  while (!local.signal.aborted) {
    let done = false;
    try {
      status.className = 'status-box';
      status.textContent = '事件流已连接，等待阶段变化…';
      await consumeMiningStream(
        `/api/mining/runs/${encodeURIComponent(runId)}/stream?after=${cursor}`,
        local.signal,
        (event, payload) => {
          if (event === 'mining') {
            appendEvents(timeline, [payload], seen);
            cursor = Math.max(cursor, Number(payload.sequence) || 0);
            status.textContent = `阶段 ${payload.stage} · ${payload.event_type}`;
          } else if (event === 'error') {
            throw new MiningRequestError(payload.message, payload.code, payload.request_id);
          } else if (event === 'done') {
            done = true;
            cursor = Math.max(cursor, Number(payload.last_sequence) || 0);
          }
        },
        sequence => { cursor = Math.max(cursor, sequence); },
      );
      if (done && current()) await showRun(app, runId, current);
      return;
    } catch (error) {
      if (local.signal.aborted) return;
      retries += 1;
      status.className = 'status-box warning';
      status.textContent = `事件流中断：${errorText(error)}；正在从序号 ${cursor} 重连（第 ${retries} 次）…`;
      await new Promise(resolve => setTimeout(resolve, Math.min(5000, retries * 800)));
    }
  }
}

async function candidatePage(app, runId, candidateId, current) {
  const opts = await miningOptions();
  const detail = await miningRequest(
    `/api/mining/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(candidateId)}`,
  );
  if (!current()) return;
  renderCandidate(
    app,
    detail,
    opts,
    async (decision, reviewer, note) => {
      await miningJSON(
        `/api/mining/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(candidateId)}/decision`,
        {
          decision,
          reviewer,
          note,
          candidate_content_hash: detail.review.candidate_content_hash,
          evidence_hash: detail.review.evidence_hash,
        },
      );
      if (current()) await candidatePage(app, runId, candidateId, current);
    },
    async candidateIds => {
      await miningJSON(
        `/api/mining/runs/${encodeURIComponent(runId)}/publication-batches`,
        {
          candidate_ids: candidateIds,
          expected_context_hash: detail.run.context_hash || detail.run.context.context_hash,
        },
      );
      if (current()) await candidatePage(app, runId, candidateId, current);
    },
    async () => {
      if (current()) await candidatePage(app, runId, candidateId, current);
    },
  );
}

export async function miningRoute(hash, current) {
  const app = document.querySelector('#app');
  try {
    if (hash === '#/mining' || hash === '#/mining/') {
      await listPage(app, current);
      return;
    }
    if (hash === '#/mining/new') {
      await newPage(app, current);
      return;
    }
    const candidate = hash.match(/^#\/mining\/([^/]+)\/candidate\/([^/]+)$/);
    if (candidate) {
      await candidatePage(
        app,
        decodeURIComponent(candidate[1]),
        decodeURIComponent(candidate[2]),
        current,
      );
      return;
    }
    const run = hash.match(/^#\/mining\/([^/]+)$/);
    if (run) {
      await showRun(app, decodeURIComponent(run[1]), current);
      return;
    }
    app.innerHTML = '<p role="alert">未知的挖掘页面。</p><a href="#/mining">返回挖掘列表</a>';
  } catch (error) {
    if (current()) {
      app.innerHTML = `<p role="alert">${esc(errorText(error))}</p><a href="#/mining">返回挖掘列表</a>`;
    }
  }
}
