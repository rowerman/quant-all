import {esc, fmt, table} from './charts.js';
import {formatTime, statusLabel} from './mining_api.js';

function budgetBar(label, used, total) {
  const maximum = Number(total) || 0;
  const current = Number(used) || 0;
  const percent = maximum > 0 ? Math.min(100, current / maximum * 100) : 0;
  return `<div class="budget-item"><div><span>${esc(label)}</span><strong>${esc(fmt(current))} / ${esc(fmt(maximum))}</strong></div><div class="budget-track"><span style="width:${percent.toFixed(1)}%"></span></div></div>`;
}

function statusTag(status) {
  return `<span class="tag mining-status mining-status-${esc(status)}">${esc(statusLabel(status))}</span>`;
}

function candidateEvidence(item) {
  const diagnostics = item.selection_diagnostics || {};
  const correlation = diagnostics.correlation || {};
  const incremental = diagnostics.incremental || {};
  const stability = item.stability || {};
  const statistical = item.multiple_testing || {};
  return {
    cluster: correlation.cluster_id || '—',
    incremental: incremental.incremental_rank_ic,
    stability: stability.status ? statusLabel(stability.status) : '证据不足',
    stabilityReason: stability.reason || '—',
    qValue: statistical.bh_q_value,
  };
}

export function searchBudgetFields(defaults, limits) {
  const input = (name, label, value, maximum, hint = '') => `<label>${esc(label)}<input type="number" name="${name}" value="${esc(value)}" step="1" min="1" max="${esc(maximum)}" required>${hint ? `<span class="mini muted">${esc(hint)}</span>` : ''}</label>`;
  return [
    input('search_turns', '最大搜索轮次', defaults.max_search_turns, limits.max_search_turns),
    input('hypothesis_branches', '最大假设分支', defaults.max_hypothesis_branches, limits.max_hypothesis_branches),
    input('refines_per_branch', '每分支最大修正', defaults.max_refines_per_branch, limits.max_refines_per_branch),
    input('candidates_per_turn', '每轮最大候选', defaults.max_candidates_per_turn, limits.max_candidates_per_turn),
    input('candidates_per_run', '每运行最大候选', defaults.max_candidates_per_run, limits.max_candidates_per_run),
    input('full_evaluations', '完整评估候选', defaults.max_full_evaluations, limits.max_full_evaluations),
  ].join('');
}

export function renderSearchRun(app, detail, mutateRun) {
  const run = detail.run;
  const progress = detail.progress;
  const budget = progress.budget || {};
  const attempts = progress.attempts || {};
  const canCancel = ['queued', 'searching', 'selecting', 'evaluating', 'holding_out'].includes(run.status);
  const canResume = run.status === 'interrupted';
  app.innerHTML = `<div class="page-heading">
    <div><h1>V2 搜索 ${esc(run.run_id)}</h1><p class="intro muted">${esc(run.hypothesis)}</p></div>
    <a class="button-link secondary-link" href="#/mining">返回运行列表</a>
  </div>
  <section class="card">
    <div class="card-header"><h2>搜索状态</h2><div>${statusTag(run.status)}</div></div>
    <div id="mining-run-status" class="status-box">${progress.stop_reason ? esc(progress.stop_reason) : `当前阶段：${esc(statusLabel(progress.current_stage))} · 分支 ${esc(progress.current_branch_id || '—')}`}</div>
    <div class="run-action-row">
      ${canCancel ? '<button type="button" id="cancel-mining" class="secondary">取消运行</button>' : ''}
      ${canResume ? '<button type="button" id="resume-mining">恢复运行</button>' : ''}
    </div>
    <p class="mini muted">实验链路只读取开发期反馈；选择期、最终研究期和测试段不会进入 Worker。</p>
  </section>
  <section class="card">
    <h2>预算与尝试</h2>
    <div class="budget-progress">
      ${budgetBar('搜索轮次', progress.turns_completed, budget.max_search_turns)}
      ${budgetBar('候选尝试', attempts.proposed, budget.max_candidates_per_run)}
      ${budgetBar('完整评估', attempts.full_evaluated, budget.max_full_evaluations)}
      ${budgetBar('测试段领取', attempts.holdout_claimed, 1)}
    </div>
    <p class="mini muted">假设 ${esc(progress.hypotheses)} · 分支 ${esc(detail.branch_count)} · 候选 ${esc(progress.candidates)} · 失败候选 ${esc(progress.failed_candidates)} · 拒绝 ${esc(attempts.rejected || 0)} · 门禁失败 ${esc(attempts.gate_failed || 0)} · 当前动作 ${esc(statusLabel(progress.current_action))}</p>
  </section>
  <section class="card">
    <h2>假设树</h2>
    ${detail.hypotheses.length ? '<div id="search-hypotheses"></div>' : '<p class="muted">尚未登记假设。</p>'}
  </section>
  <section class="card">
    <h2>逐轮反馈</h2>
    ${detail.turns.length ? '<div id="search-turns"></div>' : '<p class="muted">尚未开始搜索轮次。</p>'}
  </section>
  <section class="card">
    <h2>工具补充证据</h2>
    <p class="mini muted">工具只能读取当前运行开发期摘要。主证据、工具补充证据与工具失败分开展示；缓存命中仍计一次调用。</p>
    <p class="mini muted">本运行调用 ${esc(detail.analysis_tools?.total_calls || 0)} 次 · 失败 ${esc(detail.analysis_tools?.failed_calls || 0)} 次</p>
    ${detail.analysis_tools?.total_calls ? '<div id="search-tool-calls"></div>' : '<p class="muted">本运行尚未使用分析工具。</p>'}
  </section>
  <section class="card">
    <h2>候选与评分卡</h2>
    ${detail.candidates.length ? '<div id="search-candidates"></div>' : '<p class="muted">尚无候选。</p>'}
  </section>
  <section class="card">
    <h2>防过拟合证据</h2>
    <p class="mini muted">质量通过只表示达到冻结研究门槛，不等于人工批准。模块测试段只允许一次性隔离验收。</p>
    ${detail.multiple_testing ? `<p class="mini muted">统计尝试 ${esc(detail.multiple_testing.attempt_count)} · 评估 ${esc(detail.multiple_testing.evaluated_count)} · 可检验 ${esc(detail.multiple_testing.tested_count)} · BH 分母 ${esc(detail.multiple_testing.denominator)}</p>` : '<p class="mini muted">统计诊断尚未生成。</p>'}
    <div id="search-overfit-evidence"></div>
  </section>
  <section class="card">
    <h2>安全事件</h2>
    <p class="mini muted">只展示脱敏业务事件，不展示 Prompt、表达式正文、行情数据或密钥。</p>
    <div id="mining-events" class="timeline" aria-live="polite"></div>
  </section>`;

  if (detail.hypotheses.length) {
    table(app.querySelector('#search-hypotheses'), detail.hypotheses.map(item => ({
      轮次: item.turn_index,
      分支: item.branch_id,
      假设: item.statement,
      父假设: item.parent_hypothesis_id || '—',
    })));
  }
  if (detail.turns.length) {
    table(app.querySelector('#search-turns'), detail.turns.map(item => ({
      轮次: item.turn_index,
      状态: statusLabel(item.status),
      动作: item.action ? statusLabel(item.action) : '—',
      原因: item.reason || '—',
      候选数: item.feedback?.candidate_count ?? 0,
      失败数: item.feedback?.failure_count ?? 0,
    })));
  }
  if (detail.candidates.length) {
    table(app.querySelector('#search-candidates'), detail.candidates.map(item => ({
      候选: `${item.name} · ${String(item.candidate_id).slice(-8)}`,
      状态: statusLabel(item.status),
      父候选: item.parent_candidate_id || '—',
      父假设: item.hypothesis_id,
      累计分析: item.analysis_count || 0,
      评分阶段: item.decision_stage || '证据不足',
      缺失项: item.missing_fields.length ? item.missing_fields.join(', ') : '—',
    })));
    app.querySelectorAll('#search-candidates tbody tr').forEach((row, index) => {
      const item = detail.candidates[index];
      const cell = row.querySelector('td');
      cell.innerHTML = `<a href="#/mining/${encodeURIComponent(run.run_id)}/candidate/${encodeURIComponent(item.candidate_id)}">${esc(item.name)}</a><div class="mini muted">${esc(item.candidate_id)}</div>`;
    });
    table(app.querySelector('#search-overfit-evidence'), detail.candidates.map(item => {
      const evidence = candidateEvidence(item);
      return {
        候选: `${item.name} · ${String(item.candidate_id).slice(-8)}`,
        质量结论: statusLabel(item.research_quality?.status),
        尝试次数: item.attempts?.proposed ?? 0,
        相关簇: evidence.cluster,
        增量RankIC: evidence.incremental,
        稳定性: evidence.stability,
        'BH q-value': evidence.qValue,
        说明: evidence.stabilityReason,
      };
    }));
  }
  if (detail.analysis_tools?.total_calls) {
    const calls = Object.entries(detail.analysis_tools.calls || {}).flatMap(([turn, items]) => items.map(item => ({...item, turn_index: Number(turn)})));
    table(app.querySelector('#search-tool-calls'), calls.map(item => ({
      轮次: item.turn_index,
      工具: `${item.tool_name} v${item.tool_version}`,
      候选: item.candidate_id ? String(item.candidate_id).slice(-8) : '运行级',
      类型: item.evidence_class === 'failure' ? '工具失败' : '补充证据',
      状态: item.status,
      参数: JSON.stringify(item.arguments_summary || {}),
      耗时毫秒: Number(item.duration_ms || 0).toFixed(1),
      缓存: item.cache_hit ? '命中' : '未命中',
      错误: item.error_code || '—',
    })));
  }
  app.querySelector('#cancel-mining')?.addEventListener('click', () => mutateRun('cancel'));
  app.querySelector('#resume-mining')?.addEventListener('click', () => mutateRun('resume'));
}
