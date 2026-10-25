import {esc, fmt, generic, renderMetric, table} from './charts.js';
import {formatTime, statusLabel} from './mining_api.js';

const stageLabels = {
  technical: '技术门禁',
  quick: '研究期快速筛选',
  full: '完整评估',
  holdout: '模块测试段',
};

function latestReport(reports, stage) {
  return reports.filter(item => item.stage === stage).at(-1) || null;
}

function reportRow(label, report) {
  if (!report) return {阶段: label, 结论: '缺失', 耗时: '—', 警告: '—'};
  return {
    阶段: label,
    结论: statusLabel(report.status),
    耗时: report.started_at && report.finished_at
      ? `${Math.max(0, (new Date(report.finished_at) - new Date(report.started_at)) / 1000).toFixed(1)} 秒`
      : '—',
    警告: report.warnings?.join('；') || '—',
    错误: report.error_message || '—',
  };
}

function renderStageSections(container, events) {
  const entries = Object.entries(events || {}).filter(([stage]) => !['meta', 'done'].includes(stage));
  if (!entries.length) {
    container.innerHTML = '<p class="muted">本阶段没有可展示的评估结果。</p>';
    return;
  }
  for (const [stage, payload] of entries) {
    const section = document.createElement('section');
    section.className = 'card candidate-stage';
    section.innerHTML = `<h3>${esc(stageLabels[stage] || stage)}</h3>`;
    const metrics = payload.metrics || {};
    if (!Object.keys(metrics).length) {
      section.insertAdjacentHTML('beforeend', '<p class="muted">没有可展示指标。</p>');
    }
    for (const [key, value] of Object.entries(metrics)) {
      const node = document.createElement('div');
      node.className = 'candidate-chart';
      node.innerHTML = `<h4>${esc(key)}</h4>`;
      const content = document.createElement('div');
      node.append(content);
      renderMetric(content, key, value);
      section.append(node);
    }
    container.append(section);
  }
}

function decisionForm(detail) {
  const review = detail.review;
  const candidate = detail.candidate;
  if (detail.approval) {
    return `<section class="card decision-card">
      <h2>人工审批</h2>
      <p><span class="tag decision-${esc(detail.approval.decision)}">${detail.approval.decision === 'approved' ? '已批准' : '已拒绝'}</span>
      审核人 ${esc(detail.approval.reviewer)} · ${esc(formatTime(detail.approval.created_at))}</p>
      <p>${esc(detail.approval.note)}</p>
      <p class="mini muted">审批记录 ${esc(detail.approval.approval_id)}</p>
      <p class="note">人工决定已记录；正式入库状态以“正式入库”卡片为准。</p>
    </section>`;
  }
  if (!review.eligible) {
    return `<section class="card decision-card">
      <h2>人工审批</h2>
      <p class="warning-text">${esc(review.reason || '当前候选不可审批')}</p>
      <p class="note">只有技术门禁、完整评估和模块测试段均有效，且数据快照未变化的候选才能审批。</p>
    </section>`;
  }
  const quick = latestReport(detail.evaluations, 'quick');
  const holdout = latestReport(detail.evaluations, 'holdout');
  return `<section class="card decision-card" id="decision-card">
    <h2>人工审批</h2>
    <p>请核对候选哈希和关键结论。批准只记录决定，正式入库仍由后续阶段执行。</p>
    <div class="hash-grid">
      <div><span>候选内容哈希</span><code>${esc(review.candidate_content_hash)}</code></div>
      <div><span>审批证据哈希</span><code>${esc(review.evidence_hash)}</code></div>
    </div>
    <div class="quality-summary">
      <span>技术门禁：${detail.technical_gate?.passed ? '通过' : '未通过'}</span>
      <span>研究期 RankIC：${esc(fmt(quick?.metrics?.rank_ic_mean))}</span>
      <span>RankICIR：${esc(fmt(quick?.metrics?.rank_ic_ir))}</span>
      <span>测试段：${esc(statusLabel(holdout?.status))}</span>
    </div>
    <form id="decision-form" class="decision-form">
      <label>审核人<input name="reviewer" maxlength="200" required></label>
      <label>审核备注<textarea name="note" maxlength="4000" required placeholder="批准或拒绝的具体依据"></textarea></label>
      <div class="form-actions">
        <button type="submit" data-decision="approved">批准候选</button>
        <button type="submit" class="danger" data-decision="rejected">拒绝候选</button>
      </div>
      <p id="decision-status" class="mini muted"></p>
    </form>
  </section>`;
}

function publicationSection(detail) {
  const publication = detail.publication;
  const latest = publication?.latest_batch;
  if (latest) {
    const active = ['staging', 'validating'].includes(latest.status);
    const failed = ['failed', 'rolled_back'].includes(latest.status);
    const candidates = (latest.candidates || [])
      .map(item => `${item.name}：${statusLabel(item.status)}`)
      .join('；');
    return `<section class="card publication-card">
      <h2>正式入库</h2>
      <p><span class="tag mining-status-${esc(latest.status)}">${esc(statusLabel(latest.status))}</span>
      批次 ${esc(latest.batch_id)} · 尝试 ${esc(latest.attempt)}</p>
      <p>${esc(candidates || '—')}</p>
      <div class="hash-grid">
        <div><span>发布批次 ID</span><code>${esc(latest.batch_id)}</code></div>
        <div><span>定义清单哈希</span><code>${esc(latest.manifest_hash || '构建中')}</code></div>
        <div><span>正式产物树哈希</span><code>${esc(latest.artifact_sha256 || '校验中')}</code></div>
        <div><span>错误</span><span>${esc(latest.error_code || '—')} ${esc(latest.error_message || '')}</span></div>
      </div>
      ${active ? '<p class="note">正式构建在后台执行；页面会刷新最新批次状态。</p>' : ''}
      ${failed && publication.eligible ? '<button type="button" id="publish-batch">重试整批发布</button>' : ''}
    </section>`;
  }
  if (publication?.eligible) {
    return `<section class="card publication-card">
      <h2>正式入库</h2>
      <p>候选已批准。系统会把本运行内全部已批准候选作为一个批次，先全量构建和独立校验，再原子更新 <code>mined</code> 因子族。</p>
      <p class="mini muted">本批候选数：${publication.approved_candidate_ids.length}。成功前不会修改正式因子；失败会恢复旧版本。</p>
      <button type="button" id="publish-batch">构建并发布本批</button>
    </section>`;
  }
  return `<section class="card publication-card">
    <h2>正式入库</h2>
    <p class="muted">${esc(publication?.reason || '当前候选不可发布')}</p>
  </section>`;
}

export function renderCandidate(
  app,
  detail,
  options,
  onDecision,
  onPublish,
  onRefresh,
) {
  const candidate = detail.candidate;
  const definition = candidate.definition;
  const technical = latestReport(detail.evaluations, 'technical');
  const quick = latestReport(detail.evaluations, 'quick');
  const full = latestReport(detail.evaluations, 'full');
  const holdout = latestReport(detail.evaluations, 'holdout');
  const stageRows = [
    reportRow('技术门禁', technical),
    reportRow('快速筛选', quick),
    reportRow('完整评估', full),
    reportRow('模块测试段', holdout),
  ];
  app.innerHTML = `<div class="page-heading">
    <div><h1>${esc(candidate.name)}</h1>
      <p class="intro muted">${esc(definition.description)} · ${esc(definition.mechanism)} · ${candidate.status === 'review_ready' ? '待审核' : esc(statusLabel(candidate.status))}</p>
    </div>
    <a class="button-link secondary-link" href="#/mining/${encodeURIComponent(candidate.run_id)}">返回运行</a>
  </div>
  <section class="card">
    <h2>候选定义</h2>
    <p><strong>公式</strong></p><pre>${esc(definition.canonical_expression)}</pre>
    <p><strong>白话步骤</strong></p><ol>${definition.steps.map(item => `<li>${esc(item)}</li>`).join('')}</ol>
    <p><strong>字段依赖</strong> ${definition.dependencies.map(name => `<span class="tag">${esc(name)}</span>`).join(' ')}</p>
    <div class="hash-grid">
      <div><span>候选 ID</span><code>${esc(candidate.candidate_id)}</code></div>
      <div><span>语义哈希</span><code>${esc(candidate.semantic_hash)}</code></div>
      <div><span>内容哈希</span><code>${esc(candidate.content_hash)}</code></div>
      <div><span>发现轮次</span><span>第 ${esc(candidate.discovery_round)} 轮 · 第 ${esc(candidate.discovery_index)} 个</span></div>
    </div>
  </section>
  <section class="card">
    <h2>阶段结论</h2>
    <div id="candidate-stage-summary"></div>
    <p class="note">模块测试段只用于候选冻结后的模块隔离验收，结论不反馈给 Agent，也不作为全项目盲测。</p>
  </section>
  <section class="card">
    <h2>技术门禁证据</h2>
    <div id="technical-gate"></div>
  </section>
  <section class="card">
    <h2>研究期快速筛选</h2>
    <div id="quick-evaluation"></div>
  </section>
  <section class="card">
    <h2>完整评估图表</h2>
    <div id="full-evaluation" class="candidate-stages"></div>
  </section>
  <section class="card">
    <h2>模块测试段</h2>
    <div id="holdout-evaluation"></div>
    <div class="hash-grid">
      <div><span>测试段批次哈希</span><code>${esc(detail.review.holdout_batch_hash || '—')}</code></div>
      <div><span>本批候选数</span><span>${detail.review.holdout_candidates.length}</span></div>
    </div>
  </section>
  ${publicationSection(detail)}
  ${decisionForm(detail)}
  <p class="note">数据快照：${detail.review.snapshot_current ? '当前有效' : esc(detail.review.snapshot_error?.message || '已失效')} · 内容哈希 ${esc(detail.review.candidate_content_hash)}</p>`;

  table(app.querySelector('#candidate-stage-summary'), stageRows);
  const gateNode = app.querySelector('#technical-gate');
  if (detail.technical_gate) {
    gateNode.insertAdjacentHTML('beforeend', `<p>规则版本 ${esc(detail.technical_gate.rule_version)} · ${detail.technical_gate.passed ? '通过' : '未通过'}</p>`);
    generic(gateNode, detail.technical_gate.checks);
  } else {
    gateNode.innerHTML = '<p class="muted">尚无技术门禁记录。</p>';
  }
  const quickNode = app.querySelector('#quick-evaluation');
  if (quick) {
    table(quickNode, [quick.metrics || {}]);
    if (quick.warnings?.length) {
      quickNode.insertAdjacentHTML('beforeend', `<p class="warning-text">${quick.warnings.map(esc).join('；')}</p>`);
    }
  } else {
    quickNode.innerHTML = '<p class="muted">候选未进入快速筛选。</p>';
  }
  renderStageSections(app.querySelector('#full-evaluation'), full?.metrics?.events || {});
  const holdoutNode = app.querySelector('#holdout-evaluation');
  if (holdout) {
    table(holdoutNode, [{
      结论: statusLabel(holdout.status),
      有效交易日: fmt(holdout.metrics?.valid_days),
      最低要求: fmt(holdout.metrics?.min_valid_days),
      评估完整: holdout.metrics?.complete ? '是' : '否',
    }]);
    renderStageSections(holdoutNode, holdout.metrics?.stages || {});
    if (holdout.warnings?.length) {
      holdoutNode.insertAdjacentHTML('beforeend', `<p class="warning-text">${holdout.warnings.map(esc).join('；')}</p>`);
    }
  } else {
    holdoutNode.innerHTML = '<p class="muted">尚无模块测试段报告。</p>';
  }

  const form = app.querySelector('#decision-form');
  if (form) {
    form.querySelector('[name=reviewer]').value = options.reviewer || '';
    form.addEventListener('submit', async event => {
      event.preventDefault();
      const decision = event.submitter?.dataset.decision;
      if (!decision) return;
      const data = new FormData(form);
      const note = String(data.get('note') || '').trim();
      const reviewer = String(data.get('reviewer') || '').trim();
      const status = app.querySelector('#decision-status');
      if (!note || !reviewer) {
        status.className = 'warning-text';
        status.textContent = '审核人和备注必须填写。';
        return;
      }
      const action = decision === 'approved' ? '批准' : '拒绝';
      if (!confirm(`${action}候选？\n\n内容哈希：${detail.review.candidate_content_hash}\n证据哈希：${detail.review.evidence_hash}\n\n批准仅记录决定，不会正式入库。`)) return;
      for (const button of form.querySelectorAll('button')) button.disabled = true;
      status.className = 'mini muted';
      status.textContent = `正在提交${action}决定…`;
      try {
        await onDecision(decision, reviewer, note);
      } catch (error) {
        status.className = 'warning-text';
        status.textContent = `${error.message}${error.code ? ` [${error.code}]` : ''}${error.requestId ? ` 请求 ${error.requestId}` : ''}；请刷新候选后重试。`;
        for (const button of form.querySelectorAll('button')) button.disabled = false;
      }
    });
  }

  const latest = detail.publication?.latest_batch;
  if (latest && ['staging', 'validating'].includes(latest.status)) {
    setTimeout(() => onRefresh?.(), 1000);
  }
  const publish = app.querySelector('#publish-batch');
  if (!publish) return;
  publish.addEventListener('click', async () => {
    const ids = detail.publication.approved_candidate_ids;
    if (!confirm(`整批构建并发布 ${ids.length} 个已批准候选？\n\n成功后会原子更新正式 mined 因子族。`)) return;
    publish.disabled = true;
    publish.textContent = '正在启动正式构建…';
    try {
      await onPublish(ids);
    } catch (error) {
      publish.disabled = false;
      publish.textContent = '构建并发布本批';
      app.querySelector('.publication-card').insertAdjacentHTML(
        'beforeend',
        `<p class="warning-text">${esc(error.message)} ${esc(error.code || '')}</p>`,
      );
    }
  });
}
