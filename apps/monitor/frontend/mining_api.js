// 挖掘 API 与可续传 SSE；只在浏览器内处理和显示服务端已保存结果。
export class MiningRequestError extends Error {
  constructor(message, code, requestId) {
    super(message);
    this.name = 'MiningRequestError';
    this.code = code;
    this.requestId = requestId;
  }
}

export async function miningRequest(url, init = {}) {
  const response = await fetch(url, init);
  const data = await response.json();
  if (!response.ok) {
    throw new MiningRequestError(
      data.message || '接口请求失败',
      data.code || 'UNKNOWN',
      data.request_id || '',
    );
  }
  return data;
}

export function miningJSON(url, body, method = 'POST') {
  return miningRequest(url, {
    method,
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
}

function parseEventBlock(block) {
  let event = 'message';
  let sequence = null;
  const data = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    else if (line.startsWith('id:')) sequence = Number(line.slice(3).trim());
  }
  if (!data.length) return null;
  return {event, sequence, data: JSON.parse(data.join('\n'))};
}

export async function consumeMiningStream(url, signal, onEvent, onSequence) {
  const response = await fetch(url, {signal});
  if (!response.ok) {
    const error = await response.json();
    throw new MiningRequestError(
      error.message || '事件流连接失败',
      error.code || 'UNKNOWN',
      error.request_id || '',
    );
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let terminal = false;
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseEventBlock(block);
        if (!parsed) continue;
        if (Number.isInteger(parsed.sequence)) onSequence(parsed.sequence);
        onEvent(parsed.event, parsed.data);
        if (parsed.event === 'done' || parsed.event === 'error') {
          terminal = true;
        }
      }
      if (done) break;
      if (terminal) break;
    }
  } finally {
    reader.releaseLock();
  }
  if (!terminal) throw new Error('事件流提前结束，准备从最后序号重连');
}

export function formatTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', {hour12: false});
}

export function statusLabel(status) {
  return {
    queued: '排队',
    generating: '生成候选',
    screening: '快速筛选',
    evaluating: '完整评估',
    holding_out: '模块测试段',
    review_ready: '待人工审核',
    completed: '已完成',
    interrupted: '已中断，可恢复',
    failed: '失败',
    cancelled: '已取消',
    stale: '证据已过期',
    proposed: '已提出',
    validating: '技术校验',
    technical_failed: '技术门禁失败',
    screened_out: '快速筛选未入围',
    evaluated: '完整评估完成',
    holdout_running: '执行测试段',
    holdout_failed: '测试段未通过',
    approved: '已批准',
    rejected: '已拒绝',
    building: '正式构建中',
    published: '已发布',
    staging: '发布暂存',
    validating: '发布校验中',
    rolled_back: '已回滚',
    build_failed: '正式构建失败',
  }[status] || status || '—';
}
