import {backtestChart,disposeCharts,esc,fmt,table,renderMetric} from './charts.js';
import {consumeStream} from './stream.js';

const app=document.querySelector('#app');let options,factors,controller=null;let routeVersion=0;
const labels={start:'开始日期',end:'结束日期',universe:'股票池',horizon:'标签周期',price:'价格通道',quantiles:'分位数',mode:'分析模式'};
// 回测页状态：可选参数只取一次；需要按百分比展示的绩效指标
let backtestOptions=null,preferredStrategyId=null;const selectedRuns=new Set();
const rateKeys=new Set(['total_return','annual_return','annual_volatility','max_drawdown','daily_win_rate','monthly_win_rate','benchmark_total_return','benchmark_annual_return','excess_annual_return','excess_win_rate']);
// 规则拒单原因与执行校准字段的中文名
const reasonLabels={suspended:'停牌不能交易',limit_up_or_not_buyable:'涨停或开盘不可买',limit_down_or_not_sellable:'跌停或开盘不可卖'};
const calibrationLabels={cash_buffer:'现金缓冲（执行层预留）',allocation_iterations:'资金分配迭代次数',allocation_converged:'资金分配是否收敛',carry_iterations:'现金补回的调整次数',carry_converged:'现金补回是否收敛',probe_orders:'探针成交笔数',probe_buys:'探针买入笔数',probe_sells:'探针卖出笔数',planned_orders:'计划下单格数',rebalance_orders:'其中再平衡格数',rejected_orders:'规则拒单笔数',forced_liquidations:'退市强制清算笔数',retry_attempts:'被拦交易的重试尝试次数',retried_filled_cells:'非调仓日补单成交格数',pending_at_end_cells:'区间末仍未解决的待重试格数',cash_shrunk_cells:'现金缩水行数',cash_limited_cells:'新补回事件数',cash_limited_value:'新补回事件缩水金额（元）',cash_retry_attempts:'现金补回的尝试格次',cash_retry_filled_cells:'现金补回补齐格数',cash_pending_at_end_cells:'区间末仍未补回的格数',allocated_cells:'分配出目标股数的格数',bumped_one_lot_cells:'用取整余量补到一手的格数',unaffordable_lot_cells:'一手金额超出预算、未能建仓的格数',no_price_cells:'缺执行价、未下单的格数',final_orders:'最终成交笔数',final_buys:'最终买入笔数',final_sells:'最终卖出笔数',fallback_fee_orders:'没有成交估计、按意图方向计费笔数',fee_matrix_max_diff:'费用矩阵核对最大偏差（元）',cash_clipped_cells:'被现金裁剪的格数',allocation_gap_max_shares:'实际持股与目标股数的最大差额（股）',target_cells:'目标下单格数',weight_deviation_median:'权重偏差中位数',weight_deviation_p95:'权重偏差 95 分位',weight_deviation_max:'权重偏差最大值',weight_deviation_over_1pct:'权重偏差超 1% 的格数'};
// 需要按百分比展示的执行校准字段
const calibrationRates=new Set(['cash_buffer','weight_deviation_median','weight_deviation_p95','weight_deviation_max']);

// 读取JSON接口，错误显示服务端提供的原因。
async function request(url,init){const r=await fetch(url,init);const data=await r.json();if(!r.ok)throw new Error(data.message||'接口请求失败');return data;}
async function get(url){return request(url);}

// 渲染日期、股票池、周期、分组和模式控件。
function controls(){
  const defaults=options.defaults;const choices={universe:options.universes.map(v=>[v.key,v.label]),horizon:options.horizons.map(v=>[v,v+'日']),price:[['close','收盘'],['open','开盘']],quantiles:options.quantiles.map(v=>[v,v+'组']),mode:[['full','完整分析'],['ic','轻量 IC（支持全历史）']]};
  return `<form id="evaluate-form" class="controls">${['start','end'].map(k=>`<label>${labels[k]}<input type="date" name="${k}" value="${esc(defaults[k])}" min="${options.start}" max="${options.end}" required></label>`).join('')}${Object.entries(choices).map(([k,values])=>`<label>${labels[k]}<select name="${k}">${values.map(([v,label])=>`<option value="${v}" ${String(defaults[k])===String(v)?'selected':''}>${label}</option>`).join('')}</select></label>`).join('')}<label>Pearson 去极值<select name="winsorize"><option value="true">3×MAD</option><option value="false">关闭</option></select></label><label>行业层<select name="industry"><option value="off">关闭（推荐）</option><option value="stock_basic">静态行业（研究参考）</option></select></label><label>相关 Top-N<input name="top_n" type="number" min="1" max="100" value="10" style="width:85px"></label><button type="submit">开始分析</button><button type="button" id="cancel" class="secondary" disabled>取消</button></form>`;
}

// 路由切换停止旧请求并清理图表。
async function route(){
  const version=++routeVersion;controller?.abort();controller=null;disposeCharts();
  const hash=location.hash||'#/factors';
  try{
    if(hash==='#/backtest'||hash.startsWith('#/backtest/')){
      await backtestRoute(hash,version);
    }else if(hash.startsWith('#/factor/')){
      const name=decodeURIComponent(hash.slice(9));const detail=await get('/api/factors/'+encodeURIComponent(name));if(version!==routeVersion)return;
      app.innerHTML=`<h1>${esc(name)}</h1><p class="intro muted">${esc(detail.description)} · ${esc(detail.family_label)} · ${detail.direction>0?'越大越看多':'越大越看空'}</p><details class="card"><summary>公式与计算步骤</summary><p>${esc(detail.formula)}</p><pre>${esc(detail.steps)}</pre></details>${controls()}<div id="status">选择区间后开始分析；完整分析通常需要数分钟。</div><div id="results" class="cards"></div>`;
      bindForm(()=>[name]);
    }else if(hash==='#/compare'){
      app.innerHTML=`<h1>因子对比</h1><p class="intro muted">选择 2～4 个因子，共用参数，逐个计算。各因子的有效样本可能不同。</p><div class="selection">${factors.map(f=>`<label><input type="checkbox" value="${esc(f.name)}">${esc(f.name)}</label>`).join('')}</div>${controls()}<div id="status">等待选择因子</div><div id="comparison" class="card" hidden></div><div id="results" class="cards"></div>`;
      bindForm(()=>[...app.querySelectorAll('.selection input:checked')].map(v=>v.value));
    }else{
      app.innerHTML='<h1>因子列表</h1><p class="intro muted">从覆盖率、预测能力、稳定性和风格重复程度判断因子质量。</p><div class="toolbar"><input id="search" placeholder="搜索名称、描述或因子族" aria-label="搜索因子"><span class="muted">'+factors.length+' 个已落盘因子</span></div><div id="factor-list"></div>';
      const display=()=>{const query=app.querySelector('#search').value.toLowerCase();const chosen=factors.filter(f=>[f.name,f.description,f.family_label].join(' ').toLowerCase().includes(query));
        app.querySelector('#factor-list').innerHTML='<div class="table-wrap"><table><thead><tr><th>因子</th><th>族</th><th>说明 / 公式</th><th>方向</th><th>平均覆盖率</th></tr></thead><tbody>'+chosen.map(f=>`<tr><td><a href="#/factor/${encodeURIComponent(f.name)}">${esc(f.name)}</a></td><td><span class="tag">${esc(f.family_label)}</span></td><td>${esc(f.description)}<div class="mini muted">${esc(f.formula)}</div></td><td>${f.direction>0?'看多':'看空'}</td><td>${f.avg_coverage==null?'—':(f.avg_coverage*100).toFixed(1)+'%'}</td></tr>`).join('')+'</tbody></table></div>';};
      app.querySelector('#search').addEventListener('input',display);display();
    }
  }catch(error){if(version===routeVersion)app.innerHTML=`<p role="alert">${esc(error.message)}</p><a href="#/factors">返回因子列表</a>`;}
}

// 绑定一次分析请求，保存的仅为本次页面结果，路由切换即丢弃。
function bindForm(selected){
  const form=app.querySelector('#evaluate-form'),status=app.querySelector('#status'),results=app.querySelector('#results'),cancel=app.querySelector('#cancel');
  form.addEventListener('submit',async event=>{
    event.preventDefault();const names=selected();const compare=location.hash==='#/compare';
    if(compare&&(names.length<2||names.length>4)){status.textContent='请选择2～4个不同因子';status.className='error';return;}
    controller?.abort();disposeCharts();results.innerHTML='';const local=new AbortController();controller=local;
    const query=new URLSearchParams(new FormData(form));query.set(compare?'factors':'factor',names.join(','));
    const industry=query.get('industry'),top_n=Number(query.get('top_n'));query.delete('industry');query.delete('top_n');
    if(query.get('mode')==='full')query.set('metric_options',JSON.stringify({exposure:{industry},incremental_ic:{industry},correlation:{top_n}}));
    const mode=query.get('mode'),cards=new Map(),metrics=options.metrics.filter(m=>mode==='full'||['ic_stats','ic_series','ic_dist'].includes(m.key));
    for(const name of names)for(const metric of metrics){const card=document.createElement('section');card.className='card wide';card.dataset.metric=metric.key;card.dataset.factor=name;
      card.innerHTML=`<div class="card-header"><h2>${esc(name)} · ${esc(metric.label)}</h2><span class="muted timing">等待</span></div><div class="content skeleton">等待阶段结果…</div>`;results.append(card);cards.set(name+':'+metric.key,card);}
    const summaries={};const comparison=app.querySelector('#comparison');if(comparison){comparison.hidden=true;comparison.innerHTML='';}
    const updateComparison=()=>{if(!comparison)return;comparison.hidden=false;comparison.innerHTML='<h2>同口径汇总</h2>';table(comparison,names.map(name=>({因子:name,...summaries[name]})));};
    status.className='';status.textContent='请求已发出，等待计算名额…';cancel.disabled=false;form.querySelector('[type=submit]').disabled=true;
    try{
      await consumeStream('/api/'+(compare?'compare':'evaluate')+'/stream?'+query,local.signal,(stage,payload)=>{
        if(stage==='queued'){status.textContent='已进入任务队列，运行时将逐阶段显示结果';return;}
        if(stage==='meta'){status.textContent='正在分析 '+payload.factor+' · '+(mode==='ic'?'轻量 IC':'完整分析');return;}
        if(stage==='error')throw new Error(`${payload.message} [${payload.code}] 请求 ${payload.request_id}`);
        if(stage==='done'){status.textContent=`分析完成 · 总耗时 ${(payload.total_elapsed_ms/1000).toFixed(1)} 秒`;return;}
        if(stage==='factor_done'){status.textContent=payload.factor+' 已完成';return;}
        for(const [key,data] of Object.entries(payload.metrics||{})){
          let card=cards.get(payload.factor+':'+key);
          if(!card){card=document.createElement('section');card.className='card wide';card.innerHTML=`<h2>${esc(payload.factor)} · ${esc(key)}</h2><div class="content"></div>`;results.append(card);}
          const content=card.querySelector('.content');content.className='content';content.innerHTML='';renderMetric(content,key,data);
          const timing=card.querySelector('.timing');if(timing)timing.textContent='阶段 '+(payload.elapsed_ms/1000).toFixed(2)+' 秒';
          summaries[payload.factor]??={};const row=summaries[payload.factor];
          if(key==='ic_stats')Object.assign(row,{IC:data.ic.mean,ICIR:data.ic.ir,RankIC:data.rank_ic.mean,RankICIR:data.rank_ic.ir,有效日:data.valid_days});
          if(key==='spread')Object.assign(row,{等权多空:data.equal?.total_return,市值多空:data.market_cap?.total_return});
          if(key==='incremental_ic')row.残差RankIC=data.residual.mean;
          updateComparison();
        }
        status.textContent=payload.factor+' · '+stage+' 已完成';
      });
    }catch(error){status.className=error.name==='AbortError'?'':'error';status.textContent=error.name==='AbortError'?'已取消；运行中的阶段将在安全边界停止':error.message;
      for(const card of cards.values()){const pending=card.querySelector('.skeleton');if(pending)pending.textContent=error.name==='AbortError'?'已取消':'本阶段未完成';}
    }finally{local.abort();cancel.disabled=true;form.querySelector('[type=submit]').disabled=false;if(controller===local)controller=null;}
  });
  cancel.addEventListener('click',()=>controller?.abort());
}

window.addEventListener('hashchange',route);
try{[options,factors]=await Promise.all([get('/api/options'),get('/api/factors')]);await route();}
catch(error){app.innerHTML=`<p role="alert">${esc(error.message)}</p>`;}

// ===== 回测：列表页、详情页与历史运行 =====
// 回测路由：运行详情页或列表页。
async function backtestRoute(hash,version){
  if(hash==='#/backtest/new')return strategyPage(version);
  if(hash.startsWith('#/backtest/compare'))return compareRunsPage(hash,version);
  if(hash.startsWith('#/backtest/'))return runPage(decodeURIComponent(hash.slice('#/backtest/'.length)),version);
  return backtestPage(version);
}

// 取回测可选项，只在首次进入回测页时请求。
async function loadBacktestOptions(){if(!backtestOptions)backtestOptions=await get('/api/backtest/options');return backtestOptions;}

// 回测页：选策略、股票池、区间与资金，用 SSE 看阶段进度。
async function backtestPage(version){
  const opts=await loadBacktestOptions();if(version!==routeVersion)return;
  const runs=await get('/api/backtests');if(version!==routeVersion)return;
  const d=opts.defaults;
  app.innerHTML=`<div class="page-heading"><div><h1>策略回测</h1><p class="intro muted">VectorBT 负责组合执行与账本，项目负责 A 股规则与成本。目标权重在执行日按真实现金与持仓分配成 100 股整数倍的目标股数，并预留现金缓冲；被规则拦下或受现金限制的订单会按交易日继续补回。</p></div><a class="button-link" href="#/backtest/new">新建策略</a></div>
  <form id="backtest-form" class="controls">
    <label>策略<select name="strategy_id">${opts.strategies.map(row=>`<option value="${esc(row.strategy_id)}" ${row.strategy_id===preferredStrategyId?'selected':''}>${esc(row.name)} · v${row.current_version}</option>`).join('')}</select></label>
    <label>股票池<select name="universe">${opts.universes.map(item=>`<option value="${esc(item.key)}">${esc(item.label)}</option>`).join('')}</select></label>
    <label>基准<select name="benchmark">${opts.benchmarks.map(item=>`<option value="${esc(item.code)}" ${item.code===d.benchmark?'selected':''}>${esc(item.label)}</option>`).join('')}</select></label>
    <label>开始日期<input type="date" name="start" value="${esc(d.start)}" required></label>
    <label>结束日期<input type="date" name="end" value="${esc(d.end)}" required></label>
    <label>初始资金<input type="number" name="init_cash" value="${d.init_cash}" min="10000" step="10000" required></label>
    <button type="submit">运行回测</button>
    <button type="button" id="backtest-cancel" class="secondary" disabled>取消</button>
  </form>
  <div id="backtest-status" class="muted">默认成本：佣金 ${opts.costs.commission_rate}、最低佣金 ${opts.costs.minimum_commission} 元、卖出印花税 ${opts.costs.stamp_tax_rate}、滑点 ${opts.costs.slippage}。上限 ${opts.limits.trade_days} 个交易日 / ${(opts.limits.panel_cells/1e6).toFixed(0)} 百万单元格；十年全 A 约需三到四分钟，回测默认串行执行。</div>
  <section class="card"><div class="card-header"><h2>历史运行</h2><button type="button" id="compare-runs" class="secondary" disabled>对比所选运行</button></div><div id="run-list"></div></section>`;
  renderRuns(runs);
  preferredStrategyId=null;
  const form=app.querySelector('#backtest-form'),status=app.querySelector('#backtest-status'),cancel=app.querySelector('#backtest-cancel');
  form.addEventListener('submit',async event=>{
    event.preventDefault();const local=new AbortController();controller=local;
    const query=new URLSearchParams(new FormData(form));
    form.querySelector('[type=submit]').disabled=true;cancel.disabled=false;status.className='muted';
    status.textContent='已提交，等待运行名额…';
    try{
      await consumeStream('/api/backtests/stream?'+query,local.signal,(stage,payload)=>{
        if(stage==='queued')return void(status.textContent='已进入回测队列，等待运行名额…');
        if(stage==='error')throw new Error(`${payload.message} [${payload.code}] 请求 ${payload.request_id}`);
        if(stage==='meta')return void(status.textContent=`策略 ${payload.name} v${payload.version} · 池 ${payload.universe} · 基准 ${payload.benchmark||'无'} · 因子 ${payload.factors.join(' + ')}`);
        if(stage==='load')return void(status.textContent=`数据就绪：${payload.trading_days} 个交易日 × ${payload.stocks} 只股票`);
        if(stage==='signals')return void(status.textContent=`信号完成：${payload.rebalance_days} 个调仓日`);
        if(stage==='probe')return void(status.textContent=`探针运行完成：${payload.probe_orders} 笔成交，开始规则过滤与费用校准`);
        if(stage==='orders')return void(status.textContent=`规则过滤完成：计划 ${payload.planned_orders} 笔，规则拒单 ${payload.rejected_orders} 笔`);
        if(stage==='baseline')return void(status.textContent=`基准遍完成：执行层预留 ${pct(payload.cash_buffer)} 现金，开始资金分配`);
        if(stage==='allocation')return void(status.textContent=`资金分配完成：建仓 ${payload.allocated_cells} 格，补一手 ${payload.bumped_one_lot_cells} 格，未建仓 ${payload.unaffordable_lot_cells} 格，被现金缩小 ${payload.cash_shrunk_cells} 格（其中 ${payload.cash_limited_cells} 格待补回）`);
        if(stage==='execute')return void(status.textContent=`最终运行完成：${payload.final_orders} 笔成交，费用矩阵偏差 ${fmt(payload.fee_matrix_max_diff)}`);
        if(stage==='done'){status.innerHTML=`回测完成，用时 ${fmt(payload.elapsed_seconds)} 秒，成交 ${payload.orders} 笔，规则拒单 ${payload.rejections} 笔。<a href="#/backtest/${encodeURIComponent(payload.run_id)}">查看结果</a>`;loadRuns();}
      });
    }catch(error){
      status.className=error.name==='AbortError'?'muted':'error';
      status.textContent=error.name==='AbortError'?'已取消；运行中的阶段会在安全边界停止':error.message;
    }finally{local.abort();cancel.disabled=true;form.querySelector('[type=submit]').disabled=false;if(controller===local)controller=null;}
  });
  cancel.addEventListener('click',()=>controller?.abort());
}

// 刷新历史运行列表。
async function loadRuns(){
  const node=app.querySelector('#run-list');if(!node)return;
  try{renderRuns(await get('/api/backtests'));}
  catch(error){node.innerHTML=`<p role="alert">${esc(error.message)}</p>`;}
}

// 渲染历史运行表，运行编号可点进详情。
function renderRuns(runs){
  const node=app.querySelector('#run-list');if(!node)return;
  const compare=app.querySelector('#compare-runs');
  if(!runs.length){selectedRuns.clear();if(compare)compare.disabled=true;node.innerHTML='<p class="muted">还没有运行记录，先在上面选好参数运行一次。</p>';return;}
  const eligible=new Set(runs.filter(row=>row.status==='success').map(row=>row.run_id));
  for(const runId of [...selectedRuns])if(!eligible.has(runId))selectedRuns.delete(runId);
  const cell=value=>value===null||value===undefined?'—':pct(value);
  node.innerHTML='<div class="table-wrap"><table><thead><tr><th>选择</th><th>运行</th><th>策略版本</th><th>区间</th><th>股票池</th><th>基准</th><th>状态</th><th>累计收益</th><th>年化</th><th>最大回撤</th><th>耗时（秒）</th><th>操作</th></tr></thead><tbody>'+
    runs.map(row=>`<tr><td><input class="run-choice" type="checkbox" value="${esc(row.run_id)}" aria-label="选择运行 ${esc(String(row.run_id).slice(0,8))}" ${row.status==='success'?'':'disabled'} ${selectedRuns.has(row.run_id)?'checked':''}></td><td><a href="#/backtest/${encodeURIComponent(row.run_id)}">${esc(String(row.run_id).slice(0,8))}</a></td><td>${esc(row.strategy_id)} · v${esc(row.strategy_version)}</td><td>${esc(String(row.start_date).slice(0,10))} ~ ${esc(String(row.end_date).slice(0,10))}</td><td>${esc(row.universe||'—')}</td><td>${esc(row.benchmark||'—')}</td><td>${esc(row.status)}</td>`+
      ['total_return','annual_return','max_drawdown'].map(key=>`<td>${cell(row.metrics?row.metrics[key]:null)}</td>`).join('')+
      `<td>${row.elapsed_seconds==null?'—':fmt(row.elapsed_seconds)}</td><td><button type="button" class="table-command delete-run" data-run="${esc(row.run_id)}" ${row.status==='running'?'disabled':''}>删除</button></td></tr>`).join('')+'</tbody></table></div>';
  const syncCompare=()=>{if(compare){compare.disabled=selectedRuns.size<2||selectedRuns.size>4;compare.textContent=selectedRuns.size?`对比所选运行（${selectedRuns.size}）`:'对比所选运行';}};
  node.querySelectorAll('.run-choice').forEach(input=>input.addEventListener('change',()=>{
    const status=app.querySelector('#backtest-status');
    if(input.checked&&selectedRuns.size>=4){input.checked=false;if(status){status.className='error';status.textContent='最多同时对比 4 次运行';}return;}
    if(input.checked)selectedRuns.add(input.value);else selectedRuns.delete(input.value);syncCompare();
  }));
  node.querySelectorAll('.delete-run').forEach(button=>button.addEventListener('click',async()=>{
    const runId=button.dataset.run;if(!confirm(`确定删除运行 ${runId.slice(0,8)} 及其全部结果？此操作不可恢复。`))return;
    button.disabled=true;
    try{await request('/api/backtests/'+encodeURIComponent(runId),{method:'DELETE'});selectedRuns.delete(runId);await loadRuns();}
    catch(error){button.disabled=false;const status=app.querySelector('#backtest-status');if(status){status.className='error';status.textContent=error.message;}}
  }));
  if(compare&&!compare.dataset.bound){compare.dataset.bound='true';compare.addEventListener('click',()=>{if(selectedRuns.size>=2&&selectedRuns.size<=4)location.hash='#/backtest/compare?run_ids='+[...selectedRuns].map(encodeURIComponent).join(',');});}
  syncCompare();
}

// 多运行对比页：服务端已按共同交易日完成归一和累计超额计算。
async function compareRunsPage(hash,version){
  const params=new URLSearchParams(hash.split('?')[1]||'');
  const runIds=(params.get('run_ids')||'').split(',').map(item=>item.trim()).filter(Boolean);
  const result=await get('/api/backtests/compare?run_ids='+runIds.map(encodeURIComponent).join(','));if(version!==routeVersion)return;
  const metricKeys=[['total_return','累计收益'],['annual_return','年化收益'],['max_drawdown','最大回撤'],['excess_annual_return','年化超额'],['information_ratio','信息比率']];
  app.innerHTML=`<div class="page-heading"><div><h1>运行对比</h1><p class="intro muted">共同交易日 ${esc(result.start)} ~ ${esc(result.end)}；策略净值统一从 1 开始，累计超额统一从 0 开始。</p></div><a class="button-link secondary-link" href="#/backtest">返回回测</a></div>
  <section class="card"><h2>运行与指标</h2><div id="compare-summary"></div></section>
  <section class="card"><h2>共同区间曲线</h2><div class="cards" id="compare-curves"></div></section>`;
  table(app.querySelector('#compare-summary'),result.runs.map(row=>{
    const record={运行:String(row.run_id).slice(0,8),策略:`${row.strategy_name} · v${row.strategy_version}`,原区间:`${String(row.start_date).slice(0,10)} ~ ${String(row.end_date).slice(0,10)}`,股票池:row.universe,基准:row.benchmark};
    for(const [key,label] of metricKeys)record[label]=key==='information_ratio'?fmt(row.metrics[key]):pct(row.metrics[key]);
    return record;
  }));
  const named=row=>`${row.strategy_name} · ${String(row.run_id).slice(0,8)}`;
  backtestChart(app.querySelector('#compare-curves'),result.dates,result.runs.map(row=>({name:named(row),data:row.nav})),'策略净值（共同首日 = 1）');
  backtestChart(app.querySelector('#compare-curves'),result.dates,result.runs.map(row=>({name:named(row),data:row.cumulative_excess})),'累计超额（共同首日 = 0）');
}

// 新建策略页：把当前 StrategySpec 支持的配置保存为不可变第一版。
async function strategyPage(version){
  const opts=await loadBacktestOptions();if(version!==routeVersion)return;
  const factorRows=opts.factors.map(item=>`<tr><td><input class="factor-enabled" type="checkbox" value="${esc(item.name)}" ${item.name==='MOM_20'?'checked':''} aria-label="选择因子 ${esc(item.name)}"></td><td>${esc(item.name)}</td><td>${item.direction>0?'越大越看多':'越大越看空'}</td><td>${esc(item.description||'')}</td><td><input class="factor-weight" type="number" value="${item.name==='MOM_20'?'1':'0'}" step="0.1" ${item.name==='MOM_20'?'':'disabled'} aria-label="${esc(item.name)} 权重"></td></tr>`).join('');
  app.innerHTML=`<div class="page-heading"><div><h1>新建策略</h1><p class="intro muted">保存后生成不可变的策略第一版，并与内置策略共用同一条回测路径。</p></div><a class="button-link secondary-link" href="#/backtest">返回回测</a></div>
  <form id="strategy-form">
    <section class="card"><h2>基本信息</h2><div class="form-grid">
      <label>策略名称<input name="name" required maxlength="80" value="自定义多因子策略"></label>
      <label>说明<textarea name="description" rows="3" maxlength="500"></textarea></label>
      <label>股票池<select name="universe">${opts.universes.map(item=>`<option value="${esc(item.key)}">${esc(item.label)}</option>`).join('')}</select></label>
      <label>基准<select name="benchmark">${opts.benchmarks.map(item=>`<option value="${esc(item.code)}" ${item.code===opts.defaults.benchmark?'selected':''}>${esc(item.label)}</option>`).join('')}</select></label>
    </div></section>
    <section class="card"><h2>因子信号</h2><div class="table-wrap"><table class="factor-table"><thead><tr><th>选择</th><th>因子</th><th>方向</th><th>说明</th><th>权重</th></tr></thead><tbody>${factorRows}</tbody></table></div><div class="toggle-row"><label><input type="checkbox" name="standardize" checked>截面标准化</label><label><input type="checkbox" name="winsorize">3×MAD 去极值</label></div></section>
    <section class="card"><h2>组合与调仓</h2><div class="form-grid">
      <label>持仓只数<input name="top_n" type="number" min="1" step="1" value="30" required></label>
      <label>权重方式<select name="weighting"><option value="equal">等权</option><option value="market_cap">市值加权</option></select></label>
      <label>调仓频率<select name="frequency"><option value="daily">日频</option><option value="weekly">周频</option><option value="monthly" selected>月频</option></select></label>
    </div><p class="mini muted">固定口径：Top-N 选股、只做多、收盘生成信号、下一交易日开盘执行。</p></section>
    <section class="card"><h2>执行规则</h2><div class="form-grid">
      <label>现金缓冲<input name="cash_buffer" type="number" min="0" max="0.499999" step="0.005" value="0.02" required></label>
      <label>交易单位<input value="100 股" readonly></label>
    </div><div class="toggle-row"><label><input type="checkbox" name="reject_suspended" checked>停牌拒单</label><label><input type="checkbox" name="reject_limit_up_buy" checked>涨停买入拒单</label><label><input type="checkbox" name="reject_limit_down_sell" checked>跌停卖出拒单</label><label><input type="checkbox" name="enforce_t_plus_1" checked>T+1</label></div></section>
    <section class="card"><h2>交易成本</h2><div class="form-grid">
      <label>佣金率<input name="commission_rate" type="number" min="0" step="0.0001" value="${opts.costs.commission_rate}" required></label>
      <label>最低佣金（元）<input name="minimum_commission" type="number" min="0" step="0.1" value="${opts.costs.minimum_commission}" required></label>
      <label>卖出印花税率<input name="stamp_tax_rate" type="number" min="0" step="0.0001" value="${opts.costs.stamp_tax_rate}" required></label>
      <label>滑点<input name="slippage" type="number" min="0" step="0.0001" value="${opts.costs.slippage}" required></label>
    </div></section>
    <div id="strategy-status" class="muted">至少选择一个因子，权重不能全部为 0。</div>
    <div class="form-actions"><button type="submit">创建策略</button><a class="button-link secondary-link" href="#/backtest">取消</a></div>
  </form>`;
  const form=app.querySelector('#strategy-form'),status=app.querySelector('#strategy-status');
  form.querySelectorAll('.factor-enabled').forEach(input=>input.addEventListener('change',()=>{const weight=input.closest('tr').querySelector('.factor-weight');weight.disabled=!input.checked;if(input.checked&&Number(weight.value)===0)weight.value='1';}));
  form.addEventListener('submit',async event=>{
    event.preventDefault();status.className='muted';
    const data=new FormData(form);const chosen=[...form.querySelectorAll('.factor-enabled:checked')];
    const factorNames=chosen.map(input=>input.value);const factorWeights=chosen.map(input=>Number(input.closest('tr').querySelector('.factor-weight').value));
    const topN=Number(data.get('top_n')),cashBuffer=Number(data.get('cash_buffer'));
    const costs=['commission_rate','minimum_commission','stamp_tax_rate','slippage'].map(name=>Number(data.get(name)));
    if(!String(data.get('name')).trim())return void(status.className='error',status.textContent='策略名称不能为空');
    if(!factorNames.length)return void(status.className='error',status.textContent='至少选择一个因子');
    if(factorWeights.some(value=>!Number.isFinite(value))||factorWeights.every(value=>value===0))return void(status.className='error',status.textContent='因子权重必须是数字且不能全部为 0');
    if(!Number.isInteger(topN)||topN<=0)return void(status.className='error',status.textContent='持仓只数必须是正整数');
    if(!Number.isFinite(cashBuffer)||cashBuffer<0||cashBuffer>=0.5)return void(status.className='error',status.textContent='现金缓冲必须在 0 到 0.5 之间');
    if(costs.some(value=>!Number.isFinite(value)||value<0))return void(status.className='error',status.textContent='交易成本必须是非负数');
    const payload={description:String(data.get('description')||''),spec:{name:String(data.get('name')).trim(),universe:data.get('universe'),signal:{factors:factorNames,weights:factorWeights,winsorize:data.has('winsorize'),standardize:data.has('standardize')},portfolio:{selection:'top_n',top_n:topN,weighting:data.get('weighting'),long_only:true},rebalance:{frequency:data.get('frequency'),signal_at:'close',execute_at:'next_open'},execution:{lot_size:100,cash_buffer:cashBuffer,reject_suspended:data.has('reject_suspended'),reject_limit_up_buy:data.has('reject_limit_up_buy'),reject_limit_down_sell:data.has('reject_limit_down_sell'),enforce_t_plus_1:data.has('enforce_t_plus_1')},costs:{commission_rate:costs[0],minimum_commission:costs[1],stamp_tax_rate:costs[2],slippage:costs[3]},benchmark:{index_code:data.get('benchmark')}}};
    const submit=form.querySelector('[type=submit]');submit.disabled=true;status.textContent='正在创建策略…';
    try{const created=await request('/api/strategies',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});preferredStrategyId=created.strategy_id;backtestOptions=null;location.hash='#/backtest';}
    catch(error){status.className='error';status.textContent=error.message;submit.disabled=false;}
  });
}

// 百分比展示，无效值显示破折号。
function pct(value){const number=Number(value);return Number.isFinite(number)?(number*100).toFixed(2)+'%':'—';}

// 运行详情页：绩效、曲线与明细表。
async function runPage(runId,version){
  const base='/api/backtests/'+encodeURIComponent(runId);
  const run=await get(base);if(version!==routeVersion)return;
  if(run.status!=='success'){
    app.innerHTML=`<h1>回测结果</h1>
    <p class="intro muted">运行 ${esc(runId)} · 状态 ${esc(run.status)} · 区间 ${esc(String(run.start_date).slice(0,10))} ~ ${esc(String(run.end_date).slice(0,10))}</p>
    <section class="card"><h2>运行未完成</h2><p role="alert">${esc(run.error_message||'运行尚未结束或已被取消，没有可展示的日结果。')}</p><a href="#/backtest">返回回测页</a></section>`;
    return;
  }
  const [metrics,daily,holding,orders,rejections,liquidations]=await Promise.all([
    get(base+'/metrics'),get(base+'/daily'),get(base+'/holding'),get(base+'/order'),get(base+'/rejection'),
    get(base+'/liquidation')
  ]);
  if(version!==routeVersion)return;
  const metadata=run.metadata||{};const params=metadata.params||{};
  const dates=daily.map(row=>String(row.trade_date).slice(0,10));
  const column=key=>daily.map(row=>Number.isFinite(Number(row[key]))?Number(row[key]):null);
  const series=[{name:'策略净值',data:column('nav')}];
  if(metadata.benchmark)series.push({name:'基准净值',data:column('benchmark_nav')});
  const reasonCounts={};for(const row of rejections)reasonCounts[row.reason]=(reasonCounts[row.reason]||0)+1;
  const lastDate=holding.length?holding[holding.length-1].trade_date:null;
  const lastHoldings=holding.filter(row=>row.trade_date===lastDate);
  app.innerHTML=`<h1>回测结果</h1>
  <p class="intro muted">运行 ${esc(runId)} · 状态 ${esc(run.status)} · 区间 ${esc(params.start||'')} ~ ${esc(params.end||'')} · 股票池 ${esc(metadata.universe||'')}</p>
  <section class="card"><h2>运行信息</h2><div class="cards" id="run-info"></div></section>
  <section class="card"><h2>绩效汇总</h2><div class="cards" id="run-metrics"></div></section>
  <section class="card"><h2>净值与回撤</h2><div class="cards" id="run-curves"></div></section>
  <section class="card"><h2>换手与成本</h2><div class="cards" id="run-cost"></div></section>
  <section class="card"><h2>规则拒单</h2><div class="cards" id="run-rejection"></div></section>
  <section class="card"><h2>退市强制清算</h2><div class="cards" id="run-liquidation"></div></section>
  <section class="card"><h2>执行校准</h2><div class="cards" id="run-calibration"></div></section>
  <section class="card"><h2>成交明细${orders.length>100?'（最近 100 笔）':''}</h2><div class="cards" id="run-orders"></div></section>
  <section class="card"><h2>最新持仓 ${esc(lastDate?String(lastDate).slice(0,10):'')}</h2><div class="cards" id="run-holding"></div></section>`;
  table(app.querySelector('#run-info'),[{
    策略:metadata.strategy?metadata.strategy.name:run.strategy_id,版本:run.strategy_version,
    策略哈希:metadata.strategy_hash,股票池:metadata.universe,
    基准:metadata.benchmark?`${metadata.benchmark.name}（${metadata.benchmark.return_type==='total_return'?'全收益':'价格'}）`:'无',
    执行口径:metadata.execution?`目标股数 · 现金缓冲 ${pct(metadata.execution.cash_buffer)}`:'目标百分比（旧口径）',
    起始:params.start,截止:params.end,初始资金:params.init_cash,
    交易日:metadata.panel?metadata.panel.trading_days:null,
    股票数:metadata.panel?metadata.panel.stocks:null,
    订单数:orders.length,拒单数:rejections.length,耗时秒:metadata.elapsed_seconds
  }]);
  const labels={total_return:'累计收益',annual_return:'年化收益',annual_volatility:'年化波动',sharpe:'夏普',max_drawdown:'最大回撤',calmar:'卡玛',daily_win_rate:'日胜率',monthly_win_rate:'月胜率',turnover_annual:'年化换手',total_cost:'累计成本',benchmark_total_return:'基准累计',benchmark_annual_return:'基准年化',excess_annual_return:'年化超额',tracking_error:'跟踪误差',information_ratio:'信息比率',excess_win_rate:'超额胜率',trading_days:'交易日',months:'月数'};
  const shown=['total_return','annual_return','annual_volatility','sharpe','max_drawdown','calmar','daily_win_rate','monthly_win_rate','turnover_annual','total_cost','benchmark_annual_return','excess_annual_return','information_ratio','excess_win_rate'];
  table(app.querySelector('#run-metrics'),shown.filter(key=>key in metrics).map(key=>({指标:labels[key],数值:rateKeys.has(key)?pct(metrics[key]):metrics[key]})));
  backtestChart(app.querySelector('#run-curves'),dates,series,'净值（起始 = 1）');
  backtestChart(app.querySelector('#run-curves'),dates,[{name:'回撤',data:column('drawdown'),area:true}],'回撤');
  if(daily.some(row=>row.excess_return!==null&&row.excess_return!==undefined))backtestChart(app.querySelector('#run-curves'),dates,[{name:'日超额收益',data:column('excess_return'),bar:true}],'日超额收益');
  backtestChart(app.querySelector('#run-cost'),dates,[{name:'换手（占前一日组合价值）',data:column('turnover'),bar:true}],'日换手');
  backtestChart(app.querySelector('#run-cost'),dates,[{name:'当日交易成本（元）',data:column('cost'),bar:true}],'交易成本');
  const reasonRows=Object.entries(reasonCounts).map(([reason,count])=>({拒单原因:reasonLabels[reason]||reason,笔数:count}));
  if(reasonRows.length)table(app.querySelector('#run-rejection'),reasonRows);
  else app.querySelector('#run-rejection').innerHTML='<p class="muted">本次运行没有规则拒单</p>';
  if(liquidations.length)table(app.querySelector('#run-liquidation'),liquidations.map(row=>({日期:String(row.trade_date).slice(0,10),股票:row.stock_code,退市日:String(row.delist_date||'').slice(0,10),股数:row.size,价格:row.price,金额:row.value})));
  else app.querySelector('#run-liquidation').innerHTML='<p class="muted">本次运行没有强制清算</p>';
  const calibration=metadata.calibration||{};
  const calibrationValue=key=>(key==='allocation_converged'||key==='carry_converged')?(calibration[key]?'是':'否'):(calibrationRates.has(key)?pct(calibration[key]):fmt(calibration[key]));
  table(app.querySelector('#run-calibration'),Object.keys(calibrationLabels).filter(key=>key in calibration).map(key=>({指标:calibrationLabels[key],数值:calibrationValue(key)})));
  if(orders.length)table(app.querySelector('#run-orders'),orders.slice(-100).map(row=>({日期:String(row.trade_date).slice(0,10),股票:row.stock_code,方向:row.Side,股数:row.Size,价格:row.Price,费用:row.Fees})));
  else app.querySelector('#run-orders').innerHTML='<p class="muted">本次运行没有成交</p>';
  if(lastHoldings.length)table(app.querySelector('#run-holding'),lastHoldings.map(row=>({股票:row.stock_code,市值:row.market_value,权重:pct(row.weight)})));
  else app.querySelector('#run-holding').innerHTML='<p class="muted">期末没有持仓</p>';
}

