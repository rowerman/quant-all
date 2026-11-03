// 图表只消费服务端结果；不在页面重算收益、IC或回归。
const palette = ['#635bff','#0a2540','#0f9fae','#7a73ff','#b7791f','#64748b'];
const instances = new Set();
const columnLabels={mean:'均值',std:'样本标准差',stderr:'标准误',ir:'ICIR',t:'t 值',win_rate:'正值比例',days:'有效日',horizon:'周期（日）',factor:'因子',correlation:'相关性',key:'风格',positive_share:'系数为正比例',significant_share:'显著占比',bin:'分位',count:'样本数',min:'最小值',max:'最大值',p05:'5%分位',p25:'25%分位',p50:'中位数',p75:'75%分位',p95:'95%分位',annual_sharpe:'年化夏普',total_return:'累计收益',max_drawdown:'最大回撤',rank_ic_mean:'RankIC均值',rank_ic_ir:'RankICIR',samples:'配对数',insufficient:'样本不足',column:'标签列',label:'标签'};
export const fmt = value => value === null || value === undefined ? '—' : typeof value === 'number' ? Number(value.toPrecision(5)).toLocaleString('en-US',{maximumFractionDigits:6}) : String(value);
export const esc = text => String(text ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

// 清理路由离开时的图表，避免事件和画布累积。
export function disposeCharts(){for(const chart of instances)chart.dispose();instances.clear();}
window.addEventListener('resize',()=>{for(const chart of instances)chart.resize();});

// 在容器内添加图表，option为ECharts配置。
function chart(container, option, title='', heatmap=false){
  const wrap=document.createElement('div');if(title){const h=document.createElement('h3');h.textContent=title;wrap.append(h);}
  const node=document.createElement('div');node.className='chart'+(heatmap?' heatmap':'');wrap.append(node);container.append(wrap);
  if(!window.echarts){node.textContent='图表库未加载，请检查本地静态资源';return;}
  const instance=echarts.init(node);instances.add(instance);
  const axisTheme={axisLine:{lineStyle:{color:'#d7e0ea'}},axisTick:{lineStyle:{color:'#d7e0ea'}},axisLabel:{color:'#52677d',fontSize:11},splitLine:{lineStyle:{color:'#edf1f6'}}};
  const withAxisTheme=value=>Array.isArray(value)?value.map(axis=>({...axisTheme,...axis})):{...axisTheme,...value};
  const merged={...option};
  merged.xAxis=withAxisTheme(option.xAxis);
  merged.yAxis=Array.isArray(option.yAxis)?option.yAxis.map(axis=>({...axisTheme,...axis})):{...axisTheme,...option.yAxis};
  Object.assign(merged,{
    color:palette,
    animation:false,
    backgroundColor:'transparent',
    textStyle:{fontFamily:'"Avenir Next","Noto Sans CJK SC","Source Han Sans SC","PingFang SC",sans-serif',color:'#425466'},
    tooltip:{trigger:'axis',backgroundColor:'#ffffff',borderColor:'#e3e8ee',borderWidth:1,padding:[10,12],textStyle:{color:'#0a2540',fontSize:12},extraCssText:'box-shadow:0 8px 24px rgba(10,37,64,.12)',...(option.tooltip||{})},
    legend:{type:'scroll',top:0,textStyle:{color:'#52677d'},...(option.legend||{})},
    grid:{left:65,right:25,top:50,bottom:65,...(option.grid||{})},
  });
  instance.setOption(merged);
}

// 创建标量/记录表，rows为记录序列。
export function table(container, rows, columns=null){
  if(!rows?.length){container.insertAdjacentHTML('beforeend','<p class="muted">没有可展示样本</p>');return;}
  const keys=columns||[...new Set(rows.flatMap(row=>Object.keys(row)))];
  const wrap=document.createElement('div');wrap.className='table-wrap';
  wrap.innerHTML='<table><thead><tr>'+keys.map(k=>`<th>${esc(columnLabels[k]||k)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+keys.map(k=>`<td>${esc(fmt(row[k]))}</td>`).join('')+'</tr>').join('')+'</tbody></table>';container.append(wrap);
}

// 保留陌生指标的全部数据，嵌套结构通过可展开表格展示。
export function generic(container, data){
  if(Array.isArray(data)){if(data.every(v=>v&&typeof v==='object'&&!Array.isArray(v)))table(container,data);else table(container,data.map((v,i)=>({序号:i+1,值:v})));return;}
  if(data&&typeof data==='object'){
    const scalars=Object.entries(data).filter(([,v])=>v===null||typeof v!=='object');
    if(scalars.length)table(container,scalars.map(([k,v])=>({字段:k,值:v})));
    for(const [key,value] of Object.entries(data))if(value&&typeof value==='object'){
      const detail=document.createElement('details');detail.innerHTML=`<summary>${esc(key)}</summary>`;container.append(detail);generic(detail,value);
    }
  }
}

const line = (name,data) => ({name,type:'line',data,showSymbol:false,connectNulls:false});
const bar = (name,data) => ({name,type:'bar',data});

// 回测结果曲线：dates为交易日标签，series为{name,data,bar,area}。
export function backtestChart(container,dates,series,title=''){
  chart(container,{xAxis:{type:'category',data:dates},yAxis:{type:'value',scale:true},dataZoom:[{type:'inside'},{type:'slider',height:16}],
    series:series.map(item=>item.bar?bar(item.name,item.data):{...line(item.name,item.data),areaStyle:item.area?{opacity:.15}:undefined})},title);
}
// 通用时序图，series包含已经计算好的值。
function timeseries(container,dates,series,title=''){chart(container,{xAxis:{type:'category',data:dates},yAxis:{type:'value',scale:true},dataZoom:[{type:'inside'},{type:'slider',height:16}],series},title);}

// 渲染注册指标的专用图，未识别key退回通用数据展示。
export function renderMetric(container,key,data){
  if(data.empty){container.textContent=data.hint||'没有可用样本';return;}
  const notes=[data.hint,data.basis,data.note,data.basis_note,data.stock_note,data.industry?.note].filter(Boolean);
  if(data.insufficient){const p=document.createElement('p');p.textContent='样本不足，空值不用于质量判断';container.append(p);}
  switch(key){
    case 'coverage':table(container,[{全部行数:data.rows,池内行数:data.pool_rows,股票数:data.stocks,交易日数:data.days,全样本覆盖率:data.coverage_all,池内覆盖率:data.coverage_universe,有效配对率:data.pair_coverage,缺失值:data.missing,无穷值:data.non_finite}]);{const detail=document.createElement('details');detail.innerHTML='<summary>池内取值分布与每日样本数</summary>';container.append(detail);generic(detail,{取值分布:data.distribution,每日有效配对数:data.daily_pairs});}break;
    case 'ic_stats':table(container,[{指标:'IC',...data.ic},{指标:'RankIC',...data.rank_ic}]);break;
    case 'ic_series':timeseries(container,data.dates,[line('累计 IC',data.cumulative.ic),line('累计 RankIC',data.cumulative.rank_ic)]);break;
    case 'ic_dist':{const grid=document.createElement('div');grid.className='charts';container.append(grid);for(const k of ['ic','rank_ic']){const d=data[k];const labels=d.edges.slice(0,-1).map((v,i)=>`${fmt(v)}～${fmt(d.edges[i+1])}`);const zeroBin=d.edges.findIndex((v,i)=>i<d.edges.length-1&&v<=0&&d.edges[i+1]>=0);const interval=Math.max(0,Math.ceil(labels.length/8)-1);const series={...bar(k,d.counts)};if(zeroBin>=0)series.markLine={silent:true,symbol:'none',lineStyle:{color:'#a34a42',type:'dashed'},label:{formatter:'0',color:'#a34a42'},data:[{xAxis:zeroBin}]};chart(grid,{xAxis:{type:'category',data:labels,axisLabel:{show:true,interval,rotate:45}},yAxis:{type:'value'},series:[series]},k+' 分布');}break;}
    case 'bin_returns':{const grid=document.createElement('div');grid.className='charts';container.append(grid);for(const w of data.weightings){const d=data[w];chart(grid,{xAxis:{type:'category',data:data.bins.map(String)},yAxis:{type:'value'},series:[bar('平均期收益',d.mean),{name:'标准误',type:'custom',data:d.mean.map((v,i)=>[i,v===null||d.stderr[i]===null?null:v-d.stderr[i],v===null||d.stderr[i]===null?null:v+d.stderr[i]]),renderItem:(params,api)=>{if(api.value(1)==null||api.value(2)==null)return;const a=api.coord([api.value(0),api.value(1)]),b=api.coord([api.value(0),api.value(2)]);return{type:'group',children:[{type:'line',shape:{x1:a[0],y1:a[1],x2:b[0],y2:b[1]},style:{stroke:'#0a2540'}},...[a,b].map(p=>({type:'line',shape:{x1:p[0]-5,y1:p[1],x2:p[0]+5,y2:p[1]},style:{stroke:'#0a2540'}}))]};}}]},data.weight_labels[w]);table(grid,[{口径:data.weight_labels[w],单调性:data.monotonicity[w].spearman}]);table(grid,data.bins.map((b,i)=>({分位:b,均值:d.mean[i],标准误:d.stderr[i],t:d.t[i],天数:d.days[i]})));}break;}
    case 'bin_curves':{const grid=document.createElement('div');grid.className='charts';container.append(grid);for(const w of data.weightings)timeseries(grid,data.dates,data[w].curves.map((v,i)=>line('分位'+data.bins[i],v)),data.weight_labels[w]);break;}
    case 'spread':{const grid=document.createElement('div');grid.className='charts';container.append(grid);for(const w of data.weightings){timeseries(grid,data.dates,[line('多空净值',data[w].cumulative),line('日度价差',data[w].daily)],data.weight_labels[w]);const {daily,cumulative,...stats}=data[w];table(grid,[stats]);}break;}
    case 'bin_dist':{const grid=document.createElement('div');grid.className='charts';container.append(grid);for(const w of data.weightings){chart(grid,{xAxis:{type:'category',data:data.bins.map(String)},yAxis:{type:'value',scale:true},series:[{name:'分位收益分布',type:'boxplot',data:data.daily[w].map(d=>d.count?[d.min,d.p25,d.p50,d.p75,d.max]:[null,null,null,null,null])}]},data.weight_labels[w]);}const detail=document.createElement('details');detail.innerHTML='<summary>分位内个股收益分布</summary>';container.append(detail);table(detail,data.stocks);break;}
    case 'turnover':timeseries(container,data.daily_turnover.dates,[line('全池分位换手',data.daily_turnover.mean)]);table(container,Object.entries(data.autocorrelation).map(([lag,v])=>({lag,...v})));table(container,data.bins.map((bin,i)=>({分位:bin,平均换手:data.bin_turnover.mean[i],天数:data.bin_turnover.days[i]})));break;
    case 'decay':chart(container,{xAxis:{type:'category',data:data.curve.map(v=>v.horizon+'日')},yAxis:[{type:'value'},{type:'value'}],series:[line('RankIC均值',data.curve.map(v=>v.rank_ic_mean)),{...line('RankICIR',data.curve.map(v=>v.rank_ic_ir)),yAxisIndex:1}]});table(container,data.curve);for(const w of data.weightings||[])table(container,(data.long_excess[w]||[]).map((v,i)=>({口径:data.weight_labels[w],周期:v.horizon,多头超额:v.mean,多空:data.long_short[w][i].mean,t:v.t})));break;
    case 'exposure':if(data.stats){chart(container,{xAxis:{type:'category',data:data.stats.map(v=>v.key)},yAxis:{type:'value'},series:[bar('平均暴露系数',data.stats.map(v=>v.mean))]});table(container,data.stats);timeseries(container,data.series.dates,Object.entries(data.series.coefficient).map(([k,v])=>line(k,v)));}else generic(container,data);break;
    case 'incremental_ic':table(container,[{指标:'原始 RankIC（同样本）',...data.raw},{指标:'残差 RankIC',...data.residual}]);if(data.series)timeseries(container,data.series.dates,[line('原始',data.series.raw),line('残差',data.series.residual)]);if(data.delta)table(container,[{差值:data.delta.mean,保留比例:data.survivor_share}]);break;
    case 'correlation':table(container,data.top);{const cells=[];data.matrix.forEach((row,y)=>row.forEach((v,x)=>{if(v!==null)cells.push([x,y,v,data.days[y][x]]);}));chart(container,{tooltip:{position:'top',formatter:p=>`${esc(data.names[p.data[1]])} × ${esc(data.names[p.data[0]])}<br>相关性 ${fmt(p.data[2])}<br>有效日 ${p.data[3]}`},grid:{left:110,right:40,top:20,bottom:135},xAxis:{type:'category',data:data.names,axisLabel:{rotate:60,fontSize:10}},yAxis:{type:'category',data:data.names,axisLabel:{fontSize:10}},visualMap:{min:-1,max:1,calculable:true,orient:'horizontal',left:'center',bottom:0,inRange:{color:['#0a2540','#f6f9fc','#635bff']}},series:[{type:'heatmap',data:cells}]},'逐日截面 Spearman 均值',true);}break;
    default:generic(container,data);
  }
  for(const text of notes){const p=document.createElement('p');p.className='note';p.textContent=text;container.append(p);}
}
