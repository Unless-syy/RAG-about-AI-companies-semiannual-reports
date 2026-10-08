/* visual.html 的前端逻辑自检（可选，需要 Node；不参与服务运行）。
 *
 *   node verify_viewer.js
 *
 * 它在一个最小 DOM 桩里真跑 web/visual.html 的内联脚本（页面代码一行不改），
 * 再拿 data/chunks.jsonl 当基准做交叉核对。查的是浏览器里最容易悄悄错、
 * 又没人盯着的两类东西：
 *   1. 逐块的元数据推断——compRanges 二分定位公司、引用串拼接、页码格式；
 *   2. 渲染路径是否真的执行到（画点、画召回环、出卡片），而不是抛在半路。
 *
 * 它查不到浏览器渲染的实际观感，也查不到真实 SSE 链路——那要起服务后在浏览器里看。
 */
const fs = require('fs'), path = require('path'), vm = require('vm');

const ROOT = __dirname;
const p = (...a) => path.join(ROOT, ...a);
const read = f => fs.readFileSync(p(f), 'utf8');

// ---------------------------------------------------------------- 最小 DOM 桩
function mkEl(tag){
  const el = {
    tagName: tag, _html: '', textContent: '', style: {}, dataset: {}, disabled: false,
    clientWidth: 900, clientHeight: 520, width: 0, height: 0,
    offsetWidth: 200, offsetHeight: 40,
    classList: {
      _s: new Set(),
      add(c){ this._s.add(c); }, remove(c){ this._s.delete(c); },
      toggle(c, f){ const on = (f === undefined) ? !this._s.has(c) : !!f;
                    on ? this._s.add(c) : this._s.delete(c); return on; },
      contains(c){ return this._s.has(c); },
    },
    get innerHTML(){ return this._html; },
    set innerHTML(v){ this._html = String(v); },
    addEventListener(){}, removeEventListener(){}, appendChild(){}, focus(){},
    getBoundingClientRect(){ return {left:0, top:0, width:this.clientWidth, height:this.clientHeight}; },
    querySelector(){ return mkEl('div'); }, querySelectorAll(){ return []; },
    scrollIntoView(){}, closest(){ return null; },
    getContext(){ return CTX2D; },
    parentElement: null,
  };
  return el;
}

// 绘图上下文：所有调用 no-op，但记调用次数，用来断言「真的画到了」
const CTX2D = new Proxy({ calls: {} }, {
  get(t, k){
    if (k === 'calls') return t.calls;
    if (k === 'canvas') return null;
    if (typeof k === 'symbol') return undefined;
    return () => { t.calls[k] = (t.calls[k] || 0) + 1; };
  },
  set(){ return true; },
});

// ---------------------------------------------------------------- 载入页面
const html = read('web/visual.html');
const inline = (html.match(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/) || [])[1];
if (!inline) { console.error('✗ web/visual.html 里找不到内联脚本'); process.exit(1); }

const els = {};
for (const id of [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]))
  els[id] = mkEl(id === 'map' ? 'canvas' : 'div');
els.map.parentElement = mkEl('div');

const document_ = {
  getElementById: id => els[id] || null,
  querySelector: () => mkEl('div'), querySelectorAll: () => [],
  createElement: t => mkEl(t), head: { appendChild(){} }, addEventListener(){},
};
const win = {
  document: document_, devicePixelRatio: 2, location: {protocol: 'http:'},
  addEventListener(){}, removeEventListener(){},
  requestAnimationFrame: () => 1,
  performance: {now: () => Date.now()},
  navigator: {clipboard: null}, CSS: {escape: s => s},
};
win.window = win; win.self = win;
const ctx = vm.createContext(win);
Object.assign(ctx, {
  document: document_, performance: win.performance, navigator: win.navigator,
  requestAnimationFrame: win.requestAnimationFrame, setTimeout, clearTimeout,
  console, location: win.location, CSS: win.CSS,
});

const A = [];
const ok = (name, cond, extra) =>
  A.push([name, !!cond, extra === undefined ? '' : String(extra)]);

let meta = null;
try {
  vm.runInContext(read('web/viewer_meta.js'), ctx, {filename: 'viewer_meta.js'});
  vm.runInContext(inline, ctx, {filename: 'visual.html(inline)'});
  ok('viewer_meta.js 与内联脚本都能跑完', true);
} catch (e){
  ok('viewer_meta.js 与内联脚本都能跑完', false, e.message);
}
meta = ctx.window && ctx.window.VIEWER_META;
if (!meta){ console.error('✗ VIEWER_META 未定义，先运行 python build_viewer.py'); process.exit(1); }

// ---------------------------------------------------------------- 产物自洽
const N = meta.corpus.n_chunks;
ok('xyRaw/xyRes 与块数一致', meta.xyRaw.length === 2 * N && meta.xyRes.length === 2 * N,
   `xyRaw=${meta.xyRaw.length / 2} xyRes=${meta.xyRes.length / 2} n=${N}`);
ok('坐标无 NaN/Inf', [...meta.xyRaw, ...meta.xyRes].every(Number.isFinite));
ok('逐块数组长度都是 ' + N, ['groupIdx','sectionIdx','startPage','endPage','hasTable','snippet']
   .every(k => meta[k].length === N));
ok('compRanges [起始,块数] 首尾相接铺满', (() => {
  if (!meta.compRanges) return false;
  let prev = 0;
  for (const [s, c] of meta.compRanges){ if (s !== prev) return false; prev = s + c; }
  return prev === N;
})(), meta.compRanges ? meta.compRanges.length + ' 段' : 'null');
ok('分组块数之和 = 块数', meta.groupCounts.reduce((a, b) => a + b, 0) === N,
   JSON.stringify(meta.groupCounts));
ok('apiBase 端口与 config.PORT 一致', meta.apiBase.endsWith(':' + meta.port), meta.apiBase);

// ---------------------------------------------------------------- 页面渲染
ok('标题栏已填', /家公司/.test(els.meta.textContent), els.meta.textContent.slice(0, 60));
ok('语料 KPI 已渲染', /个块/.test(els.kpis._html),
   (els.kpis._html.match(/<b>[^<]+<\/b>/g) || []).join(' '));
ok('公司表行数 = 公司数', (els.coBody._html.match(/co-row/g) || []).length === meta.companies.length,
   (els.coBody._html.match(/co-row/g) || []).length + ' 行');
ok('图例列出三组', meta.groups.every(g => els.legend._html.includes(g.name)));
ok('地图画了 ' + N + ' 个点', CTX2D.calls.arc >= N, 'arc=' + CTX2D.calls.arc);

// ---------------------------------------------------------------- 逐块元数据交叉核对
const chunks = read('data/chunks.jsonl').trim().split('\n').map(JSON.parse);
let badCo = 0, badCit = 0, badSec = 0, firstBad = '';
for (let i = 0; i < chunks.length; i++){
  const ci = ctx.companyOf(i);
  if (!meta.companies[ci] || meta.companies[ci].company !== chunks[i].company){
    badCo++; if (!firstBad) firstBad = `#${i} 得 ${meta.companies[ci] && meta.companies[ci].company}，应为 ${chunks[i].company}`;
  }
  if (ctx.citationOf(i) !== chunks[i].citation.replace(/^\[|\]$/g, '')){
    badCit++; if (!firstBad) firstBad = `#${i} 引用串不符`;
  }
  if (ctx.sectionOf(i) !== chunks[i].section) badSec++;
}
ok(`companyOf 二分定位 ${chunks.length} 个块全对`, badCo === 0, badCo ? badCo + ' 个错（' + firstBad + '）' : '');
ok(`citationOf 与 chunks.jsonl 的 citation 逐字一致`, badCit === 0, badCit ? badCit + ' 个不符' : '');
ok('sectionOf 与章节名一致', badSec === 0, badSec ? badSec + ' 个不符' : '');

const single = chunks.findIndex(c => c.start_page === c.end_page);
const multi = chunks.findIndex(c => c.end_page > c.start_page);
ok('单页块页码 pN', ctx.pagesOf(single) === 'p' + chunks[single].start_page, ctx.pagesOf(single));
ok('跨页块页码 pA-B', multi >= 0 && /^p\d+-\d+$/.test(ctx.pagesOf(multi)), ctx.pagesOf(multi));

// ---------------------------------------------------------------- 召回事件渲染
// 字段与 rag/server.py 的 _public_hit() 输出同构；这里自己造 10 条，
// 不依赖服务在跑（真实链路另由起服务后的浏览器验证）。
const idxs = Array.from({length: 10}, (_, k) => Math.floor(k * N / 10) + 3);
const hits = idxs.map((i, k) => ({
  chunk_id: chunks[i].chunk_id, company: chunks[i].company, code: chunks[i].code,
  period: chunks[i].period, section: chunks[i].section, pages: chunks[i].pages,
  citation: chunks[i].citation, has_table: chunks[i].has_table, rank: k + 1,
  rrf_score: (0.0328 - k * 0.001).toFixed(6), vec_rank: k + 1, vec_score: 0.7 - k * 0.01,
  bm25_rank: (k % 3) ? k + 2 : null, bm25_score: (k % 3) ? 20 - k : null,
  text: chunks[i].text, global_index: i,
}));
ctx.renderHits(hits);
ok('召回卡片 10 张', (els.hits._html.match(/class="hit"/g) || []).length === 10,
   (els.hits._html.match(/class="hit"/g) || []).length + ' 张');
ok('卡片带分数与定位按钮',
   els.hits._html.includes('RRF') && els.hits._html.includes('在地图上定位'));
ok('原文进了卡片', els.hits._html.includes(chunks[idxs[0]].text.slice(0, 20)));

const before = CTX2D.calls.arc;
ctx.draw();
ok('召回环已画', CTX2D.calls.arc - before >= 10, `本轮新增 arc ${CTX2D.calls.arc - before}`);

// 两个视图都要能画（xyRes 是另一套坐标）
let viewOk = true, viewMsg = '';
try {
  ctx.setView('res'); ctx.draw();
  ctx.setView('raw'); ctx.draw();
} catch (e){ viewOk = false; viewMsg = e.message; }
ok('两套坐标都能切换并绘制', viewOk, viewMsg);
ok('切换视图后底部说明已更新', /召回空间/.test(els.mapfoot._html), '');

// 「在地图上定位」：镜头应当把该点放到画布中心附近
let locOk = false, locMsg = '';
try {
  ctx.setView('res');                       // 先在语义视角，定位应自动切回召回空间
  ctx.locateHit(0);
  const a = ctx.view === 'raw' ? ctx.window.VIEWER_META.xyRaw : ctx.window.VIEWER_META.xyRes;
  const i = hits[0].global_index;
  const sx = a[2 * i] * ctx.cam.k + ctx.cam.tx, sy = a[2 * i + 1] * ctx.cam.k + ctx.cam.ty;
  locOk = ctx.view === 'raw' && Math.abs(sx - 900 / 2) < 1.5 && Math.abs(sy - 520 / 2) < 1.5;
  locMsg = `view=${ctx.view} 该点屏幕位置 (${sx.toFixed(1)}, ${sy.toFixed(1)})，画布中心 (450, 260)`;
} catch (e){ locMsg = e.message; }
ok('定位后该点落在画布中心，且自动切回召回空间', locOk, locMsg);

// 点公司名高亮：镜头应罩住该公司的点，且不能把 K0（点半径基准）改掉
let coOk = false, coMsg = '';
try {
  ctx.setView('raw');
  const k0 = ctx.K0;
  ctx.centerOnCompany(0);
  const seg = meta.compRanges[0];
  let inb = true;
  for (let i = seg[0]; i < seg[0] + seg[1]; i++){
    const sx = meta.xyRaw[2 * i] * ctx.cam.k + ctx.cam.tx;
    const sy = meta.xyRaw[2 * i + 1] * ctx.cam.k + ctx.cam.ty;
    if (sx < -1 || sx > 901 || sy < -1 || sy > 521){ inb = false; break; }
  }
  coOk = inb && ctx.K0 === k0;
  coMsg = `该公司的块是否全在画布内：${inb}；K0 未被改写：${ctx.K0 === k0}`;
} catch (e){ coMsg = e.message; }
ok('点公司高亮后该公司全部落进视野', coOk, coMsg);

// ---------------------------------------------------------------- 结果
console.log('\n=== visual.html 前端逻辑自检 ===');
let pass = 0;
for (const [n, c, e] of A){ console.log(`${c ? '✓' : '✗'} ${n}${e ? '  —— ' + e : ''}`); if (c) pass++; }
console.log(`\n${pass}/${A.length} 通过`);
process.exit(pass === A.length ? 0 : 1);
