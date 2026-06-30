/* ============================================================
   LERNREIHE — geteilte Engine
   ------------------------------------------------------------
   Zentralisiert: Scope-Plotting, die echten Zell-Integratoren,
   die 6 Benchmark-ODE-Systeme + euler_odeint (faithful zum Code),
   Chrome (Hero/TOC/Topbar/Prev-Next) und Slider-Helfer.

   Klassisches Script (kein Modul) -> alle top-level Bindings sind
   für die seiteneigenen Inline-Scripts sichtbar.
   ============================================================ */

/* ---------- Mikro-Helfer ---------- */
const $  = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const sig   = x => 1/(1+Math.exp(-x));
const tanh  = x => Math.tanh(x);
const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
const lerp  = (a,b,t) => a+(b-a)*t;
const fmt   = (v,n=2) => (v>=0?' ':'') + v.toFixed(n);
// deterministischer PRNG (mulberry32) — gleicher seed -> gleiche Folge
function rng(seed){ return function(){ seed|=0; seed=seed+0x6D2B79F5|0; let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

const CSS = getComputedStyle(document.documentElement);
const COL = {
  ct:  CSS.getPropertyValue('--ct').trim()  || '#6b7280',
  ltc: CSS.getPropertyValue('--ltc').trim() || '#c2710c',
  lrc: CSS.getPropertyValue('--lrc').trim() || '#2563eb',
  cfc: CSS.getPropertyValue('--cfc').trim() || '#0f9d6b',
  ncp: CSS.getPropertyValue('--ncp').trim() || '#7c3aed',
  lstm:CSS.getPropertyValue('--lstm').trim()|| '#be123c',
  data:CSS.getPropertyValue('--data').trim()|| '#0891b2',
  accent:CSS.getPropertyValue('--accent').trim()|| '#d64518',
  grid:'#1d2738', line:'#283449', faint:'#5d6b82', ink:'#cdd6e4', sub:'#7e8ca6'
};

/* ============================================================
   SERIES — eine einzige Quelle für Nav / Index / Prev-Next
   ============================================================ */
const SERIES = [
  {n:'00', file:'00-eec-family.html', ac:'--ltc',  track:'Grundlagen',
   title:'Die EEC-Familie', sub:'Vom Stromkreis zum liquiden Neuron — CT-RNN, LTC, LRC, CfC, steife ODE, Lipschitz.',
   tags:['EEC','steife ODE','Lipschitz']},
  {n:'01', file:'01-rnn-vanishing-gradient.html', ac:'--ct', track:'Grundlagen',
   title:'RNNs & das Vanishing-Gradient-Problem', sub:'Warum gewöhnliche RNNs vergessen: BPTT, das Jacobian-Produkt, exploding/vanishing — die Wurzel aller folgenden Tricks.',
   tags:['BPTT','Jacobian','RQ4']},
  {n:'02', file:'02-lstm-gru.html', ac:'--lstm', track:'Klassische Baselines',
   title:'LSTM & GRU', sub:'Die gegateten Klassiker. Constant Error Carousel, Input/Forget/Output-Gate, GRU als schlanke Variante.',
   tags:['Gates','CEC','Baseline']},
  {n:'03', file:'03-ltc-lrc.html', ac:'--ltc', track:'Liquide Zellen',
   title:'LTC & LRC', sub:'Die liquide Kern-Familie in voller Tiefe: Conductance-ODE, fused Euler mit 6 unfolds, liquid elastance ε.',
   tags:['ode_unfolds','elastance','Sättigung']},
  {n:'04', file:'04-cfc.html', ac:'--cfc', track:'Liquide Zellen',
   title:'CfC — Closed-form Continuous-time', sub:'Die ODE ohne Solver: zwei gelernte Regime, interpoliert von einem Sigmoid-Zeitgate. Mit Zahlenbeispielen.',
   tags:['ff1/ff2','time-gate','kein Solver']},
  {n:'05', file:'05-mixed-memory.html', ac:'--cfc', track:'Liquide Zellen',
   title:'Mixed Memory (ODE-LSTM)', sub:'LSTM-Speicherpfad c um eine ODE-Zelle gelegt — der Gradient umgeht die ODE-Jacobi. Konkrete Architektur.',
   tags:['ODE-LSTM','c-Pfad','mm_ltc/mm_lrc']},
  {n:'06', file:'06-ncp.html', ac:'--ncp', track:'Verschaltung',
   title:'NCP — Neural Circuit Policies', sub:'Spärliche, biologisch inspirierte Verdrahtung: sensory → inter → command → motor, mit fixer Maske.',
   tags:['Sparsity','C. elegans','SparseLinear']},
  {n:'07', file:'07-benchmarks-datasets.html', ac:'--data', track:'Das Experiment',
   title:'Benchmarks: Datensätze & dynamische Systeme', sub:'Die 6 ODE-Systeme (Lotka-Volterra & Co.), wie eine Trajektorie entsteht, Formel neben Code.',
   tags:['Lotka-Volterra','DOP853','NRMSE']},
  {n:'08', file:'08-training-code.html', ac:'--data', track:'Das Experiment',
   title:'Der Trainings-Code', sub:'Zeile für Zeile: euler_odeint, ODEFuncModel, Mini-Batch, Adam, Gradient-Clipping, der Gradient-Tracker.',
   tags:['euler_odeint','Adam','clip_norm']},
  {n:'09', file:'09-results-animation.html', ac:'--lrc', track:'Das Experiment',
   title:'Ergebnisse lesen & animieren', sub:'Wie der Graph entsteht: Phasenporträt, Lernkurve, Gradient-Flow — Schritt für Schritt aufgebaut.',
   tags:['Phasenporträt','Rollout','Lernkurve']},
  {n:'10', file:'10-statistics.html', ac:'--accent', track:'Das Experiment',
   title:'Von Zahlen zu Aussagen: Statistik', sub:'Wie aus 240 Läufen eine belastbare Behauptung wird: Wilcoxon, Bonferroni, Cohen’s d, gepaarte Vergleiche.',
   tags:['Wilcoxon','Bonferroni','Cohen’s d']},
];
const seriesIndex = id => SERIES.findIndex(s=>s.n===id);

/* ============================================================
   SCOPE — schlankes Canvas-Plot-Framework (DPR-scharf)
   ============================================================ */
class Scope{
  constructor(id, pad){
    this.cv = (typeof id==='string') ? document.getElementById(id) : id;
    this.ctx = this.cv.getContext('2d');
    this.dpr = Math.min(window.devicePixelRatio||1, 2);
    this.pad = pad || {l:44,r:16,t:18,b:26};
    this.resize();
    if(window.ResizeObserver){
      new ResizeObserver(()=>{ this.resize(); this.onresize && this.onresize(); })
        .observe(this.cv.parentElement);
    }
  }
  resize(){
    const r = this.cv.parentElement.getBoundingClientRect();
    this.w = Math.max(r.width, 240);
    this.h = Math.max(r.height, 280);
    this.cv.width  = this.w*this.dpr;
    this.cv.height = this.h*this.dpr;
    this.cv.style.height = this.h+'px';
    this.ctx.setTransform(this.dpr,0,0,this.dpr,0,0);
  }
  range(x0,x1,y0,y1){ this.x0=x0; this.x1=x1; this.y0=y0; this.y1=y1; }
  X(x){ return this.pad.l + (x-this.x0)/(this.x1-this.x0)*(this.w-this.pad.l-this.pad.r); }
  Y(y){ return this.pad.t + (1-(y-this.y0)/(this.y1-this.y0))*(this.h-this.pad.t-this.pad.b); }
  clear(){ this.ctx.clearRect(0,0,this.w,this.h); }
  grid(nx,ny,opt){
    opt=opt||{}; const c=this.ctx;
    c.strokeStyle=COL.grid; c.lineWidth=1; c.font='10px JetBrains Mono';
    c.fillStyle=COL.faint; c.textAlign='right'; c.textBaseline='middle';
    for(let i=0;i<=ny;i++){
      const y=this.y0+(this.y1-this.y0)*i/ny, py=this.Y(y);
      c.globalAlpha = (Math.abs(y)<1e-9)?0.9:0.4;
      c.beginPath(); c.moveTo(this.pad.l,py); c.lineTo(this.w-this.pad.r,py); c.stroke();
      c.globalAlpha=1;
      if(opt.yl!==false) c.fillText(y.toFixed(opt.yd!=null?opt.yd:1), this.pad.l-6, py);
    }
    c.textAlign='center'; c.textBaseline='top';
    for(let i=0;i<=nx;i++){
      const x=this.x0+(this.x1-this.x0)*i/nx, px=this.X(x);
      c.globalAlpha=0.28;
      c.beginPath(); c.moveTo(px,this.pad.t); c.lineTo(px,this.h-this.pad.b); c.stroke();
      c.globalAlpha=1;
      if(opt.xl) c.fillText(x.toFixed(opt.xd!=null?opt.xd:0), px, this.h-this.pad.b+5);
    }
  }
  path(pts, color, w, opt){
    opt=opt||{}; const c=this.ctx; if(pts.length<1) return;
    c.lineJoin='round'; c.lineCap='round'; c.strokeStyle=color; c.lineWidth=w||2;
    if(opt.dash) c.setLineDash(opt.dash); else c.setLineDash([]);
    c.globalAlpha = opt.alpha!=null?opt.alpha:1;
    c.beginPath();
    let started=false;
    for(const p of pts){
      if(p==null){ started=false; continue; }
      const px=this.X(p[0]), py=this.Y(clamp(p[1],this.y0-99,this.y1+99));
      if(!started){ c.moveTo(px,py); started=true; } else c.lineTo(px,py);
    }
    c.stroke(); c.setLineDash([]); c.globalAlpha=1;
  }
  /* gefüllte Fläche unter einer Kurve (bis y=baseline) */
  area(pts, color, baseline, alpha){
    const c=this.ctx; if(pts.length<2) return;
    const b = baseline!=null?baseline:this.y0;
    c.globalAlpha = alpha!=null?alpha:0.14; c.fillStyle=color;
    c.beginPath(); c.moveTo(this.X(pts[0][0]), this.Y(b));
    for(const p of pts){ c.lineTo(this.X(p[0]), this.Y(clamp(p[1],this.y0-99,this.y1+99))); }
    c.lineTo(this.X(pts[pts.length-1][0]), this.Y(b)); c.closePath(); c.fill(); c.globalAlpha=1;
  }
  dot(x,y,color,r){ const c=this.ctx; c.fillStyle=color; c.beginPath(); c.arc(this.X(x),this.Y(y),r||3.5,0,7); c.fill(); }
  ring(x,y,color,r,w){ const c=this.ctx; c.strokeStyle=color; c.lineWidth=w||2; c.beginPath(); c.arc(this.X(x),this.Y(y),r||5,0,7); c.stroke(); }
  vline(x,color,w,dash){ const c=this.ctx; c.strokeStyle=color; c.lineWidth=w||1; if(dash)c.setLineDash(dash); c.beginPath(); c.moveTo(this.X(x),this.pad.t); c.lineTo(this.X(x),this.h-this.pad.b); c.stroke(); c.setLineDash([]); }
  hline(y,color,w,dash){ const c=this.ctx; c.strokeStyle=color; c.lineWidth=w||1; if(dash)c.setLineDash(dash); c.beginPath(); c.moveTo(this.pad.l,this.Y(y)); c.lineTo(this.w-this.pad.r,this.Y(y)); c.stroke(); c.setLineDash([]); }
  seg(x0,y0,x1,y1,color,w,dash){ const c=this.ctx; c.strokeStyle=color; c.lineWidth=w||2; if(dash)c.setLineDash(dash); c.beginPath(); c.moveTo(this.X(x0),this.Y(y0)); c.lineTo(this.X(x1),this.Y(y1)); c.stroke(); c.setLineDash([]); }
  arrow(x0,y0,x1,y1,color,w){
    const c=this.ctx, A=[this.X(x0),this.Y(y0)], B=[this.X(x1),this.Y(y1)];
    c.strokeStyle=color; c.fillStyle=color; c.lineWidth=w||2;
    c.beginPath(); c.moveTo(A[0],A[1]); c.lineTo(B[0],B[1]); c.stroke();
    const ang=Math.atan2(B[1]-A[1],B[0]-A[0]), s=7;
    c.beginPath(); c.moveTo(B[0],B[1]);
    c.lineTo(B[0]-s*Math.cos(ang-0.4),B[1]-s*Math.sin(ang-0.4));
    c.lineTo(B[0]-s*Math.cos(ang+0.4),B[1]-s*Math.sin(ang+0.4));
    c.closePath(); c.fill();
  }
  /* Balken: items = [{x, h, color, w?}] in Datenkoordinaten, Basis y=0 */
  bars(items, baseline){
    const c=this.ctx, b=baseline!=null?baseline:0;
    items.forEach(it=>{
      const x0=this.X(it.x-(it.w||0.3)), x1=this.X(it.x+(it.w||0.3));
      const yT=this.Y(it.h), yB=this.Y(b);
      c.fillStyle=it.color; c.globalAlpha=it.alpha!=null?it.alpha:1;
      c.fillRect(Math.min(x0,x1), Math.min(yT,yB), Math.abs(x1-x0), Math.abs(yB-yT));
      c.globalAlpha=1;
    });
  }
  label(txt,x,y,color,align,font){
    const c=this.ctx; c.font=font||'600 11px JetBrains Mono'; c.fillStyle=color;
    c.textAlign=align||'left'; c.textBaseline='middle'; c.fillText(txt,this.X(x),this.Y(y));
  }
  /* Text in Pixelkoordinaten (für Achsentitel etc.) */
  text(txt,px,py,color,align,font,baseline){
    const c=this.ctx; c.font=font||'600 11px JetBrains Mono'; c.fillStyle=color;
    c.textAlign=align||'left'; c.textBaseline=baseline||'alphabetic'; c.fillText(txt,px,py);
  }
}

/* ============================================================
   ZELL-INTEGRATOREN — skalare Reduktion der echten Updates.
   Jeweils 1 Neuron; Vektorterme W·x werden zu Skalaren win·x.
   Bezug zur Quelle steht im Kommentar.
   ============================================================ */
// CT-RNN: forward Euler, festes τ.  (ctrnn_cell.py:50-58)
function stepCTRNN(h,x,dt,p){
  const drive = Math.tanh(p.win*x + p.wrec*h + p.b);
  return h + (dt/p.tau)*(-h + drive);
}
// LTC: kompakte ODE, wählbarer Solver, ode_unfolds.  (ltc_cell.py:213-246)
function stepLTC(h,x,dt,p){
  let v=h; const U=p.unfolds||6; const sub=dt/U;
  for(let i=0;i<U;i++){
    const f = (p.gain!=null?p.gain:1) * sig(p.win*x + p.wrec*v + p.b); // liquide Synapse, f>0
    if(p.solver==='explicit') v = v + sub*(-(1/p.tau + f)*v + f*p.A);
    else                      v = (v + sub*f*p.A)/(1 + sub*(1/p.tau + f)); // fused semi-implicit
  }
  return v;
}
// LRC: forget-gate-Form, Sättigung + Elastance-Gate, explicit Euler.  (lrc_cell.py:229-280)
function stepLRC(h,x,dt,p){
  const v=h;
  const fpre = p.wfIn*x + p.wfRec*v + p.bf;
  const gpre = p.wgIn*x + p.wgRec*v + p.bg;
  const vprime = -v*sig(fpre) + p.vleak*Math.tanh(gpre);            // beide Terme gesättigt
  const eps = p.elast ? sig(p.weIn*x + p.weRec*v + p.be) : 1.0;     // ε∈(0,1) bzw. =1 → saturierte LTC
  return v + eps*dt*vprime;
}
// CfC: closed-form, Sigmoid-Zeitgate interpoliert zwei Heads.  (cfc_cell.py:74-92)
function stepCFC(h,x,dt,p){
  const z   = 1.7159*Math.tanh(0.666*(p.win*x + p.wrec*h + p.b));   // backbone (lecun-tanh)
  const ff1 = Math.tanh(p.a1*z + p.wrec*h + p.c1);                  // Regime 1 (halten)
  const ff2 = Math.tanh(p.a2*z + p.win*x + p.c2);                   // Regime 2 (Ziel)
  const ta  = p.ta*(1 + 0.4*Math.abs(x));                           // leicht input-abhängiges Zeitgate
  const gate= sig(ta*dt + p.tb);
  return (1-gate)*ff1 + gate*ff2;
}
// LSTM (skalar): vier Gates, additiver c-Pfad (CEC).  (lstm_cell.py / Keras)
function stepLSTM(state,x,p){
  const h=state.h||0, c=state.c||0;
  const i=sig(p.wix*x + p.wih*h + p.bi);
  const f=sig(p.wfx*x + p.wfh*h + p.bf);
  const o=sig(p.wox*x + p.woh*h + p.bo);
  const g=Math.tanh(p.wgx*x + p.wgh*h + p.bg);
  const nc=f*c + i*g;
  const nh=o*Math.tanh(nc);
  return {h:nh, c:nc, i, f, o, g};
}
// GRU (skalar): reset + update gate, konvexe Mischung.  (gru_cell.py / Keras)
function stepGRU(state,x,p){
  const h=state.h||0;
  const r=sig(p.wrx*x + p.wrh*h + p.br);
  const u=sig(p.wux*x + p.wuh*h + p.bu);
  const hc=Math.tanh(p.whx*x + p.whh*(r*h) + p.bh);
  const nh=(1-u)*hc + u*h;
  return {h:nh, r, u, hc};
}

/* Trajektorie einer skalaren Zelle über ein Zeitfenster integrieren */
function integrate(stepFn, p, signal, dt, T, h0){
  let h = h0||0; const pts=[[0,h]]; const n=Math.max(2,Math.round(T/dt));
  for(let i=1;i<=n;i++){ const t=i*dt; h=stepFn(h, signal(t-dt), dt, p); pts.push([t, h]); }
  return pts;
}

/* ============================================================
   BENCHMARK-DYNAMIK — die 6 Systeme exakt aus datasets.py.
   RHS gibt dy = f(y) zurück (y = [y0,y1]).
   ============================================================ */
const SYSTEMS = {
  spiral:{ // dy = y @ [[-0.1,3],[-3,-0.1]]
    y0:[0.5,0.01], tspan:[0,25], label:'Spiral',
    f:(y)=>[ -0.1*y[0] - 3.0*y[1],  3.0*y[0] - 0.1*y[1] ]},
  duffing:{
    y0:[-1,1], tspan:[0,25], label:'Duffing-Oszillator',
    f:(y)=>[ y[1], y[0] - y[0]**3 ]},
  periodic_sinusoidal:{
    y0:[1,1], tspan:[0,10], label:'Periodic Sinusoidal',
    f:(y)=>{ const r=Math.hypot(y[0],y[1]); return [ y[0]*(1-r)-y[1], y[0]+y[1]*(1-r) ]; }},
  periodic_predator_prey:{ // Lotka-Volterra
    y0:[1,1], tspan:[0,10], label:'Periodic Predator-Prey (Lotka-Volterra)',
    f:(y)=>[ 1.5*y[0] - y[0]*y[1], -3.0*y[1] + y[0]*y[1] ]},
  limited_predator_prey:{
    y0:[1,1], tspan:[0,20], label:'Limited Predator-Prey',
    f:(y)=>[ y[0]*(1-y[0]) - y[0]*y[1], -y[1] + 2.0*y[0]*y[1] ]},
  nonlinear_predator_prey:{
    y0:[2,1], tspan:[0,20], label:'Nonlinear Predator-Prey',
    f:(y)=>[ y[0]*(1-y[0]) + 0.33*y[0]*y[1], y[1]*(1-y[1]) + y[0]*y[1] ]},
};
/* Referenz-Trajektorie per RK4-Feinintegration (im Code: scipy DOP853).
   Liefert {t:[...], y:[[y0,y1],...]} mit n Punkten über tspan. */
function trueTrajectory(name, n){
  const S=SYSTEMS[name]; n=n||600;
  const [t0,t1]=S.tspan; const dt=(t1-t0)/(n-1);
  let y=S.y0.slice(); const T=[t0], Y=[y.slice()];
  const add=(a,b,s)=>[a[0]+s*b[0], a[1]+s*b[1]];
  for(let i=1;i<n;i++){
    const k1=S.f(y);
    const k2=S.f(add(y,k1,dt/2));
    const k3=S.f(add(y,k2,dt/2));
    const k4=S.f(add(y,k3,dt));
    y=[ y[0]+dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
        y[1]+dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]) ];
    T.push(t0+i*dt); Y.push(y.slice());
  }
  return {t:T, y:Y};
}
/* euler_odeint — faithful zu solver.py: y_{i+1} = y_i + dt*f(t_i,y_i) */
function eulerOdeint(f, y0, t){
  const ys=[y0.slice()];
  for(let i=0;i<t.length-1;i++){
    const dt=t[i+1]-t[i]; const dy=f(ys[ys.length-1]);
    ys.push([ ys[ys.length-1][0]+dt*dy[0], ys[ys.length-1][1]+dt*dy[1] ]);
  }
  return ys;
}
/* NRMSE — faithful zu metrics.py: RMSE / (max-min) über alle Komponenten */
function nrmse(yTrue, yPred){
  let se=0, n=0, mx=-1e9, mn=1e9;
  for(let i=0;i<yTrue.length;i++) for(let j=0;j<yTrue[i].length;j++){
    const a=yTrue[i][j], b=yPred[i][j]; se+=(a-b)*(a-b); n++;
    if(a>mx)mx=a; if(a<mn)mn=a;
  }
  return Math.sqrt(se/n)/((mx-mn)+1e-12);
}

/* ============================================================
   CHROME — auf jeder Seite gleich
   ============================================================ */
function renderMath(){
  if(typeof katex==='undefined'){ return setTimeout(renderMath, 80); }
  $$('[data-eq]').forEach(el=>{
    try{ katex.render(el.dataset.eq, el, {displayMode: el.dataset.inline?false:true, throwOnError:false}); }
    catch(e){ el.textContent=el.dataset.eq; }
  });
}
function initReveal(){
  const heroR = $$('.hero .reveal');
  if(typeof gsap==='undefined'){ $$('.reveal').forEach(e=>{e.style.opacity=1; e.style.transform='none';}); return; }
  gsap.to(heroR, {opacity:1, y:0, duration:1, ease:'power3.out', stagger:0.1, delay:0.15});
  if(typeof ScrollTrigger==='undefined'){ $$('.reveal').forEach(e=>{ if(!e.closest('.hero')){e.style.opacity=1;e.style.transform='none';} }); return; }
  gsap.registerPlugin(ScrollTrigger);
  $$('.reveal').forEach(el=>{
    if(el.closest('.hero')) return;
    gsap.to(el, {opacity:1, y:0, duration:0.7, ease:'power2.out', scrollTrigger:{trigger:el, start:'top 92%'}});
  });
}
function initProgress(){
  let bar=$('#progress'); if(!bar){ bar=document.createElement('div'); bar.id='progress'; document.body.appendChild(bar); }
  const upd=()=>{ const h=document.documentElement.scrollHeight-innerHeight; bar.style.width=(h>0?clamp(scrollY/h,0,1)*100:0)+'%'; };
  addEventListener('scroll', upd, {passive:true}); upd();
}
/* TOC automatisch aus <section id data-toc="Label"> */
function initTOC(){
  const secs=$$('section[data-toc]'); if(!secs.length) return;
  let nav=$('#toc'); if(!nav){ nav=document.createElement('nav'); nav.id='toc'; document.body.appendChild(nav); }
  secs.forEach(s=>{ const a=document.createElement('a'); a.href='#'+s.id; a.innerHTML=`<span>${s.dataset.toc}</span>`; a.dataset.id=s.id; nav.appendChild(a); });
  const links=$$('#toc a');
  const obs=new IntersectionObserver(es=>{ es.forEach(e=>{ if(e.isIntersecting){ links.forEach(l=>l.classList.toggle('active', l.dataset.id===e.target.id)); } }); }, {rootMargin:'-45% 0px -45% 0px'});
  secs.forEach(s=>obs.observe(s));
  addEventListener('scroll', ()=>{ nav.classList.toggle('show', scrollY>innerHeight*0.6); }, {passive:true});
}
/* Topbar + Prev/Next aus SERIES und <body data-lecture="03"> */
function buildChrome(){
  const id = document.body.dataset.lecture; if(!id) return;
  const idx = seriesIndex(id); const cur = SERIES[idx];
  document.body.classList.add('has-topbar');
  const bar=document.createElement('div'); bar.id='topbar';
  bar.innerHTML =
    `<a class="home" href="index.html"><span class="sq"></span>Lernreihe</a>`+
    `<span class="crumb">Vorlesung <b>${cur.n}</b> · ${cur.title}</span>`+
    `<span class="right">`+
      (idx>0?`<a href="${SERIES[idx-1].file}" title="${SERIES[idx-1].title}">&larr; ${SERIES[idx-1].n}</a>`:``)+
      `<a href="index.html">Übersicht</a>`+
      (idx<SERIES.length-1?`<a href="${SERIES[idx+1].file}" title="${SERIES[idx+1].title}">${SERIES[idx+1].n} &rarr;</a>`:``)+
    `</span>`;
  document.body.prepend(bar);
  const pn=document.createElement('nav'); pn.className='prevnext';
  const prev=idx>0?SERIES[idx-1]:null, next=idx<SERIES.length-1?SERIES[idx+1]:null;
  pn.innerHTML =
    (prev?`<a class="pn prev" href="${prev.file}"><div class="dir">&larr; Vorige Vorlesung</div><div class="t">${prev.n} · ${prev.title}</div></a>`
         :`<span class="pn prev disabled"><div class="dir">Start der Reihe</div><div class="t">—</div></span>`)+
    (next?`<a class="pn next" href="${next.file}"><div class="dir">Nächste Vorlesung &rarr;</div><div class="t">${next.n} · ${next.title}</div></a>`
         :`<a class="pn next" href="index.html"><div class="dir">Reihe abgeschlossen &rarr;</div><div class="t">Zurück zur Übersicht</div></a>`);
  const footer=$('footer');
  if(footer) footer.parentNode.insertBefore(pn, footer); else document.body.appendChild(pn);
}

/* ---------- HERO — Three.js Membran-Wellen-Gitter ---------- */
function initHero(){
  const canvas=$('.hero-canvas'); if(!canvas) return;
  if(typeof THREE==='undefined') return;
  const renderer=new THREE.WebGLRenderer({canvas, alpha:true, antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  const scene=new THREE.Scene();
  const cam=new THREE.PerspectiveCamera(52,1,0.1,200);
  cam.position.set(0,7.5,15); cam.lookAt(0,-0.5,0);
  const a1=canvas.dataset.c1||COL.ltc, a2=canvas.dataset.c2||COL.lrc, a3=canvas.dataset.c3||COL.cfc;
  const NX=46, NY=30, gap=0.92, N=NX*NY;
  const pos=new Float32Array(N*3), col=new Float32Array(N*3), base=[];
  const c1=new THREE.Color(a1), c2=new THREE.Color(a2), c3=new THREE.Color(a3), cink=new THREE.Color('#243043');
  let i=0;
  for(let y=0;y<NY;y++)for(let x=0;x<NX;x++){
    const px=(x-NX/2)*gap, pz=(y-NY/2)*gap; base.push([px,pz]);
    pos[i*3]=px; pos[i*3+1]=0; pos[i*3+2]=pz; i++;
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color', new THREE.BufferAttribute(col,3));
  const points=new THREE.Points(geo, new THREE.PointsMaterial({size:0.085, vertexColors:true, transparent:true, opacity:0.85, sizeAttenuation:true}));
  scene.add(points);
  const lpos=[]; const STRIDE=2;
  for(let y=0;y<NY;y+=STRIDE)for(let x=0;x<NX;x+=STRIDE){
    const a=base[y*NX+x];
    if(x+STRIDE<NX){ const b=base[y*NX+x+STRIDE]; lpos.push(a[0],0,a[1], b[0],0,b[1]); }
    if(y+STRIDE<NY){ const b=base[(y+STRIDE)*NX+x]; lpos.push(a[0],0,a[1], b[0],0,b[1]); }
  }
  const lgeo=new THREE.BufferGeometry(); lgeo.setAttribute('position', new THREE.Float32BufferAttribute(lpos,3));
  const lines=new THREE.LineSegments(lgeo, new THREE.LineBasicMaterial({color:0x9fb0c8, transparent:true, opacity:0.14}));
  scene.add(lines);
  function resize(){ const r=canvas.getBoundingClientRect(); renderer.setSize(r.width,r.height,false); cam.aspect=r.width/Math.max(r.height,1); cam.updateProjectionMatrix(); }
  resize(); addEventListener('resize', resize);
  let mx=0,my=0;
  addEventListener('pointermove', e=>{ mx=(e.clientX/innerWidth-0.5); my=(e.clientY/innerHeight-0.5); });
  let t=0, raf;
  const tmp=new THREE.Color();
  function wave(px,pz,t){
    const d=Math.sqrt(px*px+pz*pz);
    return Math.sin(d*0.52 - t*1.5)*0.95*Math.exp(-d*0.05)
         + Math.sin(px*0.32 + t*0.9)*0.28
         + Math.cos(pz*0.4 - t*0.7)*0.2;
  }
  function loop(){
    t+=0.016;
    const P=geo.attributes.position.array, C=geo.attributes.color.array;
    for(let k=0;k<N;k++){
      const px=base[k][0], pz=base[k][1], h=wave(px,pz,t);
      P[k*3+1]=h;
      const u=clamp((h+1.1)/2.2,0,1);
      if(u<0.5) tmp.copy(cink).lerp(c1, u*2); else { const v=(u-0.5)*2; tmp.copy(c1).lerp(c2,v).lerp(c3,v*0.5); }
      const lum=0.45+0.55*u;
      C[k*3]=tmp.r*lum; C[k*3+1]=tmp.g*lum; C[k*3+2]=tmp.b*lum;
    }
    geo.attributes.position.needsUpdate=true; geo.attributes.color.needsUpdate=true;
    const lp=lgeo.attributes.position.array;
    for(let k=0;k<lp.length;k+=3){ lp[k+1]=wave(lp[k],lp[k+2],t); }
    lgeo.attributes.position.needsUpdate=true;
    points.rotation.y = lines.rotation.y = Math.sin(t*0.08)*0.12 + mx*0.25;
    cam.position.x += ((mx*3) - cam.position.x)*0.03;
    cam.position.y += ((7.5 - my*2) - cam.position.y)*0.03;
    cam.lookAt(0,-0.5,0);
    renderer.render(scene,cam);
    raf=requestAnimationFrame(loop);
  }
  loop();
  new IntersectionObserver(es=>{ es.forEach(e=>{ if(e.isIntersecting && !raf) loop(); else if(!e.isIntersecting && raf){ cancelAnimationFrame(raf); raf=null; } }); }).observe(canvas);
}

/* ---------- Slider/Segment/Chip-Helfer ---------- */
function bind(id, valId, fn, dec){
  const el=$('#'+id); if(!el) return {el:null, upd:()=>{}};
  const v=valId?$('#'+valId):null;
  const upd=()=>{ if(v) v.textContent=parseFloat(el.value).toFixed(dec!=null?dec:2); fn(parseFloat(el.value)); };
  el.addEventListener('input', upd); return {el, upd};
}
function seg(containerId, fn){
  const cont=$('#'+containerId); if(!cont) return;
  cont.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
    cont.querySelectorAll('button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); fn(b.dataset.v||b.dataset.s);
  }));
}
function chips(containerId, fn){
  const cont=$('#'+containerId); if(!cont) return;
  cont.querySelectorAll('.chip').forEach(b=>b.addEventListener('click',()=>{
    b.classList.toggle('on'); fn(b.dataset.c, b.classList.contains('on'));
  }));
}
function whenVisible(el, cb){
  if(typeof el==='string') el=$('#'+el);
  if(!el || !window.IntersectionObserver){ cb&&cb(); return; }
  let seen=false;
  new IntersectionObserver((es,o)=>{ es.forEach(e=>{ if(e.isIntersecting){ seen=true; cb(); } }); }, {threshold:0.04}).observe(el);
  // Layout-Reflows (Fonts/KaTeX) lassen Scope den Canvas neu dimensionieren -> Canvas wird geleert.
  // Statische Labs daher nach jedem Resize neu auslösen (rAF-debounced, erst ab erstem Sichtbarwerden).
  if(window.ResizeObserver){
    let raf=0;
    new ResizeObserver(()=>{ if(!seen) return; cancelAnimationFrame(raf); raf=requestAnimationFrame(()=>{ try{ cb(); }catch(e){} }); }).observe(el);
  }
}

/* ---------- BOOT ---------- */
function boot(labs){
  const run=()=>{
    buildChrome();
    renderMath();
    initReveal();
    initProgress();
    initTOC();
    try{ initHero(); }catch(e){ console.warn('hero failed', e); }
    (labs||[]).forEach(fn=>{ try{ fn(); }catch(e){ console.warn((fn&&fn.name)||'lab', e); } });
  };
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', run);
  else run();
}

/* Namespace für Index/Sonderfälle */
window.LR = { SERIES, seriesIndex, Scope, SYSTEMS, trueTrajectory, eulerOdeint, nrmse, COL, boot };
