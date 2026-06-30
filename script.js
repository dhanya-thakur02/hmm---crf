// ════════════════════════════════════════════════════════════════
// HMM & CRF — Sequence Labeling From Scratch
// Vanilla JS port of the provided React component, restyled to
// match the Sentiment Analysis Lab design system (sidebar nav, dark
// hero per section, eyebrow labels, card system).
// ════════════════════════════════════════════════════════════════

// ─── small utils ───────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ─── DATA (extracted from the provided component) ─────────────────────────
const TAGS = ['DT','NN','VB','JJ','RB','IN','PRP','CD'];

const TAG_NAMES = {DT:'Determiner',NN:'Noun',VB:'Verb',JJ:'Adjective',RB:'Adverb',IN:'Preposition',PRP:'Pronoun',CD:'Cardinal'};

const HMM_PRIORS = {DT:0.25,NN:0.12,VB:0.05,JJ:0.06,RB:0.04,IN:0.08,PRP:0.18,CD:0.03};

const HMM_TRANS = {
  DT: {DT:0.02,NN:0.55,VB:0.02,JJ:0.28,RB:0.03,IN:0.04,PRP:0.04,CD:0.02},
  NN: {DT:0.04,NN:0.14,VB:0.30,JJ:0.06,RB:0.05,IN:0.22,PRP:0.06,CD:0.13},
  VB: {DT:0.18,NN:0.08,VB:0.05,JJ:0.08,RB:0.28,IN:0.15,PRP:0.12,CD:0.06},
  JJ: {DT:0.03,NN:0.62,VB:0.03,JJ:0.14,RB:0.04,IN:0.07,PRP:0.04,CD:0.03},
  RB: {DT:0.05,NN:0.06,VB:0.32,JJ:0.25,RB:0.12,IN:0.08,PRP:0.08,CD:0.04},
  IN: {DT:0.35,NN:0.10,VB:0.04,JJ:0.08,RB:0.05,IN:0.03,PRP:0.28,CD:0.07},
  PRP:{DT:0.04,NN:0.05,VB:0.52,JJ:0.04,RB:0.16,IN:0.06,PRP:0.06,CD:0.07},
  CD: {DT:0.04,NN:0.42,VB:0.08,JJ:0.06,RB:0.04,IN:0.18,PRP:0.04,CD:0.14},
};

const HMM_EMIT = {
  DT: {the:0.35,a:0.22,an:0.11,this:0.09,that:0.08,these:0.05,those:0.04,every:0.02,'some':0.03,'no':0.01},
  NN: {dog:0.04,cat:0.04,fox:0.03,time:0.04,man:0.03,city:0.03,book:0.03,car:0.03,house:0.03,run:0.01,'runs':0.01,fish:0.03,bird:0.03,idea:0.02,ball:0.02,tree:0.02,rain:0.02,sun:0.02,food:0.02,love:0.01},
  VB: {runs:0.05,run:0.04,is:0.06,are:0.05,was:0.04,eat:0.03,go:0.03,see:0.03,plays:0.03,'does':0.04,jump:0.03,walk:0.03,write:0.02,think:0.03,know:0.03,make:0.03,move:0.02,likes:0.03,lived:0.02},
  JJ: {quick:0.07,fast:0.06,slow:0.05,big:0.06,small:0.05,good:0.06,bad:0.05,old:0.05,new:0.06,red:0.04,blue:0.04,tall:0.04,short:0.04,hot:0.04,cold:0.04,happy:0.03,dark:0.04,bright:0.04,long:0.04},
  RB: {fast:0.06,quickly:0.07,slowly:0.06,very:0.08,quite:0.07,never:0.06,always:0.06,often:0.05,soon:0.04,now:0.05,here:0.05,just:0.05,then:0.04,really:0.04,well:0.04},
  IN: {in:0.14,on:0.10,at:0.10,of:0.13,by:0.07,for:0.09,with:0.08,from:0.07,about:0.05,into:0.05,over:0.04,through:0.05,under:0.03},
  PRP:{he:0.18,she:0.18,they:0.14,it:0.15,we:0.12,i:0.12,you:0.08,them:0.02,him:0.01},
  CD: {'1':0.07,'2':0.07,'3':0.07,one:0.09,two:0.09,three:0.08,four:0.07,five:0.07,ten:0.06,hundred:0.05,first:0.04,second:0.04,third:0.04,zero:0.04},
};

const CRF_FEATURES = [
  {name:'word_is(DT,the)',desc:'Word is "the" and tag is DT',type:'emission'},
  {name:'word_is(NN,*)',desc:'Word is a known noun and tag is NN',type:'emission'},
  {name:'word_is(VB,*)',desc:'Word is a known verb and tag is VB',type:'emission'},
  {name:'word_ends(-ly,RB)',desc:'Word ends in "-ly" and tag is RB',type:'morphology'},
  {name:'word_ends(-ing,VB)',desc:'Word ends in "-ing" and tag is VB',type:'morphology'},
  {name:'word_ends(-ed,VB)',desc:'Word ends in "-ed" and tag is VB',type:'morphology'},
  {name:'prev_tag(DT)→cur(NN)',desc:'Previous tag DT, current tag NN',type:'transition'},
  {name:'prev_tag(NN)→cur(VB)',desc:'Previous tag NN, current tag VB',type:'transition'},
  {name:'prev_tag(VB)→cur(RB)',desc:'Previous tag VB, current tag RB',type:'transition'},
  {name:'prev_tag(JJ)→cur(NN)',desc:'Previous tag JJ, current tag NN',type:'transition'},
  {name:'word_upper(NN)',desc:'Word starts uppercase, tag is NN (proper noun)',type:'morphology'},
  {name:'word_is(IN,in/on/at)',desc:'Word is preposition and tag is IN',type:'emission'},
];

const CRF_WEIGHTS = [2.8,2.5,2.6,3.1,2.7,2.4,2.9,2.8,2.3,2.7,2.2,2.6];

const QUIZ_QUESTIONS = [
  {id:1,tag:'HMM',question:'What does the Markov assumption in HMM mean for POS tagging?',options:['Each tag is independent of all other tags','Each tag depends only on the immediately preceding tag','Each word depends on all previous words','Each tag is generated from a uniform distribution'],answerIndex:1,explanation:'The first-order Markov assumption states P(tᵢ|t₁…tᵢ₋₁) = P(tᵢ|tᵢ₋₁). This allows efficient dynamic programming (Viterbi) but ignores longer-range dependencies.'},
  {id:2,tag:'HMM',question:'The Viterbi algorithm computes the best tag sequence in O(T·N²) time. Why not O(N^T)?',options:['It uses parallel computation across tags','It exploits the Markov property to reuse sub-problem results via dynamic programming','It prunes unlikely states below a threshold','It approximates the answer using beam search'],answerIndex:1,explanation:'Dynamic programming stores the best score for each (tag, position) pair — N×T values. At each step, we only look back one step (Markov property), so N² work per position, giving O(T·N²) total instead of the exponential O(N^T) brute force.'},
  {id:3,tag:'CRF',question:'Why are CRFs called discriminative models while HMMs are generative?',options:['CRFs discriminate between good and bad features; HMMs generate features','CRFs directly model P(tags|words) without modelling word generation; HMMs model the joint P(words,tags)','CRFs use binary features; HMMs use probability distributions','CRFs require labeled data; HMMs can train unsupervised'],answerIndex:1,explanation:'HMMs model the joint P(w,t) — they generate both words and tags. CRFs directly model the conditional P(t|w), skipping the word-generation model entirely. This makes CRFs more flexible and often more accurate, since they do not waste capacity modelling the input distribution.'},
  {id:4,tag:'CRF',question:'What is the partition function Z(w) in a CRF and why is computing it hard?',options:['The number of training examples, trivially computed as len(corpus)','The normalization constant summing exp(score) over all possible tag sequences — exponential to compute exactly','The Euclidean norm of the weight vector λ, used for L2 regularization','The total count of feature function activations in a sentence'],answerIndex:1,explanation:'Z(w) = Σ_t exp(score(t,w)) sums over all N^T possible tag sequences. Naively exponential, but the forward algorithm (analogous to Viterbi) solves it in O(T·N²) using the same dynamic programming insight.'},
  {id:5,tag:'HMM',question:'What happens when HMM encounters an unseen word (OOV) at test time?',options:['The model outputs a uniform tag distribution for that position','All emission probabilities P(word|tag) = 0, making joint probability 0 for any sequence — handled by smoothing or UNK tokens','The word is silently skipped and the adjacent tags are joined','The Viterbi score becomes 1.0 since no constraint is applied'],answerIndex:1,explanation:'Without smoothing, P(unseen_word|tag) = 0 for all tags, collapsing the entire path probability. Solutions include Laplace smoothing, replacing rare words with a special UNK token during training, or character-level emission models.'},
  {id:6,tag:'CRF',question:'What key advantage do CRF features have over HMM emission/transition tables?',options:['CRF features are always binary integers, making computation faster','CRF features can capture arbitrary context: word shape, neighboring words, regex patterns, morphology — anything about the whole sentence','CRF features are pre-trained on ImageNet and transferred','CRF features only look at pairs of adjacent tags, same as HMM'],answerIndex:1,explanation:'HMM emissions only look at the current word; transitions only look at the previous tag. CRF feature functions can look at any word in the sentence, use prefix/suffix patterns, capitalization, punctuation context, or even external gazetteers — giving CRFs far more expressive power for tagging.'},
];

const HMM_STEPS = ['Tokenize','Priors','Trellis','Backtrack','Result'];

const CRF_STEPS = ['Tokenize','Feature Extract','Score','Potentials','Viterbi/Result'];
// ─── HELPERS (extracted from the provided component) ──────────────────────
const norm = (values) => {
  const total = Object.values(values).reduce((sum, value) => sum + value, 0);
  return Object.keys(values).reduce((acc, key) => ({ ...acc, [key]: values[key] / total }), {});
};

const fmt4 = (value) => value.toFixed(4);

const fmt2 = (value) => value.toFixed(2);

const tokenize = (text) => text.toLowerCase().trim().split(/\s+/).filter(Boolean);

const lookupEmit = (tag, word) => HMM_EMIT[tag]?.[word] ?? 0;

const evalCRFFeatures = (word, tag, prevTag) => {
  let score = 0;
  const lower = word.toLowerCase();
  CRF_FEATURES.forEach((feature, index) => {
    let fire = 0;
    if (feature.type === 'emission') {
      if (feature.name.includes('the') && lower === 'the' && tag === 'DT') fire = 1;
      else if (feature.name.includes('NN') && Object.keys(HMM_EMIT.NN).includes(lower) && tag === 'NN') fire = 1;
      else if (feature.name.includes('VB') && Object.keys(HMM_EMIT.VB).includes(lower) && tag === 'VB') fire = 1;
      else if (feature.name.includes('IN') && ['in','on','at','of','by','for','with','from','about','into'].includes(lower) && tag === 'IN') fire = 1;
    } else if (feature.type === 'morphology') {
      if (feature.name.includes('-ly') && lower.endsWith('ly') && tag === 'RB') fire = 1;
      else if (feature.name.includes('-ing') && lower.endsWith('ing') && tag === 'VB') fire = 1;
      else if (feature.name.includes('-ed') && lower.endsWith('ed') && tag === 'VB') fire = 1;
      else if (feature.name.includes('upper') && word[0] === word[0].toUpperCase() && word[0] !== word[0].toLowerCase() && tag === 'NN') fire = 1;
    } else if (feature.type === 'transition') {
      if (feature.name.includes('DT') && feature.name.includes('NN') && prevTag === 'DT' && tag === 'NN') fire = 1;
      else if (feature.name.includes('NN') && feature.name.includes('VB') && prevTag === 'NN' && tag === 'VB') fire = 1;
      else if (feature.name.includes('VB') && feature.name.includes('RB') && prevTag === 'VB' && tag === 'RB') fire = 1;
      else if (feature.name.includes('JJ') && feature.name.includes('NN') && prevTag === 'JJ' && tag === 'NN') fire = 1;
    }
    score += fire * CRF_WEIGHTS[index];
  });
  return Math.exp(score);
};

const computeHmmSimData = (sentence) => {
  const tokens = tokenize(sentence);
  if (!tokens.length) return null;
  const T = tokens.length;
  const dp = Array.from({ length: T }, () => ({}));
  const bp = Array.from({ length: T }, () => ({}));
  const priorNorm = norm(HMM_PRIORS);

  TAGS.forEach((tag) => {
    const emit = lookupEmit(tag, tokens[0]);
    const smoothEmit = emit > 0 ? emit : 0.0001;
    dp[0][tag] = priorNorm[tag] * smoothEmit;
    bp[0][tag] = null;
  });

  for (let i = 1; i < T; i += 1) {
    TAGS.forEach((to) => {
      let bestScore = -1;
      let bestFrom = null;
      TAGS.forEach((from) => {
        const transition = HMM_TRANS[from]?.[to] ?? 0.0001;
        const score = dp[i - 1][from] * transition;
        if (score > bestScore) {
          bestScore = score;
          bestFrom = from;
        }
      });
      const emit = lookupEmit(to, tokens[i]);
      const smoothEmit = emit > 0 ? emit : 0.0001;
      dp[i][to] = bestScore * smoothEmit;
      bp[i][to] = bestFrom;
    });
  }

  let bestFinalTag = TAGS[0];
  for (let i = 1; i < TAGS.length; i += 1) {
    if (dp[T - 1][TAGS[i]] > dp[T - 1][bestFinalTag]) {
      bestFinalTag = TAGS[i];
    }
  }

  const path = [bestFinalTag];
  for (let i = T - 1; i > 0; i -= 1) {
    path.unshift(bp[i][path[0]]);
  }

  return { tokens, dp, bp, path, priorNorm };
};

const computeCrfPotentials = (sentence) => {
  const tokens = tokenize(sentence);
  const showTags = ['DT','NN','VB','JJ','RB','IN'];
  return tokens.map((word, index) => ({
    word,
    position: index,
    tagScores: showTags.map((tag) => {
      const pot = evalCRFFeatures(word, tag, index > 0 ? 'NN' : '<s>');
      return { tag, pot, pct: Math.round(Math.min(pot / 5, 1) * 100) };
    }),
  }));
};

const computeCrfSimData = (sentence) => {
  const tokens = tokenize(sentence);
  if (!tokens.length) return null;

  const path = [];
  let previousTag = null;
  tokens.forEach((word) => {
    let bestTag = TAGS[0];
    let bestScore = -1;
    TAGS.forEach((tag) => {
      const score = evalCRFFeatures(word, tag, previousTag || '<s>');
      if (score > bestScore) {
        bestScore = score;
        bestTag = tag;
      }
    });
    path.push({ word, tag: bestTag, score: bestScore });
    previousTag = bestTag;
  });

  const firedFeatures = tokens.map((word, index) => {
    const tag = path[index].tag;
    const prevTag = index > 0 ? path[index - 1].tag : '<s>';
    return CRF_FEATURES.map((feature) => {
      let fire = 0;
      const lower = word.toLowerCase();
      if (feature.type === 'emission') {
        if (feature.name.includes('the') && lower === 'the' && tag === 'DT') fire = 1;
        else if (feature.name.includes('NN') && Object.keys(HMM_EMIT.NN).includes(lower) && tag === 'NN') fire = 1;
        else if (feature.name.includes('VB') && Object.keys(HMM_EMIT.VB).includes(lower) && tag === 'VB') fire = 1;
        else if (feature.name.includes('IN') && ['in','on','at','of','by','for','with','from'].includes(lower) && tag === 'IN') fire = 1;
      } else if (feature.type === 'morphology') {
        if (feature.name.includes('-ly') && lower.endsWith('ly') && tag === 'RB') fire = 1;
        else if (feature.name.includes('-ing') && lower.endsWith('ing') && tag === 'VB') fire = 1;
        else if (feature.name.includes('-ed') && lower.endsWith('ed') && tag === 'VB') fire = 1;
      } else if (feature.type === 'transition') {
        if (feature.name.includes('DT') && feature.name.includes('NN') && prevTag === 'DT' && tag === 'NN') fire = 1;
        else if (feature.name.includes('NN') && feature.name.includes('VB') && prevTag === 'NN' && tag === 'VB') fire = 1;
        else if (feature.name.includes('VB') && feature.name.includes('RB') && prevTag === 'VB' && tag === 'RB') fire = 1;
      }
      return fire;
    });
  });

  return { tokens, path, firedFeatures };
};


// ─── DATA: new for the Basics tab ──────────────────────────────────────────
const TAGGER_APPROACHES = {
  hmm: { name:"Hidden Markov Model", color:"#7c3aed", bg:"#f5f0ff", bd:"#d4b8f8",
    desc:"A generative model: it learns how tags tend to produce words, then asks which tag sequence was most likely to generate this sentence.",
    pros:["Fast, exact inference (Viterbi)","No feature engineering needed"], cons:["Only looks at the previous tag"] },
  crf: { name:"Conditional Random Field", color:"#0e7490", bg:"#ecfeff", bd:"#a5f3fc",
    desc:"A discriminative model: it directly scores how well a tag sequence fits the words, using any feature you give it — prefixes, capitalization, neighboring words.",
    pros:["Rich, flexible features","Usually more accurate"], cons:["Needs hand-designed features"] },
};

const AMBIGUITY_EXAMPLES = [
  { sentence:"The dog runs fast.", word:"runs", tag:"VB", note:"Here \"runs\" comes right after a noun and acts as the main verb of the sentence." },
  { sentence:"She watched the final runs.", word:"runs", tag:"NN", note:"Here \"runs\" is the object of \"watched\" — a noun, not a verb. Same word, different job." },
];

const TAGGER_APPLICATIONS = [
  { icon:"🏷️", title:"Part-of-speech tagging", text:"Labeling each word as noun, verb, adjective, and so on — the running example on this page, and usually the first step in a syntactic pipeline." },
  { icon:"🏛️", title:"Named entity recognition", text:"Finding spans like person names, organizations, and locations in text. CRFs are especially popular here since they can use capitalization and context as features." },
  { icon:"🌳", title:"Chunking / shallow parsing", text:"Grouping words into phrases — noun phrases, verb phrases — without building a full parse tree. A lighter-weight structural step." },
  { icon:"🎙️", title:"Speech recognition", text:"The original home of HMMs — mapping a sequence of audio frames to the sequence of phonemes that produced them." },
  { icon:"🧬", title:"Bioinformatics", text:"Labeling regions of a DNA or protein sequence — gene vs. non-gene, for example. The same sequence-labeling idea, applied outside language." },
  { icon:"📄", title:"Information extraction", text:"Pulling structured fields — dates, prices, addresses — out of unstructured text like resumes, invoices, or forms." },
];

// ─── STATE ───────────────────────────────────────────────────────────────
const STATE = {
  tab: "bas",
  basics: { approach: "hmm", ambIdx: 0, tagIdx: 0, appIdx: 0 },
  hmmViz: { emissionTag: "NN", selectedTrans: null },
  hmmSim: { input: "the quick fox runs", step: 0, data: null },
  crfViz: { sentence: "the quick fox runs" },
  crfSim: { input: "the quick fox runs", step: 0, data: null },
};

const NAV_ITEMS = [
  { id:"bas",  icon:"📘", label:"Basics" },
  { id:"hmm",  icon:"🔮", label:"HMM" },
  { id:"crf",  icon:"🕸️", label:"CRF" },
  { id:"quiz", icon:"🧠", label:"Quiz" },
];

const HERO_META = {
  bas:  { pill:"Foundations",    sub:"Sequence Labeling",        title:"Tagging Words, One At A Time",  desc:"How a computer figures out which words are nouns, verbs, or adjectives — and why the words around them matter." },
  hmm:  { pill:"Probabilistic",  sub:"Hidden Markov Model",      title:"Inside The HMM Tagger",          desc:"From the raw probability tables to a full Viterbi walkthrough — everything the model uses to tag a sentence." },
  crf:  { pill:"Discriminative", sub:"Conditional Random Field", title:"Inside The CRF Tagger",          desc:"From hand-crafted features and learned weights to a full step-by-step tagging walkthrough." },
  quiz: { pill:"Check Yourself", sub:"Concept Quiz",             title:"Test What You Know",             desc:"Six questions on HMMs and CRFs, with instant feedback on every answer." },
};


// ─── APP SHELL ───────────────────────────────────────────────────────────
function initApp() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="sv-shell">
      <div class="sv-hero" id="heroBox"></div>
      <div class="sv-body">
        <aside class="sv-sidebar">
          <nav class="sv-navlist" id="navList"></nav>
          <div class="sv-navfoot">💡 New here? Start at Basics, then work down the list in order.</div>
        </aside>
        <main class="sv-main"><div class="sv-maininner" id="mainInner"></div></main>
      </div>
    </div>
  `;
  buildNav();
  buildPanels();
  updateHero();
  setActiveNav();
}

function buildNav() {
  document.getElementById("navList").innerHTML = NAV_ITEMS.map(item =>
    `<button class="sv-navitem" data-tab="${item.id}" onclick="switchTab('${item.id}')"><span class="ni-ic">${item.icon}</span>${escapeHtml(item.label)}</button>`
  ).join("");
}

function setActiveNav() {
  document.querySelectorAll(".sv-navitem").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === STATE.tab);
  });
}

function updateHero() {
  const h = HERO_META[STATE.tab];
  document.getElementById("heroBox").innerHTML = `
    <div class="sv-hero-pillrow">
      <span class="sv-hero-pill">${escapeHtml(h.pill)}</span>
      <span class="sv-hero-pillsub">${escapeHtml(h.sub)}</span>
    </div>
    <h1>${escapeHtml(h.title)}</h1>
    <p>${escapeHtml(h.desc)}</p>
  `;
}

function switchTab(id) {
  STATE.tab = id;
  updateHero();
  setActiveNav();
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.style.display = (p.dataset.tab === id) ? "" : "none";
  });
}
window.switchTab = switchTab;

function buildPanels() {
  document.getElementById("mainInner").innerHTML = `
    <section class="tab-panel" data-tab="bas" id="panel-bas"></section>
    <section class="tab-panel" data-tab="hmm" id="panel-hmm" style="display:none"></section>
    <section class="tab-panel" data-tab="crf" id="panel-crf" style="display:none"></section>
    <section class="tab-panel" data-tab="quiz" id="panel-quiz" style="display:none"></section>
  `;
  mountBasics();
  mountHmm();
  mountCrf();
  mountQuiz();
}

document.addEventListener("DOMContentLoaded", initApp);

// ─── TAB: BASICS ─────────────────────────────────────────────────────────
function taggerFlowSVG() {
  const branches = [
    { key:"hmm", x:120, label:"Hidden Markov Model", color:"#7c3aed", bg:"#f5f0ff" },
    { key:"crf", x:380, label:"Conditional Random Field", color:"#0e7490", bg:"#ecfeff" },
  ];
  const branchMarkup = branches.map(b => `
    <g class="flow-branch" onclick="selectTaggerApproach('${b.key}')" data-key="${b.key}">
      <rect id="tag-branch-rect-${b.key}" x="${b.x}" y="224" width="220" height="40" rx="8" fill="${b.bg}" stroke="${b.color}" stroke-width="1.5"></rect>
      <text id="tag-branch-text-${b.key}" x="${b.x + 110}" y="248" text-anchor="middle" fill="${b.color}" font-size="11.5" font-weight="700">${b.label}</text>
    </g>`).join("");
  return `
    <svg viewBox="0 0 720 340" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:640px;">
      <defs>
        <marker id="tagArrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 Z" fill="#6c63ff"></path>
        </marker>
      </defs>
      <rect x="260" y="8" width="200" height="42" rx="10" fill="#0f1123"></rect>
      <text x="360" y="34" text-anchor="middle" fill="#fff" font-size="13" font-weight="700">Sentence</text>
      <line x1="360" y1="50" x2="360" y2="72" stroke="#6c63ff" stroke-width="2" marker-end="url(#tagArrow)"></line>
      <rect x="235" y="74" width="250" height="46" rx="10" fill="#EEEDFE" stroke="#AFA9EC" stroke-width="1.5"></rect>
      <text x="360" y="94" text-anchor="middle" fill="#3C3489" font-size="12.5" font-weight="700">Tokenize</text>
      <text x="360" y="110" text-anchor="middle" fill="#6c63ff" font-size="10">split into words</text>
      <line x1="360" y1="120" x2="360" y2="142" stroke="#6c63ff" stroke-width="2" marker-end="url(#tagArrow)"></line>
      <rect x="210" y="144" width="300" height="42" rx="10" fill="#6c63ff"></rect>
      <text x="360" y="170" text-anchor="middle" fill="#fff" font-size="12.5" font-weight="700">Pick A Model — click one ↓</text>
      <line x1="360" y1="186" x2="360" y2="204" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="230" y1="204" x2="490" y2="204" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="230" y1="204" x2="230" y2="222" stroke="#6c63ff" stroke-width="1.5" marker-end="url(#tagArrow)"></line>
      <line x1="490" y1="204" x2="490" y2="222" stroke="#6c63ff" stroke-width="1.5" marker-end="url(#tagArrow)"></line>
      ${branchMarkup}
      <line x1="230" y1="264" x2="230" y2="280" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="490" y1="264" x2="490" y2="280" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="230" y1="280" x2="490" y2="280" stroke="#6c63ff" stroke-width="1.5"></line>
      <line x1="360" y1="280" x2="360" y2="296" stroke="#6c63ff" stroke-width="2" marker-end="url(#tagArrow)"></line>
      <rect x="260" y="298" width="200" height="38" rx="10" fill="#0f1123"></rect>
      <text x="360" y="321" text-anchor="middle" fill="#fff" font-size="13" font-weight="700">Tag Sequence</text>
    </svg>`;
}

function mountBasics() {
  const panel = document.getElementById("panel-bas");
  panel.innerHTML = `
    <div class="sv-eyebrow">Where It Sits In NLP</div>
    <div class="sv-card">
      <div class="sv-card-title">One Step In A Longer Pipeline</div>
      <p class="sv-desc">Sequence labeling usually happens right after tokenization — its output feeds every step that comes after it.</p>
      <div class="pipeline-row">
        <div class="pipeline-step">Raw Text</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Tokenization</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step hl">Sequence Labeling<span>HMM / CRF</span></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Parsing / Chunking</div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step">Downstream Tasks</div>
      </div>
      <div class="sv-insight i-hmm">💡 Get the tags wrong here and every later step — parsing, extraction, translation — inherits the mistake. That's why this step gets its own dedicated models instead of a guess.</div>
    </div>

    <div class="sv-eyebrow">How It Works, At A Glance</div>
    <div class="sv-card">
      <div class="sv-card-title">From Words To Tags</div>
      <p class="sv-desc">Sequence labeling assigns a grammatical tag — noun, verb, adjective — to every word in a sentence. Click a box below to preview each modeling approach.</p>
      <div class="flow-wrap">${taggerFlowSVG()}</div>
      <div class="sblock" id="taggerApproachInfo"></div>
      <div class="sv-insight i-bas">👉 Explore the HMM in full on the <strong>HMM</strong> page, and the CRF in full on the <strong>CRF</strong> page.</div>
    </div>

    <div class="sv-eyebrow">Why Context Matters</div>
    <div class="sv-card">
      <div class="sv-card-title">Click To See The Ambiguity</div>
      <p class="sv-desc">The same word can be a different part of speech depending on what's around it. That's the whole challenge of tagging.</p>
      <div class="word-grid" id="ambChips"></div>
      <div class="sblock" id="ambSentence" style="font-size:13px;"></div>
      <div class="sv-insight i-bas" id="ambNote"></div>
    </div>

    <div class="sv-eyebrow">The Tag Set</div>
    <div class="sv-card">
      <div class="sv-card-title">Click A Tag To See Examples</div>
      <p class="sv-desc">This page works with 8 common part-of-speech tags. Click one to see words that typically carry it.</p>
      <div class="grid3c" id="tagGrid" style="grid-template-columns:repeat(4,1fr);"></div>
      <div class="sblock" id="tagPanel" style="margin-top:10px;margin-bottom:0;"></div>
    </div>

    <div class="sv-eyebrow">Where It's Used</div>
    <div class="sv-card">
      <div class="sv-card-title">Click To See Where It's Used</div>
      <p class="sv-desc">Sequence labeling shows up anywhere you need a label for every item in an ordered list — not just words.</p>
      <div class="grid3c" id="appGrid"></div>
      <div class="sblock" id="appPanel" style="margin-top:10px;margin-bottom:0;"></div>
    </div>

    <div class="sv-eyebrow">Why It Still Matters</div>
    <div class="sv-card">
      <div class="sv-card-title">A Foundation, Not A Relic</div>
      <p class="sv-desc">HMMs and CRFs aren't just a history lesson — the ideas they introduced are baked into how modern NLP systems are built.</p>
      <div class="tl-row"><div class="tl-dot" style="background:var(--hmm);"></div><div class="tl-content">They taught the field that labeling a <em>sequence</em> jointly (with Viterbi-style decoding) beats guessing each word's tag independently.</div></div>
      <div class="tl-row"><div class="tl-dot" style="background:var(--crf);"></div><div class="tl-content">Modern neural taggers still bolt a CRF layer on top of learned features — "BiLSTM-CRF" and "Transformer-CRF" are standard architecture names today.</div></div>
      <div class="tl-row"><div class="tl-dot" style="background:var(--pos);"></div><div class="tl-content">They're fast, interpretable, and need very little data — still the first baseline most NLP courses and projects build.</div></div>
    </div>
  `;
  renderTaggerBranches();
  updateTaggerApproachInfo();
  renderAmbChips();
  updateAmbExplorer();
  renderTagGrid();
  updateTagPanel();
  renderTaggerAppGrid();
  updateTaggerAppPanel();
}

function selectTaggerApproach(key) {
  STATE.basics.approach = key;
  renderTaggerBranches();
  updateTaggerApproachInfo();
}
window.selectTaggerApproach = selectTaggerApproach;

function renderTaggerBranches() {
  ["hmm","crf"].forEach(k => {
    const a = TAGGER_APPROACHES[k];
    const rect = document.getElementById(`tag-branch-rect-${k}`);
    const text = document.getElementById(`tag-branch-text-${k}`);
    const isSel = STATE.basics.approach === k;
    rect.setAttribute("fill", isSel ? a.color : a.bg);
    rect.setAttribute("stroke-width", isSel ? "2.5" : "1.5");
    text.setAttribute("fill", isSel ? "#fff" : a.color);
  });
}

function updateTaggerApproachInfo() {
  const a = TAGGER_APPROACHES[STATE.basics.approach];
  const prosChips = a.pros.map(p => `<span class="chip c-pos" style="font-size:10px;">${escapeHtml(p)}</span>`).join("");
  const consChips = a.cons.map(p => `<span class="chip c-neg" style="font-size:10px;">${escapeHtml(p)}</span>`).join("");
  document.getElementById("taggerApproachInfo").innerHTML = `
    <span class="sv-badge" style="background:${a.bg};color:${a.color};border:0.5px solid ${a.bd};">${escapeHtml(a.name)}</span>
    <div style="font-size:12.5px;color:var(--color-text-secondary,#5a5950);margin:4px 0 8px;line-height:1.6;">${escapeHtml(a.desc)}</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;">${prosChips}${consChips}</div>
  `;
}

function renderAmbChips() {
  document.getElementById("ambChips").innerHTML = AMBIGUITY_EXAMPLES.map((e, i) => {
    const active = i === STATE.basics.ambIdx;
    return `<button class="chip ${active ? "c-bas" : "c-gray"}" style="font-weight:${active ? 600 : 500};" onclick="selectAmb(${i})">${escapeHtml(e.sentence)}</button>`;
  }).join("");
}

function selectAmb(i) {
  STATE.basics.ambIdx = i;
  renderAmbChips();
  updateAmbExplorer();
}
window.selectAmb = selectAmb;

function updateAmbExplorer() {
  const e = AMBIGUITY_EXAMPLES[STATE.basics.ambIdx];
  const highlighted = e.sentence.replace(
    new RegExp("\\b" + e.word + "\\b", "i"),
    m => `<span class="chip c-bas" style="font-weight:700;">${escapeHtml(m)} → ${e.tag}</span>`
  );
  document.getElementById("ambSentence").innerHTML = highlighted;
  document.getElementById("ambNote").textContent = e.note;
}

function renderTagGrid() {
  document.getElementById("tagGrid").innerHTML = TAGS.map((tag, i) => `
    <button class="pick-card${i === STATE.basics.tagIdx ? " active" : ""}" onclick="selectTag(${i})">
      <div class="pc-title" style="font-family:var(--mono);">${tag}</div>
    </button>`).join("");
}

function selectTag(i) {
  STATE.basics.tagIdx = i;
  renderTagGrid();
  updateTagPanel();
}
window.selectTag = selectTag;

function updateTagPanel() {
  const tag = TAGS[STATE.basics.tagIdx];
  const examples = Object.entries(HMM_EMIT[tag] || {}).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([w]) => w);
  document.getElementById("tagPanel").innerHTML = `
    <div class="sblock-lbl">${tag} — ${escapeHtml(TAG_NAMES[tag])}</div>
    <div class="word-grid">${examples.map(w => `<span class="chip c-gray c-mono">${escapeHtml(w)}</span>`).join("")}</div>
  `;
}

function renderTaggerAppGrid() {
  document.getElementById("appGrid").innerHTML = TAGGER_APPLICATIONS.map((a, i) => `
    <button class="pick-card${i === STATE.basics.appIdx ? " active" : ""}" onclick="selectTaggerApp(${i})">
      <div class="pc-ic">${a.icon}</div>
      <div class="pc-title">${escapeHtml(a.title)}</div>
    </button>`).join("");
}

function selectTaggerApp(i) {
  STATE.basics.appIdx = i;
  renderTaggerAppGrid();
  updateTaggerAppPanel();
}
window.selectTaggerApp = selectTaggerApp;

function updateTaggerAppPanel() {
  const a = TAGGER_APPLICATIONS[STATE.basics.appIdx];
  document.getElementById("appPanel").innerHTML = `
    <div class="sblock-lbl">${escapeHtml(a.title)}</div>
    <div style="font-size:12.5px;color:var(--color-text-secondary,#5a5950);line-height:1.6;">${escapeHtml(a.text)}</div>
  `;
}

// ─── PAGE: HMM (tables + step simulator, one continuous page) ─────────────
const PRIOR_NORM = norm(HMM_PRIORS);

function mountHmm() {
  const panel = document.getElementById("panel-hmm");
  panel.innerHTML = `
    <div class="sv-eyebrow">Before Reading A Word</div>
    <div class="sv-card">
      <div class="sv-card-title">HMM Priors</div>
      <p class="sv-desc">How likely is each tag to start a sentence, before the model has seen a single word?</p>
      <div class="grid3c" id="hmmPriors" style="grid-template-columns:repeat(4,1fr);"></div>
    </div>

    <div class="sv-eyebrow">Tag-To-Tag Patterns</div>
    <div class="sv-card">
      <div class="sv-card-title">Transition Matrix <em>— click a cell</em></div>
      <p class="sv-desc">P(next tag | current tag). Determiners are almost always followed by nouns or adjectives — click around and see the pattern.</p>
      <div class="mtx-wrap"><div class="mtx-grid" id="hmmTransGrid" style="grid-template-columns:80px repeat(8, minmax(70px,1fr));"></div></div>
      <div class="sv-insight i-hmm" id="hmmTransInsight"></div>
    </div>

    <div class="sv-eyebrow">What Each Tag "Sounds Like"</div>
    <div class="sv-card">
      <div class="sv-card-title">Emission Distributions</div>
      <p class="sv-desc">P(word | tag) — how likely a tag is to produce a given word.</p>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px;">
        <select class="sv-select" id="hmmEmissionTag" style="min-width:180px;"></select>
        <span style="font-size:12px;color:var(--color-text-tertiary,#8a8878);">Select a tag to inspect its words</span>
      </div>
      <div class="tbl-wrap"><div class="tbl-scroll">
        <table class="sv-table"><thead><tr><th>Word</th><th>P(word|tag)</th><th>Strength</th></tr></thead><tbody id="hmmEmissionBody"></tbody></table>
      </div></div>
    </div>

    <div class="sv-eyebrow">Put It All Together</div>
    <div class="sv-card">
      <div class="sv-card-title">Step Through Viterbi</div>
      <p class="sv-desc">Same priors, transitions, and emissions from above — now watch them combine to tag a real sentence.</p>
      <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
        <input class="sv-input sv-mono" id="hmmSimInput" style="font-size:12.5px;">
        <button class="sv-btn" style="background:var(--hmm);color:#fff;" onclick="runHmmSim()">Run Inference</button>
      </div>
      <div class="sv-stepper" id="hmmSimStepper"></div>
    </div>
    <div id="hmmSimContent"></div>
    <div class="sim-nav">
      <button class="sv-btn btn-sec" id="hmmSimPrevBtn" onclick="hmmSimNav(-1)">← Previous</button>
      <span class="sim-nav-label" id="hmmSimLabel"></span>
      <button class="sv-btn" style="background:var(--hmm);color:#fff;" id="hmmSimNextBtn" onclick="hmmSimNav(1)">Next →</button>
    </div>
  `;
  renderHmmPriors();
  renderHmmTransGrid();
  const sel = document.getElementById("hmmEmissionTag");
  sel.innerHTML = TAGS.map(t => `<option value="${t}">${t} — ${escapeHtml(TAG_NAMES[t])}</option>`).join("");
  sel.value = STATE.hmmViz.emissionTag;
  sel.addEventListener("change", e => { STATE.hmmViz.emissionTag = e.target.value; renderHmmEmissionTable(); });
  renderHmmEmissionTable();

  document.getElementById("hmmSimInput").value = STATE.hmmSim.input;
  document.getElementById("hmmSimInput").addEventListener("input", e => {
    STATE.hmmSim.input = e.target.value;
    STATE.hmmSim.data = null;
    STATE.hmmSim.step = 0;
    renderHmmSimStepper();
  });
  runHmmSim();
}

function renderHmmPriors() {
  document.getElementById("hmmPriors").innerHTML = TAGS.map(tag => {
    const v = PRIOR_NORM[tag];
    const pct = Math.min(Math.round(v * 100), 100);
    return `
      <div class="metric">
        <div class="metric-lbl">${tag}</div>
        <div style="font-family:var(--disp);font-size:20px;font-weight:600;color:var(--hmm);">${fmt4(v)}</div>
        <div class="meter-track" style="margin-top:8px;"><div class="meter-fill" style="width:${pct}%;background:var(--hmm);"></div></div>
        <div style="font-size:10.5px;color:var(--color-text-tertiary,#8a8878);margin-top:6px;">${escapeHtml(TAG_NAMES[tag])}</div>
      </div>`;
  }).join("");
}

function renderHmmTransGrid() {
  const cells = [`<div class="mtx-cell mtx-head">t → t'</div>`];
  TAGS.forEach(tag => cells.push(`<div class="mtx-cell mtx-head">${tag}</div>`));
  TAGS.forEach(from => {
    cells.push(`<div class="mtx-cell mtx-head">${from}</div>`);
    TAGS.forEach(to => {
      const v = HMM_TRANS[from]?.[to] ?? 0;
      const sel = STATE.hmmViz.selectedTrans && STATE.hmmViz.selectedTrans[0] === from && STATE.hmmViz.selectedTrans[1] === to;
      cells.push(`<button class="mtx-cell mtx-btn${sel ? " sel" : ""}" onclick="selectHmmTrans('${from}','${to}')"><span style="color:${v > 0.25 ? "var(--hmm)" : "inherit"};font-weight:600;">${fmt4(v)}</span></button>`);
    });
  });
  document.getElementById("hmmTransGrid").innerHTML = cells.join("");
  updateHmmTransInsight();
}

function selectHmmTrans(from, to) {
  STATE.hmmViz.selectedTrans = [from, to];
  renderHmmTransGrid();
}
window.selectHmmTrans = selectHmmTrans;

function updateHmmTransInsight() {
  const sel = STATE.hmmViz.selectedTrans;
  const el = document.getElementById("hmmTransInsight");
  if (!sel) {
    el.textContent = "💡 Click any transition cell to inspect the probability of a tag following the current tag.";
    return;
  }
  const [from, to] = sel;
  const p = HMM_TRANS[from]?.[to] ?? 0;
  el.textContent = `💡 P(${to}|${from}) = ${fmt4(p)} — given ${from} (${TAG_NAMES[from]}), the next tag is most likely ${to} (${TAG_NAMES[to]}).`;
}

function renderHmmEmissionTable() {
  const tag = STATE.hmmViz.emissionTag;
  const entries = Object.entries(HMM_EMIT[tag]).sort((a, b) => b[1] - a[1]);
  document.getElementById("hmmEmissionBody").innerHTML = entries.map(([word, raw]) => {
    const pct = Math.min(Math.round(raw * 300), 100);
    return `
      <tr>
        <td>${escapeHtml(word)}</td>
        <td class="sv-mono">${fmt4(raw)}</td>
        <td><div class="meter-track" style="width:120px;"><div class="meter-fill" style="width:${pct}%;background:var(--hmm);"></div></div></td>
      </tr>`;
  }).join("");
}

// ─── HMM simulator logic (mounted as part of mountHmm above) ───────────────
function runHmmSim() {
  STATE.hmmSim.data = computeHmmSimData(STATE.hmmSim.input);
  STATE.hmmSim.step = 0;
  renderHmmSimStepper();
}
window.runHmmSim = runHmmSim;

function renderHmmSimStepper() {
  document.getElementById("hmmSimStepper").innerHTML = HMM_STEPS.map((label, i) => {
    const cls = i === STATE.hmmSim.step ? "active" : i < STATE.hmmSim.step ? "done" : "";
    return `${i > 0 ? `<div class="sconn${i <= STATE.hmmSim.step ? " done" : ""}"></div>` : ""}
      <div class="snode ${cls}" onclick="jumpHmmStep(${i})">
        <div class="scircle">${i < STATE.hmmSim.step ? "✓" : i + 1}</div>
        <div class="slabel">${escapeHtml(label)}</div>
      </div>`;
  }).join("");
  document.getElementById("hmmSimLabel").textContent = `Step ${STATE.hmmSim.step + 1} of ${HMM_STEPS.length}`;
  document.getElementById("hmmSimPrevBtn").disabled = STATE.hmmSim.step === 0;
  document.getElementById("hmmSimNextBtn").disabled = STATE.hmmSim.step === HMM_STEPS.length - 1;
  renderHmmSimContent();
}

function jumpHmmStep(i) { STATE.hmmSim.step = i; renderHmmSimStepper(); }
window.jumpHmmStep = jumpHmmStep;

function hmmSimNav(delta) {
  STATE.hmmSim.step = Math.max(0, Math.min(HMM_STEPS.length - 1, STATE.hmmSim.step + delta));
  renderHmmSimStepper();
}
window.hmmSimNav = hmmSimNav;

function renderHmmSimContent() {
  const container = document.getElementById("hmmSimContent");
  const data = STATE.hmmSim.data;
  if (!data) { container.innerHTML = `<div class="sv-card"><div class="sv-insight i-info">Enter a sentence and run the HMM simulator.</div></div>`; return; }
  const { tokens, dp, bp, path } = data;
  const T = tokens.length;
  const step = STATE.hmmSim.step;

  if (step === 0) {
    const chips = tokens.map(word => {
      const known = TAGS.some(tag => lookupEmit(tag, word) > 0);
      return `<div style="min-width:80px;text-align:center;">
        <div style="padding:10px 12px;border-radius:12px;border:0.5px solid var(--color-border-tertiary,#dddbd2);background:var(--color-background-primary,#faf9f5);font-weight:600;">${escapeHtml(word)}</div>
        <div style="margin-top:6px;font-size:11px;color:var(--color-text-tertiary,#8a8878);">${known ? "in vocab" : "OOV"}</div>
      </div>`;
    }).join("");
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-hmm">Step 1 — Tokenize</div>
        <p class="sv-desc">Split the input into tokens.</p>
        <div style="display:flex;flex-wrap:wrap;gap:10px;">${chips}</div>
        <div class="sv-insight i-hmm">💡 The HMM emission table covers many word-tag pairs. OOV words default to a small smoothing probability (0.0001) across all tags.</div>
      </div>`;
    return;
  }

  if (step === 1) {
    const sorted = [...TAGS].sort((a, b) => dp[0][b] - dp[0][a]);
    const rows = sorted.map(tag => {
      const emit = lookupEmit(tag, tokens[0]);
      return `<tr>
        <td><span class="chip c-hmm">${tag}</span></td>
        <td class="sv-mono">${fmt4(PRIOR_NORM[tag])}</td>
        <td class="sv-mono">${emit > 0 ? fmt4(emit) : "0.0001 (smoothed)"}</td>
        <td class="sv-mono">${dp[0][tag].toExponential(4)}</td>
      </tr>`;
    }).join("");
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-hmm">Step 2 — Priors</div>
        <p class="sv-desc">Initial Viterbi scores for the first token.</p>
        <div class="tbl-wrap"><div class="tbl-scroll">
          <table class="sv-table"><thead><tr><th>Tag</th><th>π(tag)</th><th>P(word|tag)</th><th>Score</th></tr></thead><tbody>${rows}</tbody></table>
        </div></div>
        <div class="sv-insight i-hmm">💡 The best first-step tag is shown above, but the final path also depends on transitions later in the sentence.</div>
      </div>`;
    return;
  }

  if (step === 2) {
    let cells = [`<div class="mtx-cell mtx-head"></div>`];
    tokens.forEach((word, i) => cells.push(`<div class="mtx-cell mtx-head">${i + 1}. ${escapeHtml(word)}</div>`));
    TAGS.forEach(tag => {
      cells.push(`<div class="mtx-cell mtx-head">${tag}</div>`);
      tokens.forEach((_, i) => {
        const onPath = path[i] === tag;
        cells.push(`<div class="mtx-cell${onPath ? " mtx-onpath" : ""}">
          <div class="mtx-val" style="font-weight:600;">${dp[i][tag].toExponential(2)}</div>
          ${i > 0 ? `<div style="margin-top:4px;font-size:10px;color:var(--color-text-tertiary,#8a8878);">← ${bp[i][tag] ?? "–"}</div>` : ""}
        </div>`);
      });
    });
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-hmm">Step 3 — Trellis</div>
        <p class="sv-desc">Viterbi trellis values and backpointers.</p>
        <div class="mtx-wrap"><div class="mtx-grid" style="grid-template-columns:90px repeat(${T},minmax(90px,1fr));">${cells.join("")}</div></div>
        <div class="sv-insight i-hmm">💡 Each trellis cell stores the best path score and a backpointer to the best previous tag.</div>
      </div>`;
    return;
  }

  if (step === 3) {
    const rows = path.map((tag, i) => `
      <div style="padding:10px;border-radius:10px;background:var(--color-background-primary,#faf9f5);border:0.5px solid var(--color-border-tertiary,#dddbd2);display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:12px;color:var(--color-text-tertiary,#8a8878);">Position ${i + 1}</div>
          <div style="font-weight:700;">${escapeHtml(tokens[i])} → ${tag}</div>
        </div>
        <div class="sv-mono" style="color:var(--color-text-tertiary,#8a8878);">${bp[i] && bp[i][tag] ? `from ${bp[i][tag]}` : ""}</div>
      </div>`).join("");
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-hmm">Step 4 — Backtrack</div>
        <p class="sv-desc">Reconstruct the best tag sequence by following backpointers from the end.</p>
        <div style="display:grid;gap:8px;">${rows}</div>
        <div class="sv-insight i-hmm">💡 Backtracking uses the stored pointers to rebuild the best tag sequence from the final state backward.</div>
      </div>`;
    return;
  }

  const results = tokens.map((word, i) => `
    <div style="min-width:90px;padding:12px;border-radius:14px;background:var(--color-background-primary,#faf9f5);border:0.5px solid var(--color-border-tertiary,#dddbd2);text-align:center;">
      <div style="margin-bottom:6px;font-weight:700;">${escapeHtml(word)}</div>
      <div style="padding:6px 10px;border-radius:10px;background:var(--hmm-bg);color:var(--hmm);font-weight:700;">${path[i]}</div>
      <div class="sv-mono" style="margin-top:8px;font-size:11px;color:var(--color-text-tertiary,#8a8878);">${dp[i][path[i]].toExponential(2)}</div>
    </div>`).join("");
  container.innerHTML = `
    <div class="sv-card">
      <div class="sv-badge b-pos">Step 5 — Result</div>
      <p class="sv-desc">Predicted part-of-speech sequence.</p>
      <div style="display:flex;flex-wrap:wrap;gap:10px;">${results}</div>
      <div class="sv-insight i-hmm">💡 The final HMM score is the product of the best path scores. This is an illustrative model — weights are hand-set, not trained from data.</div>
    </div>`;
}

// ─── PAGE: CRF (tables + step simulator, one continuous page) ─────────────
function mountCrf() {
  const panel = document.getElementById("panel-crf");
  panel.innerHTML = `
    <div class="sv-eyebrow">What The CRF Looks At</div>
    <div class="sv-card">
      <div class="sv-card-title">Feature Functions</div>
      <p class="sv-desc">CRFs score a tag using feature functions — small yes/no checks about the word, its shape, or the previous tag.</p>
      <div class="tbl-wrap"><div class="tbl-scroll">
        <table class="sv-table"><thead><tr><th>Feature</th><th>Description</th><th>Type</th></tr></thead>
        <tbody>
          ${CRF_FEATURES.map(f => `<tr><td class="sv-mono">${escapeHtml(f.name)}</td><td>${escapeHtml(f.desc)}</td><td><span class="chip ${f.type === "emission" ? "c-hmm" : f.type === "transition" ? "c-pos" : "c-crf"}">${escapeHtml(f.type)}</span></td></tr>`).join("")}
        </tbody></table>
      </div></div>
    </div>

    <div class="sv-eyebrow">How Much Each Feature Counts</div>
    <div class="sv-card">
      <div class="sv-card-title">Feature Weights</div>
      <p class="sv-desc">Each feature has a learned weight λ that scales how much it contributes to a tag's score.</p>
      <div class="tbl-wrap"><div class="tbl-scroll">
        <table class="sv-table"><thead><tr><th>Feature</th><th>Weight</th><th>Strength</th></tr></thead>
        <tbody>
          ${CRF_FEATURES.map((f, i) => {
            const w = CRF_WEIGHTS[i];
            const pct = Math.round((w / Math.max(...CRF_WEIGHTS)) * 100);
            return `<tr><td class="sv-mono">${escapeHtml(f.name)}</td><td class="sv-mono">${fmt2(w)}</td><td><div class="meter-track" style="width:120px;"><div class="meter-fill" style="width:${pct}%;background:var(--crf);"></div></div></td></tr>`;
          }).join("")}
        </tbody></table>
      </div></div>
    </div>

    <div class="sv-eyebrow">The Math</div>
    <div class="sv-card">
      <div class="sv-card-title">Score → Probability</div>
      <div class="grid2">
        <div class="sblock"><div class="sblock-lbl">CRF score (unnormalized)</div><code class="sv-code" style="display:block;padding:8px;margin-top:6px;">score(t,w) = Σₖ λₖ·fₖ(tᵢ,tᵢ₋₁,w,i)</code></div>
        <div class="sblock"><div class="sblock-lbl">Posterior</div><code class="sv-code" style="display:block;padding:8px;margin-top:6px;">P(t|w) = exp(score) / Z(w)</code></div>
      </div>
    </div>

    <div class="sv-eyebrow">Try It On Your Own Sentence</div>
    <div class="sv-card">
      <div class="sv-card-title">Feature Potentials Per Word</div>
      <p class="sv-desc">Edit the sentence and watch how the candidate-tag scores change for each word.</p>
      <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;">
        <input class="sv-input" id="crfVizSentence">
      </div>
      <div id="crfVizPotentials" style="display:grid;gap:10px;"></div>
    </div>

    <div class="sv-eyebrow">Put It All Together</div>
    <div class="sv-card">
      <div class="sv-card-title">Step Through The CRF</div>
      <p class="sv-desc">Same features and weights from above — now watch them combine to tag a real sentence.</p>
      <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
        <input class="sv-input sv-mono" id="crfSimInput" style="font-size:12.5px;">
        <button class="sv-btn" style="background:var(--crf);color:#fff;" onclick="runCrfSim()">Run Inference</button>
      </div>
      <div class="sv-stepper" id="crfSimStepper"></div>
    </div>
    <div id="crfSimContent"></div>
    <div class="sim-nav">
      <button class="sv-btn btn-sec" id="crfSimPrevBtn" onclick="crfSimNav(-1)">← Previous</button>
      <span class="sim-nav-label" id="crfSimLabel"></span>
      <button class="sv-btn" style="background:var(--crf);color:#fff;" id="crfSimNextBtn" onclick="crfSimNav(1)">Next →</button>
    </div>
  `;
  const input = document.getElementById("crfVizSentence");
  input.value = STATE.crfViz.sentence;
  input.addEventListener("input", e => { STATE.crfViz.sentence = e.target.value; renderCrfVizPotentials(); });
  renderCrfVizPotentials();

  document.getElementById("crfSimInput").value = STATE.crfSim.input;
  document.getElementById("crfSimInput").addEventListener("input", e => {
    STATE.crfSim.input = e.target.value;
    STATE.crfSim.data = null;
    STATE.crfSim.step = 0;
    renderCrfSimStepper();
  });
  runCrfSim();
}

function renderCrfVizPotentials() {
  const rows = computeCrfPotentials(STATE.crfViz.sentence);
  document.getElementById("crfVizPotentials").innerHTML = rows.map(position => `
    <div class="sblock">
      <div class="sblock-lbl">Position ${position.position + 1}: "${escapeHtml(position.word)}"</div>
      <div class="grid3c" style="grid-template-columns:repeat(auto-fit,minmax(96px,1fr));">
        ${position.tagScores.map(s => `
          <div class="tag-pill">
            <div style="font-size:10px;color:var(--color-text-tertiary,#8a8878);margin-bottom:4px;">${s.tag}</div>
            <div class="sv-mono" style="font-size:13px;font-weight:700;color:var(--crf);">${fmt2(s.pot)}</div>
            <div class="meter-track" style="margin-top:8px;"><div class="meter-fill" style="width:${s.pct}%;background:var(--crf);"></div></div>
          </div>`).join("")}
      </div>
    </div>`).join("");
}

// ─── CRF simulator logic (mounted as part of mountCrf above) ──────────────
function runCrfSim() {
  STATE.crfSim.data = computeCrfSimData(STATE.crfSim.input);
  STATE.crfSim.step = 0;
  renderCrfSimStepper();
}
window.runCrfSim = runCrfSim;

function renderCrfSimStepper() {
  document.getElementById("crfSimStepper").innerHTML = CRF_STEPS.map((label, i) => {
    const cls = i === STATE.crfSim.step ? "active" : i < STATE.crfSim.step ? "done" : "";
    return `${i > 0 ? `<div class="sconn${i <= STATE.crfSim.step ? " done" : ""}"></div>` : ""}
      <div class="snode ${cls}" onclick="jumpCrfStep(${i})">
        <div class="scircle">${i < STATE.crfSim.step ? "✓" : i + 1}</div>
        <div class="slabel">${escapeHtml(label)}</div>
      </div>`;
  }).join("");
  document.getElementById("crfSimLabel").textContent = `Step ${STATE.crfSim.step + 1} of ${CRF_STEPS.length}`;
  document.getElementById("crfSimPrevBtn").disabled = STATE.crfSim.step === 0;
  document.getElementById("crfSimNextBtn").disabled = STATE.crfSim.step === CRF_STEPS.length - 1;
  renderCrfSimContent();
}

function jumpCrfStep(i) { STATE.crfSim.step = i; renderCrfSimStepper(); }
window.jumpCrfStep = jumpCrfStep;

function crfSimNav(delta) {
  STATE.crfSim.step = Math.max(0, Math.min(CRF_STEPS.length - 1, STATE.crfSim.step + delta));
  renderCrfSimStepper();
}
window.crfSimNav = crfSimNav;

function renderCrfSimContent() {
  const container = document.getElementById("crfSimContent");
  const data = STATE.crfSim.data;
  if (!data) { container.innerHTML = `<div class="sv-card"><div class="sv-insight i-info">Enter a sentence and run the CRF simulator.</div></div>`; return; }
  const { tokens, path, firedFeatures } = data;
  const step = STATE.crfSim.step;

  if (step === 0) {
    const chips = tokens.map((word, i) => `
      <div style="min-width:100px;text-align:center;">
        <div style="padding:10px 12px;background:var(--color-background-primary,#faf9f5);border:0.5px solid var(--color-border-tertiary,#dddbd2);border-radius:12px;font-weight:600;">${escapeHtml(word)}</div>
        <div style="margin-top:6px;font-size:11px;color:var(--color-text-tertiary,#8a8878);">pos ${i + 1}</div>
      </div>`).join("");
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-crf">Step 1 — Tokenize</div>
        <p class="sv-desc">Tokenization, with sentence-wide feature context available to every token.</p>
        <div style="display:flex;flex-wrap:wrap;gap:10px;">${chips}</div>
        <div class="sv-insight i-crf">💡 CRFs can inspect the full sentence and morphology before scoring each token — unlike HMMs, which only look at the current word.</div>
      </div>`;
    return;
  }

  if (step === 1) {
    const blocks = path.map((entry, index) => {
      const fired = firedFeatures[index];
      const firedRows = CRF_FEATURES.map((f, idx) => fired[idx] ? `
        <div class="feat-row"><div>${escapeHtml(f.name)}</div><div class="sv-mono" style="color:var(--crf);">λ=${fmt2(CRF_WEIGHTS[idx])}</div></div>` : "").join("");
      const anyFired = fired.some(f => f === 1);
      return `
        <div class="sblock">
          <div style="margin-bottom:8px;font-weight:700;">${index + 1}. ${escapeHtml(entry.word)} → <span class="chip c-crf">${entry.tag}</span></div>
          ${anyFired ? firedRows : `<div style="font-size:12px;color:var(--color-text-tertiary,#8a8878);">No features fired for this predicted tag.</div>`}
        </div>`;
    }).join("");
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-crf">Step 2 — Feature Extract</div>
        <p class="sv-desc">Feature extraction for each token.</p>
        ${blocks}
        <div class="sv-insight i-crf">💡 The CRF score is the weighted sum of the active feature functions for each token.</div>
      </div>`;
    return;
  }

  if (step === 2) {
    const blocks = path.map((entry, index) => {
      const cells = ["DT","NN","VB","JJ","RB"].map(tag => {
        const pot = evalCRFFeatures(entry.word, tag, index > 0 ? path[index - 1].tag : "<s>");
        const sel = tag === entry.tag;
        return `<div class="tag-pill${sel ? " sel" : ""}"><div style="font-size:11px;font-weight:700;">${tag}</div><div class="sv-mono tp-val" style="margin-top:6px;font-size:13px;">${fmt2(pot)}</div></div>`;
      }).join("");
      return `<div class="sblock"><div style="margin-bottom:8px;font-weight:700;">Position ${index + 1}: ${escapeHtml(entry.word)}</div><div class="grid3c" style="grid-template-columns:repeat(auto-fit,minmax(100px,1fr));">${cells}</div></div>`;
    }).join("");
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-crf">Step 3 — Score</div>
        <p class="sv-desc">Score each token under candidate tags.</p>
        ${blocks}
        <div class="sv-insight i-crf">💡 CRF potentials are computed via exp(score). The tag with the highest potential is selected by the decoder.</div>
      </div>`;
    return;
  }

  if (step === 3) {
    const blocks = path.map((entry, index) => {
      const Z = TAGS.reduce((total, tag) => total + evalCRFFeatures(entry.word, tag, index > 0 ? path[index - 1].tag : "<s>"), 0);
      const posterior = entry.score / Z;
      const pct = Math.min(posterior * 100, 100);
      return `
        <div class="sblock">
          <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;">
            <div style="font-weight:700;">${escapeHtml(entry.word)} → <span class="chip c-crf">${entry.tag}</span></div>
            <div class="sv-mono" style="color:var(--color-text-tertiary,#8a8878);font-size:12px;">ψ=${fmt2(entry.score)} · Z≈${fmt2(Z)} · P≈${fmt2(posterior)}</div>
          </div>
          <div class="meter-track" style="margin-top:8px;"><div class="meter-fill" style="width:${pct}%;background:var(--crf);"></div></div>
        </div>`;
    }).join("");
    container.innerHTML = `
      <div class="sv-card">
        <div class="sv-badge b-crf">Step 4 — Potentials</div>
        <p class="sv-desc">Local potentials normalized into a rough probability.</p>
        ${blocks}
        <div class="sv-insight i-crf">💡 Z(w) is the global normalizer over all tag sequences. This demo approximates normalization per token for readability.</div>
      </div>`;
    return;
  }

  const results = path.map(entry => `
    <div style="min-width:100px;padding:12px;border-radius:14px;background:var(--color-background-primary,#faf9f5);border:0.5px solid var(--color-border-tertiary,#dddbd2);text-align:center;">
      <div style="margin-bottom:8px;font-weight:700;">${escapeHtml(entry.word)}</div>
      <div style="padding:6px 10px;border-radius:999px;background:var(--crf-bg);color:var(--crf);font-weight:700;">${entry.tag}</div>
      <div class="sv-mono" style="margin-top:8px;font-size:11px;color:var(--color-text-tertiary,#8a8878);">ψ=${fmt2(entry.score)}</div>
    </div>`).join("");
  container.innerHTML = `
    <div class="sv-card">
      <div class="sv-badge b-pos">Step 5 — Result</div>
      <p class="sv-desc">Final predicted tag sequence.</p>
      <div style="display:flex;flex-wrap:wrap;gap:10px;">${results}</div>
      <div class="sv-insight i-crf">💡 This is a greedy CRF demo. Production CRFs use Viterbi-style decoding over the full sequence lattice.</div>
    </div>`;
}

// ─── TAB: QUIZ ───────────────────────────────────────────────────────────
function mountQuiz() {
  const panel = document.getElementById("panel-quiz");
  panel.innerHTML = `
    <div class="sv-eyebrow">No Pressure, Just Practice</div>
    <div class="sv-card">
      <div class="sv-card-title">Concept Quiz</div>
      <p class="sv-desc">Pick an answer for instant feedback and a short explanation — no scoring, just learning.</p>
      ${QUIZ_QUESTIONS.map((q, qi) => {
        const isHmm = q.tag === "HMM";
        return `
        ${qi > 0 ? '<hr class="qsep">' : ""}
        <div data-qid="${q.id}">
          <div style="display:flex;gap:8px;margin-bottom:10px;align-items:flex-start;">
            <span class="sv-badge ${isHmm ? "b-hmm" : "b-crf"}" style="margin:2px 0 0;">${escapeHtml(q.tag)}</span>
            <div style="font-size:13.5px;font-weight:600;line-height:1.5;">${escapeHtml(q.question)}</div>
          </div>
          ${q.options.map((opt, oi) => `<button class="qopt" id="qqopt-${q.id}-${oi}" onclick="answerHcQuiz(${q.id},${oi},${q.answerIndex})">${escapeHtml(opt)}</button>`).join("")}
          <div class="sv-insight" id="qqfb-${q.id}" style="display:none;"></div>
        </div>`;
      }).join("")}
      <div style="text-align:center;margin-top:20px;">
        <button class="sv-btn btn-sec" onclick="resetHcQuiz()">↺ Reset quiz</button>
      </div>
    </div>
  `;
}

function answerHcQuiz(qid, oi, ansIdx) {
  const firstBtn = document.getElementById(`qqopt-${qid}-0`);
  if (firstBtn.disabled) return;
  const q = QUIZ_QUESTIONS.find(x => x.id === qid);
  q.options.forEach((opt, i) => {
    const btn = document.getElementById(`qqopt-${qid}-${i}`);
    btn.disabled = true;
    if (i === ansIdx) btn.classList.add("correct");
    else if (i === oi) btn.classList.add("incorrect");
  });
  const fb = document.getElementById(`qqfb-${qid}`);
  const correct = oi === ansIdx;
  fb.style.display = "block";
  fb.className = `sv-insight ${correct ? "i-pos" : "i-warn"}`;
  fb.innerHTML = `<strong>${correct ? "✓ Correct!" : "✗ Incorrect."}</strong> ${escapeHtml(q.explanation)}`;
}
window.answerHcQuiz = answerHcQuiz;

function resetHcQuiz() { mountQuiz(); }
window.resetHcQuiz = resetHcQuiz;
