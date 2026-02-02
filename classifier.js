import mongoose from 'mongoose'

const stop = new Set(['a','an','the','and','or','but','if','then','else','on','in','at','to','from','by','for','of','with','without','about','as','into','like','through','over','after','before','between','under','above','not','no','nor','be','is','are','was','were','am','been','being','do','does','did','doing','have','has','had','having','can','could','should','would','may','might','must','will','shall','you','your','yours','me','my','mine','we','our','ours','they','their','theirs','he','she','it','this','that','these','those'])

function tokens(text) {
  const lower = String(text || '').toLowerCase()
  const arr = lower.match(/[a-z]+/g) || []
  const out = []
  for (const t of arr) {
    if (stop.has(t)) continue
    let s = t
    if (s.endsWith('ing') && s.length > 5) s = s.slice(0, -3)
    else if (s.endsWith('ed') && s.length > 4) s = s.slice(0, -2)
    else if (s.endsWith('ly') && s.length > 4) s = s.slice(0, -2)
    else if (s.endsWith('ies') && s.length > 5) s = s.slice(0, -3) + 'y'
    else if (s.endsWith('es') && s.length > 4) s = s.slice(0, -2)
    else if (s.endsWith('s') && s.length > 3) s = s.slice(0, -1)
    out.push(s)
  }
  return out
}

function tfidfWeights(docTokens, idf) {
  const tf = new Map()
  for (const w of docTokens) tf.set(w, (tf.get(w) || 0) + 1)
  const weights = new Map()
  const docLen = docTokens.length || 1
  for (const [w, c] of tf.entries()) {
    const tfv = c / docLen
    const idfv = idf.get(w) || 0
    weights.set(w, tfv * idfv)
  }
  return weights
}

const model = {
  ready: false,
  vocab: new Set(),
  priors: { fir: 0.5, nonfir: 0.5 },
  cond: { fir: new Map(), nonfir: new Map() },
  totals: { fir: 0, nonfir: 0 },
  idf: new Map(),
}

const seedFir = [
  'stolen','robbery','theft','assault','violence','attack',
  'murder','rape','kidnap','burglary','snatch','threat',
  'extortion','arson','hit','kill','weapon','injury',
  'molest','harass','beat','fight','crime','abuse','illegal'
];

const seedNon = [
  'noc','verification','certificate','passport','address',
  'proof','character','police','clearance','document',
  'lost','missing','misplace','found','helpdesk','service',
  'issue','request','application','ID','card','wallet',
  'booking','info','support','feedback'
];
const strongFir = new Set([
  'stolen','robbery','assault','murder','rape','kidnap',
  'burglary','snatch','extortion','weapon','violence','attack'
]);
const nonFirClues = new Set([
  'lost','missing','misplace','noc','verification','certificate',
  'passport','address','proof','clearance','document','found',
  'service','issue','request','application','support','feedback'
]);


function addCounts(cls, toks) {
  for (const w of toks) {
    model.vocab.add(w)
    model.cond[cls].set(w, (model.cond[cls].get(w) || 0) + 1)
    model.totals[cls] += 1
  }
}

async function trainFromDb() {
  try {
    const Complaint = mongoose.model('Complaint')
    const q = { category: { $in: ['fir','non-fir'] }, description: { $exists: true, $ne: '' } }
    const rows = await Complaint.find(q).select('description category').limit(2000).lean()
    let firDocs = 0, nonDocs = 0
    for (const r of rows) {
      const toks = tokens(r.description)
      if ((r.category || '').toLowerCase() === 'fir') { firDocs += 1; addCounts('fir', toks) }
      else { nonDocs += 1; addCounts('nonfir', toks) }
    }
    const docs = firDocs + nonDocs
    model.priors.fir = docs ? Math.max(1e-6, firDocs / docs) : 0.5
    model.priors.nonfir = docs ? Math.max(1e-6, nonDocs / docs) : 0.5
    const df = new Map()
    for (const w of model.vocab) {
      const inFir = model.cond.fir.has(w) ? 1 : 0
      const inNon = model.cond.nonfir.has(w) ? 1 : 0
      df.set(w, inFir + inNon)
    }
    const N = Math.max(docs, 1)
    for (const [w, d] of df.entries()) model.idf.set(w, Math.log((N + 1) / (d + 1)))
  } catch {}
}

function trainFromSeed() {
  addCounts('fir', seedFir)
  addCounts('nonfir', seedNon)
  model.priors.fir = 0.5
  model.priors.nonfir = 0.5
  const all = new Set([...seedFir, ...seedNon])
  const N = 2
  for (const w of all) model.idf.set(w, Math.log((N + 1) / 2))
}

export async function initClassifier() {
  model.ready = false
  model.vocab.clear()
  model.cond.fir.clear()
  model.cond.nonfir.clear()
  model.totals.fir = 0
  model.totals.nonfir = 0
  model.idf.clear()
  await trainFromDb()
  if (model.vocab.size === 0) trainFromSeed()
  model.ready = true
}

export function preprocess(text) {
  return tokens(text)
}

export function classifyText(text) {
  const toks = tokens(text)
  if (!model.ready || toks.length === 0) {
    const joined = toks.join(' ')
    const firHits = seedFir.filter(k => joined.includes(k)).length
    const nonHits = seedNon.filter(k => joined.includes(k)).length
    const label = firHits >= nonHits ? 'fir' : 'non-fir'
    return { label, probFir: firHits >= nonHits ? 0.6 : 0.4, probNonFir: firHits >= nonHits ? 0.4 : 0.6, tokens: toks }
  }
  const hasStrongFir = toks.some(w => strongFir.has(w))
  const hasNonFirCue = toks.some(w => nonFirClues.has(w))
  if (!hasStrongFir && hasNonFirCue) {
    return { label: 'non-fir', probFir: 0.35, probNonFir: 0.65, tokens: toks }
  }
  const weights = tfidfWeights(toks, model.idf)
  const V = Math.max(model.vocab.size, 1)
  const alpha = 1.0
  const totalFir = model.totals.fir + alpha * V
  const totalNon = model.totals.nonfir + alpha * V
  let sFir = Math.log(model.priors.fir || 1e-6)
  let sNon = Math.log(model.priors.nonfir || 1e-6)
  for (const [w, wgt] of weights.entries()) {
    const cFir = (model.cond.fir.get(w) || 0) + alpha
    const cNon = (model.cond.nonfir.get(w) || 0) + alpha
    const pFir = cFir / totalFir
    const pNon = cNon / totalNon
    sFir += wgt * Math.log(pFir)
    sNon += wgt * Math.log(pNon)
  }
  const m = Math.max(sFir, sNon)
  const eFir = Math.exp(sFir - m)
  const eNon = Math.exp(sNon - m)
  const Z = eFir + eNon
  const pFir = eFir / Z
  const pNon = eNon / Z
  let label = pFir >= pNon ? 'fir' : 'non-fir'
  if (!hasStrongFir && pFir < 0.6) label = 'non-fir'
  return { label, probFir: pFir, probNonFir: pNon, tokens: toks }
}

export function trainFromExamples(examples) {
  model.ready = false
  model.vocab.clear()
  model.cond.fir.clear()
  model.cond.nonfir.clear()
  model.totals.fir = 0
  model.totals.nonfir = 0
  model.idf.clear()
  let firDocs = 0, nonDocs = 0
  for (const ex of examples) {
    const toks = tokens(ex.text)
    if ((ex.label || '').toLowerCase() === 'fir') { firDocs += 1; addCounts('fir', toks) } else { nonDocs += 1; addCounts('nonfir', toks) }
  }
  const docs = firDocs + nonDocs
  model.priors.fir = docs ? Math.max(1e-6, firDocs / docs) : 0.5
  model.priors.nonfir = docs ? Math.max(1e-6, nonDocs / docs) : 0.5
  const df = new Map()
  for (const w of model.vocab) {
    const inFir = model.cond.fir.has(w) ? 1 : 0
    const inNon = model.cond.nonfir.has(w) ? 1 : 0
    df.set(w, inFir + inNon)
  }
  const N = Math.max(docs, 1)
  for (const [w, d] of df.entries()) model.idf.set(w, Math.log((N + 1) / (d + 1)))
  model.ready = true
}

export function getModel() { return model }

