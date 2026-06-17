// Epicure: browser-side ingredient embedding engine
// Ported from epicure.py — runs entirely in the browser with no backend.
// Paper: https://arxiv.org/abs/2605.22391

const EPICURE = (() => {
  let E = null;        // Float32Array[n_vocab * d_model], unit-normalized
  let vocab = null;    // {name: index}
  let itos = null;     // [index] => name
  let modes = null;     // [{id, kind, label, members[], pole[]}]
  let supPoles = null; // {key: [d_model]}
  let dModel = 300;
  let nVocab = 1790;
  let ready = false;

  // ── Load data from epicure_data/ ───────────────────────────────
  async function load(baseUrl) {
    baseUrl = baseUrl || 'epicure_data/';
    const [embBin, itosJson, modesJson, supJson] = await Promise.all([
      fetch(baseUrl + 'embeddings.bin', { cache: 'no-cache' }).then(r => r.arrayBuffer()),
      fetch(baseUrl + 'itos.json', { cache: 'no-cache' }).then(r => r.json()),
      fetch(baseUrl + 'modes.json', { cache: 'no-cache' }).then(r => r.json()),
      fetch(baseUrl + 'supervised_poles.json', { cache: 'no-cache' }).then(r => r.json()),
    ]);
    E = new Float32Array(embBin);
    nVocab = E.length / dModel;
    itos = itosJson;
    // Build vocab lookup
    vocab = {};
    for (let i = 0; i < itos.length; i++) vocab[itos[i]] = i;

    // Parse modes
    modes = modesJson.map(m => ({
      id: m.id,
      kind: m.kind,
      label: m.label,
      members: m.members,
      pole: normalize(new Float32Array(m.pole)),
    }));
    supPoles = {};
    for (const [k, v] of Object.entries(supJson)) {
      supPoles[k] = normalize(new Float32Array(v));
    }
    ready = true;
    return true;
  }

  // ── Math helpers ───────────────────────────────
  function normalize(v) {
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n);
    if (n < 1e-9) return v;
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  }

  function dot(a, b) {
    let s = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) s += a[i] * b[i];
    return s;
  }

  function vec(name) {
    const idx = vocab[name];
    const off = idx * dModel;
    return E.slice(off, off + dModel);
  }

  // ── Core operators ───────────────────────────────
  function neighbors(name, k, excludeSelf) {
    if (k === undefined) k = 5;
    if (excludeSelf === undefined) excludeSelf = true;
    if (!ready) throw new Error('Epicure not loaded');
    const idx = vocab[name];
    const v = vec(name);
    const sims = new Float32Array(nVocab);
    for (let i = 0; i < nVocab; i++) {
      const vi = E.slice(i * dModel, (i + 1) * dModel);
      sims[i] = dot(v, vi);
    }
    // Sort indices by similarity descending
    const order = Array.from({length: sims.length}, (_, i) => i)
      .sort((a, b) => sims[b] - sims[a]);
    const start = excludeSelf ? 1 : 0;
    const result = [];
    for (let j = start; j < start + k && j < order.length; j++) {
      result.push([itos[order[j]], sims[order[j]]]);
    }
    return result;
  }

  function slerp(seed, direction, thetaDeg, k, excludeSeed) {
    if (k === undefined) k = 5;
    if (excludeSeed === undefined) excludeSeed = true;
    if (!ready) throw new Error('Epicure not loaded');
    const seedIdx = vocab[seed];
    const v = vec(seed);
    let d;
    if (typeof direction === 'string') {
      d = supPoles[direction] || new Float32Array(dModel);
    } else {
      d = normalize(new Float32Array(direction));
    }

    // Gram-Schmidt: orthogonal component
    const dotvd = dot(v, d);
    const dPerp = new Float32Array(dModel);
    for (let i = 0; i < dModel; i++) dPerp[i] = d[i] - dotvd * v[i];
    let nPerp = 0;
    for (let i = 0; i < dModel; i++) nPerp += dPerp[i] * dPerp[i];
    nPerp = Math.sqrt(nPerp);
    if (nPerp < 1e-9) return neighbors(seed, k);

    for (let i = 0; i < dModel; i++) dPerp[i] /= nPerp;

    const theta = thetaDeg * Math.PI / 180;
    const q = new Float32Array(dModel);
    for (let i = 0; i < dModel; i++) {
      q[i] = Math.cos(theta) * v[i] + Math.sin(theta) * dPerp[i];
    }
    const qn = normalize(q);

    const sims = new Float32Array(nVocab);
    for (let i = 0; i < nVocab; i++) {
      const vi = E.slice(i * dModel, (i + 1) * dModel);
      sims[i] = dot(qn, vi);
    }
    if (excludeSeed) sims[seedIdx] = -Infinity;

    const order = Array.from({length: sims.length}, (_, i) => i)
      .sort((a, b) => sims[b] - sims[a]);
    const result = [];
    for (let j = 0; j < k && j < order.length; j++) {
      result.push([itos[order[j]], sims[order[j]]]);
    }
    return result;
  }

  function closestMode(name, kind, k) {
    if (k === undefined) k = 3;
    if (!ready) throw new Error('Epicure not loaded');
    const v = vec(name);
    const scored = [];
    for (const m of modes) {
      if (kind && m.kind !== kind) continue;
      scored.push({id: m.id, label: m.label, score: dot(m.pole, v)});
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k).map(s => [s.id, s.label, s.score]);
  }

  function modeMembers(modeId, k) {
    for (const m of modes) {
      if (m.id === modeId) return k ? m.members.slice(0, k) : m.members;
    }
    throw new Error(`Mode not found: ${modeId}`);
  }

  // ── Recipe suggestion ───────────────────────────────
  // Given a list of ingredient names, find the best pairings
  function suggestRecipes(ingredients, count) {
    if (count === undefined) count = 5;
    if (!ready) throw new Error('Epicure not loaded');
    const results = new Map(); // ingredient -> accumulated score
    for (const ing of ingredients) {
      if (!vocab[ing]) continue;
      const n = neighbors(ing, 20, true);
      for (const [name, score] of n) {
        results.set(name, (results.get(name) || 0) + score);
      }
    }
    // Sort by accumulated score, exclude input ingredients
    const inputSet = new Set(ingredients);
    const sorted = [...results.entries()]
      .filter(([n]) => !inputSet.has(n))
      .sort((a, b) => b[11] - a[1]);
    return sorted.slice(0, count).map(([n, s]) => [n, s]);
  }

  function isReady() { return ready; }
  function getIngredientList() { return itos; }
  function getCuisineDirections() {
    return Object.keys(supPoles).filter(k => k.startsWith('cuisine:'));
  }

  return { load, neighbors, slerp, closestMode, modeMembers, suggestRecipes, isReady, getIngredientList, getCuisineDirections, vec, normalize, dot };
})();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) module.exports = EPICURE;
