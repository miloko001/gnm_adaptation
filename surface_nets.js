// Naive surface nets on a scalar field F (Float32Array, dims [ni,nj,nk], index (i*nj+j)*nk+k), iso level L.
// Inside = F >= L. One vertex per boundary cell (mean of edge crossings); one quad per crossing grid edge.
// skip: optional Uint8Array per voxel; quads touching a flagged cell are dropped (cut planes).
// Returns { pos: Float32Array (voxel coords i,j,k), lines: Uint32Array (vertex index pairs, deduplicated) }.
function surfaceNets(F, dims, L, skip, minComp) {
  const [ni, nj, nk] = dims, sJ = nk, sI = nj * nk;
  const cellV = new Int32Array(ni * nj * nk).fill(-1);
  const pos = [];
  const corners = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]];
  const edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
  const val = new Float32Array(8);
  for (let i = 0; i < ni - 1; i++) for (let j = 0; j < nj - 1; j++) for (let k = 0; k < nk - 1; k++) {
    let mask = 0;
    for (let c = 0; c < 8; c++) {
      const o = corners[c], v = F[(i + o[0]) * sI + (j + o[1]) * sJ + k + o[2]];
      val[c] = v; if (v >= L) mask |= 1 << c;
    }
    if (mask === 0 || mask === 255) continue;
    let x = 0, y = 0, z = 0, n = 0;
    for (const [a, b] of edges) {
      if (((mask >> a) & 1) === ((mask >> b) & 1)) continue;
      const t = (L - val[a]) / (val[b] - val[a]), A = corners[a], B = corners[b];
      x += A[0] + t * (B[0] - A[0]); y += A[1] + t * (B[1] - A[1]); z += A[2] + t * (B[2] - A[2]); n++;
    }
    cellV[i * sI + j * sJ + k] = pos.length / 3;
    pos.push(i + x / n, j + y / n, k + z / n);
  }
  // quads: for each grid edge with a sign change, connect the 4 cells around it
  const seen = new Set(), lines = [];
  const nV = pos.length / 3;
  const addLine = (a, b) => { const key = a < b ? a * nV + b : b * nV + a; if (!seen.has(key)) { seen.add(key); lines.push(a, b); } };
  const quad = (c0, c1, c2, c3) => {
    const q = [c0, c1, c2, c3];
    if (skip) for (const c of q) if (skip[c]) return;
    const v = q.map(c => cellV[c]); if (v.some(x => x < 0)) return;
    addLine(v[0], v[1]); addLine(v[1], v[3]); addLine(v[3], v[2]); addLine(v[2], v[0]);
  };
  for (let i = 0; i < ni; i++) for (let j = 0; j < nj; j++) for (let k = 0; k < nk; k++) {
    const p = i * sI + j * sJ + k, inP = F[p] >= L;
    if (i < ni - 1 && j > 0 && k > 0 && (F[p + sI] >= L) !== inP) quad(p - sJ - 1, p - 1, p - sJ, p);               // edge along i
    if (j < nj - 1 && i > 0 && k > 0 && (F[p + sJ] >= L) !== inP) quad(p - sI - 1, p - 1, p - sI, p);               // edge along j
    if (k < nk - 1 && i > 0 && j > 0 && (F[p + 1] >= L) !== inP) quad(p - sI - sJ, p - sJ, p - sI, p);              // edge along k
  }
  // drop small connected pieces (specks)
  if (minComp > 0 && lines.length) {
    const par = new Int32Array(nV); for (let v = 0; v < nV; v++) par[v] = v;
    const find = v => { while (par[v] !== v) { par[v] = par[par[v]]; v = par[v]; } return v; };
    for (let e = 0; e < lines.length; e += 2) { const a = find(lines[e]), b = find(lines[e + 1]); if (a !== b) par[a] = b; }
    const size = new Int32Array(nV); for (let v = 0; v < nV; v++) size[find(v)]++;
    const kept = []; for (let e = 0; e < lines.length; e += 2) if (size[find(lines[e])] >= minComp) kept.push(lines[e], lines[e + 1]);
    return { pos: new Float32Array(pos), lines: new Uint32Array(kept) };
  }
  return { pos: new Float32Array(pos), lines: new Uint32Array(lines) };
}
