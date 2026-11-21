// Sokoban solver — make.fun "Build the best Sokoban solver" (run 20261121)
//
// Strategy (matches the structure published in the brief):
//   * A level = 1-wide hallway + 1..4 rooms (6x6 slots), one door per room.
//   * A box can never leave its room, so the push-optimum of a level is the
//     sum of the push-optima of its rooms.
//   * Each room is solved push-optimally with BFS over push states
//     (box set + canonical player region), with static dead-square pruning
//     and 2x2-block deadlock pruning.
//   * Rooms are stitched together with plain player walks through the
//     hallway (walks do not affect the push count).
//   * If decomposition does not apply (classic single-area levels), the
//     same push-BFS runs on the whole level as a fallback.
//
// I/O: reads XSB level(s) from argv file, else ./level.txt, else stdin.
// One or more levels (blocks of lines containing '#', blank-line separated)
// are solved; one solution line is printed per level.
// Solution alphabet: l/r/u/d = walk, L/R/U/D = push (XSB convention).
// Deterministic: fixed direction order, sorted box tuples, sorted rooms.

"use strict";

const fs = require("fs");

// ---------------------------------------------------------------- parsing
// Returns { W, H, wall[], goal[], boxes[], player } for one level block.
function parseLevel(lines) {
  const H = lines.length;
  let W = 0;
  for (const ln of lines) W = Math.max(W, ln.length);
  const wall = new Uint8Array(W * H).fill(1);
  const goal = new Uint8Array(W * H);
  let boxes = [];
  let player = -1;
  for (let r = 0; r < H; r++) {
    const ln = lines[r];
    for (let c = 0; c < ln.length; c++) {
      const ch = ln[c];
      const i = r * W + c;
      if (ch === "#") continue;
      wall[i] = 0;
      if (ch === "." || ch === "*" || ch === "+") goal[i] = 1;
      if (ch === "$" || ch === "*") boxes.push(i);
      if (ch === "@" || ch === "+") player = i;
    }
  }
  return { W, H, wall, goal, boxes, player };
}

// Split raw input into level blocks: maximal runs of lines containing '#'.
function splitLevels(text) {
  const blocks = [];
  let cur = [];
  for (const raw of text.split(/\r?\n/)) {
    if (raw.includes("#")) {
      cur.push(raw.replace(/\s+$/, ""));
    } else if (cur.length) {
      blocks.push(cur);
      cur = [];
    }
  }
  if (cur.length) blocks.push(cur);
  return blocks;
}

// ------------------------------------------------------------ grid helpers
const DIRS = [
  { dr: -1, dc: 0, walk: "u", push: "U" },
  { dr: 1, dc: 0, walk: "d", push: "D" },
  { dr: 0, dc: -1, walk: "l", push: "L" },
  { dr: 0, dc: 1, walk: "r", push: "R" },
];

function buildNeighbours(L) {
  const { W, H, wall } = L;
  const nbr = new Array(W * H);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const i = r * W + c;
      if (wall[i]) { nbr[i] = null; continue; }
      const ns = [];
      for (const d of DIRS) {
        const nr = r + d.dr, nc = c + d.dc;
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) { ns.push(-1); continue; }
        const j = nr * W + nc;
        ns.push(wall[j] ? -1 : j);
      }
      nbr[i] = ns;
    }
  }
  return nbr;
}

// Reachable floor = flood from player over non-wall cells. Cells outside it
// are sealed off and treated as walls from here on.
function restrictToReachable(L, nbr) {
  const seen = new Uint8Array(L.W * L.H);
  const stack = [L.player];
  seen[L.player] = 1;
  while (stack.length) {
    const i = stack.pop();
    for (const j of nbr[i]) {
      if (j >= 0 && !seen[j]) { seen[j] = 1; stack.push(j); }
    }
  }
  for (let i = 0; i < seen.length; i++) {
    if (!seen[i]) L.wall[i] = 1;
  }
  return buildNeighbours(L);
}

// Static dead squares: cells from which one box (alone on an empty board)
// can never be pushed onto any goal. Reverse (pull) BFS from the goals.
// A box resting on a dead square can never be part of a solution.
function computeDead(L, nbr) {
  const { W, H, wall, goal } = L;
  const live = new Uint8Array(W * H);
  const q = [];
  for (let i = 0; i < W * H; i++) if (goal[i] && !wall[i]) { live[i] = 1; q.push(i); }
  let head = 0;
  while (head < q.length) {
    const y = q[head++];
    for (let d = 0; d < 4; d++) {
      // box was at x = y - d, pushed in direction d onto y;
      // the player stood at x - d. Both cells must be floor.
      const back = nbr[y][d ^ 1]; // neighbour of y opposite to d
      if (back < 0) continue;
      // nbr index pairs: 0=U,1=D,2=L,3=R; step from x in dir d reaches y,
      // so x = neighbour of y in the opposite direction.
      const x = back;
      const stand = nbr[x][d ^ 1];
      if (stand < 0) continue;
      if (!live[x]) { live[x] = 1; q.push(x); }
    }
  }
  const dead = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (!wall[i] && !live[i]) dead[i] = 1;
  return dead;
}

// ------------------------------------------------------------ player walk
// Shortest walk (lowercase moves) from `from` to `to` avoiding boxes,
// staying inside cells where allowed[i] is true. Returns null if impossible.
function walkPath(from, to, boxSet, nbr, allowed) {
  if (from === to) return "";
  const prev = new Int32Array(nbr.length).fill(-2);
  const prevDir = new Int8Array(nbr.length);
  const q = [from];
  prev[from] = -1;
  let head = 0;
  while (head < q.length) {
    const i = q[head++];
    if (i === to) break;
    const ns = nbr[i];
    for (let d = 0; d < 4; d++) {
      const j = ns[d];
      if (j >= 0 && prev[j] === -2 && !boxSet.has(j) && allowed[j]) {
        prev[j] = i;
        prevDir[j] = d;
        q.push(j);
      }
    }
  }
  if (prev[to] === -2) return null;
  const chars = [];
  let cur = to;
  while (cur !== from) {
    chars.push(DIRS[prevDir[cur]].walk);
    cur = prev[cur];
  }
  chars.reverse();
  return chars.join("");
}

// ------------------------------------------------------------ push solver
// Push-optimal BFS over one area.
//   boxCells : Set of cells a box may occupy
//   walkCells: Uint8Array mask of cells the player may occupy
//   startPlayer, startBoxes (array), goals (Uint8Array), dead (Uint8Array)
// Returns { moves, pushes, boxes, player } or null.
//
// Performance notes: states are keyed by a single number when the box
// count allows it (base-`size` encoding of the sorted boxes + player
// region representative); player floods use stamp arrays instead of
// fresh allocations; the 2x2 deadlock test only inspects the squares
// around the pushed box's destination, the only place a new 2x2 block
// can appear.
function solveArea(L, nbr, dead, boxCells, walkCells, startPlayer, startBoxes) {
  const size = L.W * L.H;
  const W = L.W, H = L.H;
  const goal = L.goal;
  const wall = L.wall;
  const nBoxes = startBoxes.length;
  const useNumeric = nBoxes <= 6 &&
    Math.pow(size, nBoxes + 1) < Number.MAX_SAFE_INTEGER;

  // Stamp-based flood fills. Two independent stamp arrays are needed:
  // one holds the current node's player-reachability while children are
  // generated, the other computes each child's region representative.
  const stack = new Int32Array(size);
  function makeFlood() {
    const stamp = new Int32Array(size);
    let cur = 0;
    return {
      stamp,
      get cur() { return cur; },
      flood(start, boxMask) {
        cur++;
        let sp = 0;
        let min = start;
        stamp[start] = cur;
        stack[sp++] = start;
        while (sp > 0) {
          const i = stack[--sp];
          if (i < min) min = i;
          const ns = nbr[i];
          for (let d = 0; d < 4; d++) {
            const j = ns[d];
            if (j >= 0 && stamp[j] !== cur && !boxMask[j]) {
              stamp[j] = cur;
              stack[sp++] = j;
            }
          }
        }
        return min;
      },
    };
  }
  const reachFlood = makeFlood();
  const repFlood = makeFlood();

  const boxMask = new Uint8Array(size);
  const startBoxesSorted = [...startBoxes].sort((a, b) => a - b);
  for (const b of startBoxesSorted) boxMask[b] = 1;
  const startRep = repFlood.flood(startPlayer, boxMask);

  function keyOf(boxesSorted, rep) {
    if (useNumeric) {
      let k = rep;
      for (let i = nBoxes - 1; i >= 0; i--) k = k * size + boxesSorted[i];
      return k;
    }
    return boxesSorted.join(",") + "|" + rep;
  }

  const nodes = [];
  const index = new Map();
  function addNode(boxesSorted, player, parent, dir, stand) {
    nodes.push({ boxes: boxesSorted, player, parent, dir, stand });
    return nodes.length - 1;
  }
  index.set(keyOf(startBoxesSorted, startRep),
            addNode(startBoxesSorted, startPlayer, -1, -1, -1));
  const queue = [0];
  let qhead = 0;
  const NODE_CAP = 6000000;

  const isGoal = (boxes) => boxes.every((b) => goal[b]);
  if (isGoal(startBoxesSorted)) {
    return { moves: "", pushes: 0, boxes: startBoxesSorted, player: startPlayer };
  }

  // 2x2 deadlock local to the destination cell of a push (boxMask must
  // already reflect the position AFTER the push).
  function has2x2At(dest) {
    const r = Math.floor(dest / W), c = dest % W;
    for (let dr = -1; dr <= 0; dr++) {
      for (let dc = -1; dc <= 0; dc++) {
        const r0 = r + dr, c0 = c + dc;
        let allBlocked = true;
        let badBox = false;
        for (let a = 0; a < 2 && allBlocked; a++) {
          for (let e = 0; e < 2; e++) {
            const rr = r0 + a, cc = c0 + e;
            if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue; // OOB = wall
            const cell = rr * W + cc;
            if (boxMask[cell]) { if (!goal[cell]) badBox = true; }
            else if (!wall[cell]) allBlocked = false;
          }
        }
        if (allBlocked && badBox) return true;
      }
    }
    return false;
  }

  while (qhead < queue.length) {
    const ni = queue[qhead++];
    const node = nodes[ni];
    boxMask.fill(0);
    for (const b of node.boxes) boxMask[b] = 1;
    reachFlood.flood(node.player, boxMask);
    const reachStamp = reachFlood.cur;
    for (const b of node.boxes) {
      for (let d = 0; d < 4; d++) {
        const stand = nbr[b][d ^ 1];
        if (stand < 0 || reachFlood.stamp[stand] !== reachStamp) continue;
        const dest = nbr[b][d];
        if (dest < 0 || !boxCells.has(dest) || boxMask[dest]) continue;
        if (dead[dest]) continue;
        boxMask[b] = 0;
        boxMask[dest] = 1;
        if (has2x2At(dest)) { boxMask[b] = 1; boxMask[dest] = 0; continue; }
        const rep = repFlood.flood(b, boxMask);
        const newBoxesSorted = node.boxes.slice();
        newBoxesSorted[newBoxesSorted.indexOf(b)] = dest;
        newBoxesSorted.sort((x, y) => x - y);
        boxMask[b] = 1;
        boxMask[dest] = 0;
        const key = keyOf(newBoxesSorted, rep);
        if (index.has(key)) continue;
        const nIdx = addNode(newBoxesSorted, b, ni, d, stand);
        index.set(key, nIdx);
        if (isGoal(newBoxesSorted)) {
          // reconstruct: walk strings recomputed along the parent chain
          const chain = [];
          let cur = nIdx;
          while (nodes[cur].parent >= 0) {
            chain.push(cur);
            cur = nodes[cur].parent;
          }
          chain.reverse();
          const parts = [];
          for (const ci of chain) {
            const nd = nodes[ci];
            const parent = nodes[nd.parent];
            const walk = walkPath(parent.player, nd.stand,
                                  new Set(parent.boxes), nbr, walkCells);
            parts.push((walk === null ? "" : walk) + DIRS[nd.dir].push);
          }
          return {
            moves: parts.join(""),
            pushes: chain.length,
            boxes: newBoxesSorted,
            player: b,
          };
        }
        queue.push(nIdx);
        if (nodes.length > NODE_CAP) return null;
      }
    }
  }
  return null;
}

// ---------------------------------------------------------- decomposition
// Choke cells: floor cells with exactly 2 floor neighbours, opposite each
// other, whose removal disconnects those neighbours (true articulations).
function findChokes(L, nbr) {
  const chokes = new Set();
  const size = L.W * L.H;
  for (let i = 0; i < size; i++) {
    if (L.wall[i]) continue;
    const ns = nbr[i];
    const present = [];
    for (let d = 0; d < 4; d++) if (ns[d] >= 0) present.push(d);
    if (present.length !== 2) continue;
    const [d1, d2] = present;
    if (d2 !== (d1 ^ 1)) continue; // must be opposite
    const a = ns[d1], b = ns[d2];
    // is `a` still connected to `b` without cell i?
    const seen = new Uint8Array(size);
    const stack = [a];
    seen[a] = 1;
    let ok = false;
    while (stack.length) {
      const x = stack.pop();
      if (x === b) { ok = true; break; }
      for (let d = 0; d < 4; d++) {
        const j = nbr[x][d];
        if (j >= 0 && j !== i && !seen[j]) { seen[j] = 1; stack.push(j); }
      }
    }
    if (!ok) chokes.add(i);
  }
  return chokes;
}

// Connected components of floor after removing choke cells.
function componentsWithout(L, nbr, removed) {
  const size = L.W * L.H;
  const comp = new Int32Array(size).fill(-1);
  let n = 0;
  for (let i = 0; i < size; i++) {
    if (L.wall[i] || removed.has(i) || comp[i] >= 0) continue;
    const stack = [i];
    comp[i] = n;
    while (stack.length) {
      const x = stack.pop();
      for (let d = 0; d < 4; d++) {
        const j = nbr[x][d];
        if (j >= 0 && !removed.has(j) && comp[j] < 0) { comp[j] = n; stack.push(j); }
      }
    }
    n++;
  }
  return comp;
}

// -------------------------------------------------------------- main solve
function solveLevel(lines) {
  const L = parseLevel(lines);
  if (L.player < 0 || L.boxes.length === 0) return "";
  let nbr = buildNeighbours(L);
  nbr = restrictToReachable(L, nbr);
  const dead = computeDead(L, nbr);
  const size = L.W * L.H;
  const allFloor = new Uint8Array(size);
  for (let i = 0; i < size; i++) if (!L.wall[i]) allFloor[i] = 1;
  const allCells = new Set();
  for (let i = 0; i < size; i++) if (!L.wall[i]) allCells.add(i);

  // Any box starting on a dead square makes the level unsolvable.
  const anyDead = L.boxes.some((b) => dead[b]);

  const chokes = findChokes(L, nbr);
  let solved = null;

  if (!anyDead && chokes.size > 0) {
    const comp = componentsWithout(L, nbr, chokes);
    // a box sitting on a choke cell breaks the room model: bail out
    if (L.boxes.some((b) => comp[b] < 0)) {
      // (fall through to the whole-level fallback below)
    } else {
    // group boxes & goals by component
    const compBoxes = new Map();
    for (const b of L.boxes) {
      const c = comp[b];
      if (!compBoxes.has(c)) compBoxes.set(c, []);
      compBoxes.get(c).push(b);
    }
    const playerComp = comp[L.player];
    // A component is a room if it holds boxes. Rooms must have exactly one
    // adjacent choke (their door), except possibly the player's component.
    const rooms = [];
    let decompositionOk = true;
    for (const [c, boxes] of compBoxes) {
      const cells = [];
      for (let i = 0; i < size; i++) if (comp[i] === c) cells.push(i);
      const doors = new Set();
      for (const i of cells) {
        for (let d = 0; d < 4; d++) {
          const j = nbr[i][d];
          if (j >= 0 && chokes.has(j)) doors.add(j);
        }
      }
      if (c !== playerComp && doors.size !== 1) { decompositionOk = false; break; }
      if (c === playerComp && doors.size > 1) { decompositionOk = false; break; }
      rooms.push({
        comp: c,
        cells,
        boxes,
        door: doors.size ? [...doors][0] : -1,
        isPlayerRoom: c === playerComp,
      });
    }
    if (decompositionOk && rooms.length > 0) {
      // deterministic order: player room first, then by door position
      rooms.sort((a, b) => {
        if (a.isPlayerRoom !== b.isPlayerRoom) return a.isPlayerRoom ? -1 : 1;
        return a.door - b.door;
      });
      let curPlayer = L.player;
      let curBoxes = new Set(L.boxes);
      const parts = [];
      let totalPushes = 0;
      let failed = false;
      for (const room of rooms) {
        const boxCells = new Set(room.cells);
        const walkCells = new Uint8Array(size);
        for (const i of room.cells) walkCells[i] = 1;
        if (room.door >= 0) walkCells[room.door] = 1;
        let entry = curPlayer;
        if (!room.isPlayerRoom) {
          const w = walkPath(curPlayer, room.door, curBoxes, nbr, allFloor);
          if (w === null) { failed = true; break; }
          parts.push(w);
          entry = room.door;
        }
        const roomBoxes = room.boxes.filter((b) => curBoxes.has(b));
        const res = solveArea(L, nbr, dead, boxCells, walkCells, entry, roomBoxes);
        if (!res) { failed = true; break; }
        parts.push(res.moves);
        totalPushes += res.pushes;
        for (const b of roomBoxes) curBoxes.delete(b);
        for (const b of res.boxes) curBoxes.add(b);
        curPlayer = res.player;
      }
      if (!failed) solved = parts.join("");
    }
    }
  }

  if (solved === null && !anyDead) {
    // Fallback: solve the whole level as one area.
    const res = solveArea(L, nbr, dead, allCells, allFloor, L.player, L.boxes);
    if (res) solved = res.moves;
  }
  return solved === null ? "" : solved;
}

// -------------------------------------------------------------------- I/O
function readInput() {
  for (const arg of process.argv.slice(2)) {
    try {
      if (fs.existsSync(arg) && fs.statSync(arg).isFile()) {
        return fs.readFileSync(arg, "utf8");
      }
    } catch (e) { /* fall through */ }
  }
  try {
    if (fs.existsSync("level.txt")) return fs.readFileSync("level.txt", "utf8");
  } catch (e) { /* fall through */ }
  try {
    return fs.readFileSync(0, "utf8");
  } catch (e) {
    return "";
  }
}

function main() {
  const text = readInput();
  const blocks = splitLevels(text);
  const out = blocks.map((b) => solveLevel(b));
  process.stdout.write(out.join("\n") + (out.length ? "\n" : ""));
}

main();
