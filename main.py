#!/usr/bin/env python3
"""Sokoban solver - make.fun "Build the best Sokoban solver" (run 20261121).

Mirror of main.ts (same algorithm, same deterministic tie-breaking):
  * Decompose the level into rooms separated by 1-cell choke doors.
  * Solve each room push-optimally with BFS over push states
    (box set + canonical player region), static dead-square pruning and
    2x2-block deadlock pruning.
  * Stitch rooms with player walks through the hallway.
  * Fallback: run the same push-BFS on the whole level.

I/O: reads XSB level(s) from argv file, else ./level.txt, else stdin.
One solution line per level. Alphabet: l/r/u/d walk, L/R/U/D push.
"""
import os
import sys
from collections import deque

DIRS = [(-1, 0, "u", "U"), (1, 0, "d", "D"), (0, -1, "l", "L"), (0, 1, "r", "R")]


def parse_level(lines):
    H = len(lines)
    W = max(len(ln) for ln in lines)
    wall = bytearray(b"\x01") * (W * H)
    # build explicitly (bytearray * repeats pattern; use list instead)
    wall = [1] * (W * H)
    goal = [0] * (W * H)
    boxes = []
    player = -1
    for r, ln in enumerate(lines):
        for c, ch in enumerate(ln):
            i = r * W + c
            if ch == "#":
                continue
            wall[i] = 0
            if ch in ".*+":
                goal[i] = 1
            if ch in "$*":
                boxes.append(i)
            if ch in "@+":
                player = i
    return {"W": W, "H": H, "wall": wall, "goal": goal,
            "boxes": boxes, "player": player}


def split_levels(text):
    blocks, cur = [], []
    for raw in text.splitlines():
        if "#" in raw:
            cur.append(raw.rstrip())
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def build_neighbours(L):
    W, H, wall = L["W"], L["H"], L["wall"]
    nbr = [None] * (W * H)
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if wall[i]:
                continue
            ns = []
            for dr, dc, _, _ in DIRS:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    ns.append(-1)
                else:
                    j = nr * W + nc
                    ns.append(-1 if wall[j] else j)
            nbr[i] = ns
    return nbr


def restrict_to_reachable(L, nbr):
    size = L["W"] * L["H"]
    seen = bytearray(size)
    stack = [L["player"]]
    seen[L["player"]] = 1
    while stack:
        i = stack.pop()
        for j in nbr[i]:
            if j >= 0 and not seen[j]:
                seen[j] = 1
                stack.append(j)
    for i in range(size):
        if not seen[i]:
            L["wall"][i] = 1
    return build_neighbours(L)


def compute_dead(L, nbr):
    W, H, wall, goal = L["W"], L["H"], L["wall"], L["goal"]
    size = W * H
    live = bytearray(size)
    q = deque()
    for i in range(size):
        if goal[i] and not wall[i]:
            live[i] = 1
            q.append(i)
    while q:
        y = q.popleft()
        for d in range(4):
            x = nbr[y][d ^ 1]
            if x < 0:
                continue
            stand = nbr[x][d ^ 1]
            if stand < 0:
                continue
            if not live[x]:
                live[x] = 1
                q.append(x)
    dead = bytearray(size)
    for i in range(size):
        if not wall[i] and not live[i]:
            dead[i] = 1
    return dead


def flood_mask(start, box_set, nbr, size):
    seen = bytearray(size)
    if start in box_set:
        return seen
    stack = [start]
    seen[start] = 1
    while stack:
        i = stack.pop()
        for j in nbr[i]:
            if j >= 0 and not seen[j] and j not in box_set:
                seen[j] = 1
                stack.append(j)
    return seen


def mask_min(mask):
    for i, v in enumerate(mask):
        if v:
            return i
    return -1


def walk_path(frm, to, box_set, nbr, allowed):
    if frm == to:
        return ""
    size = len(nbr)
    prev = [-2] * size
    prev_dir = [0] * size
    q = deque([frm])
    prev[frm] = -1
    while q:
        i = q.popleft()
        if i == to:
            break
        for d in range(4):
            j = nbr[i][d]
            if j >= 0 and prev[j] == -2 and j not in box_set and allowed[j]:
                prev[j] = i
                prev_dir[j] = d
                q.append(j)
    if prev[to] == -2:
        return None
    chars = []
    cur = to
    while cur != frm:
        chars.append(DIRS[prev_dir[cur]][2])
        cur = prev[cur]
    chars.reverse()
    return "".join(chars)


def solve_area(L, nbr, dead, box_cells, walk_cells, start_player, start_boxes):
    """Push-optimal BFS. Boxes and flood regions are Python-int bitmasks
    over global cell ids; state key = (box mask, player-region rep)."""
    size = L["W"] * L["H"]
    goal = L["goal"]
    W = L["W"]
    H = L["H"]
    wall = L["wall"]

    goal_mask = 0
    for i in range(size):
        if goal[i]:
            goal_mask |= 1 << i
    dead_mask = 0
    for i in range(size):
        if dead[i]:
            dead_mask |= 1 << i
    wall_mask = 0
    for i in range(size):
        if wall[i]:
            wall_mask |= 1 << i
    nbr_mask = [0] * size
    for i in range(size):
        if nbr[i] is not None:
            m = 0
            for j in nbr[i]:
                if j >= 0:
                    m |= 1 << j
            nbr_mask[i] = m

    def flood_rep(start, box_mask):
        reach = 1 << start
        frontier = reach
        while frontier:
            expand = 0
            f = frontier
            while f:
                lsb = f & -f
                expand |= nbr_mask[lsb.bit_length() - 1]
                f ^= lsb
            frontier = expand & ~reach & ~box_mask & ~wall_mask
            reach |= frontier
        return (reach & -reach).bit_length() - 1, reach

    def has_2x2_deadlock_at(box_mask, dest):
        r, c = divmod(dest, W)
        blocked = wall_mask | box_mask
        for dr in (-1, 0):
            for dc in (-1, 0):
                r0, c0 = r + dr, c + dc
                all_blocked = True
                bad_box = False
                for a in range(2):
                    for e in range(2):
                        rr, cc = r0 + a, c0 + e
                        if rr < 0 or rr >= H or cc < 0 or cc >= W:
                            continue
                        bit = 1 << (rr * W + cc)
                        if box_mask & bit:
                            if not goal[rr * W + cc]:
                                bad_box = True
                        elif not (blocked & bit):
                            all_blocked = False
                if all_blocked and bad_box:
                    return True
        return False

    start_mask = 0
    for b in start_boxes:
        start_mask |= 1 << b
    if start_mask & ~goal_mask == 0:
        return {"moves": "", "pushes": 0,
                "boxes": sorted(start_boxes), "player": start_player}
    start_rep, _ = flood_rep(start_player, start_mask)

    # nodes: (box_mask, player, parent, dir, stand)
    nodes = [(start_mask, start_player, -1, -1, -1)]
    index = {(start_mask, start_rep): 0}
    queue = deque([0])
    NODE_CAP = 6_000_000

    def mask_to_sorted(mask):
        out = []
        m = mask
        while m:
            lsb = m & -m
            out.append(lsb.bit_length() - 1)
            m ^= lsb
        return out

    while queue:
        ni = queue.popleft()
        box_mask, player, _par, _d0, _s0 = nodes[ni]
        _rep, reach = flood_rep(player, box_mask)
        m = box_mask
        while m:
            lsb = m & -m
            b = lsb.bit_length() - 1
            m ^= lsb
            for d in range(4):
                stand = nbr[b][d ^ 1]
                if stand < 0 or not (reach >> stand) & 1:
                    continue
                dest = nbr[b][d]
                if dest < 0 or dest not in box_cells:
                    continue
                if (box_mask >> dest) & 1 or (dead_mask >> dest) & 1:
                    continue
                new_mask = box_mask ^ (1 << b) | (1 << dest)
                if has_2x2_deadlock_at(new_mask, dest):
                    continue
                rep, _r = flood_rep(b, new_mask)
                key = (new_mask, rep)
                if key in index:
                    continue
                nodes.append((new_mask, b, ni, d, stand))
                n_idx = len(nodes) - 1
                index[key] = n_idx
                if new_mask & ~goal_mask == 0:
                    chain = []
                    cur = n_idx
                    while nodes[cur][2] >= 0:
                        chain.append(cur)
                        cur = nodes[cur][2]
                    chain.reverse()
                    parts = []
                    for ci in chain:
                        _bm, _pp, par, dd, st = nodes[ci]
                        p_mask, p_player = nodes[par][0], nodes[par][1]
                        walk = walk_path(p_player, st,
                                         set(mask_to_sorted(p_mask)),
                                         nbr, walk_cells)
                        parts.append((walk or "") + DIRS[dd][3])
                    return {"moves": "".join(parts), "pushes": len(chain),
                            "boxes": mask_to_sorted(new_mask), "player": b}
                queue.append(n_idx)
                if len(nodes) > NODE_CAP:
                    return None
    return None


def find_chokes(L, nbr):
    size = L["W"] * L["H"]
    chokes = set()
    for i in range(size):
        if L["wall"][i]:
            continue
        ns = nbr[i]
        present = [d for d in range(4) if ns[d] >= 0]
        if len(present) != 2:
            continue
        d1, d2 = present
        if d2 != (d1 ^ 1):
            continue
        a, b = ns[d1], ns[d2]
        seen = bytearray(size)
        stack = [a]
        seen[a] = 1
        connected = False
        while stack:
            x = stack.pop()
            if x == b:
                connected = True
                break
            for d in range(4):
                j = nbr[x][d]
                if j >= 0 and j != i and not seen[j]:
                    seen[j] = 1
                    stack.append(j)
        if not connected:
            chokes.add(i)
    return chokes


def components_without(L, nbr, removed):
    size = L["W"] * L["H"]
    comp = [-1] * size
    n = 0
    for i in range(size):
        if L["wall"][i] or i in removed or comp[i] >= 0:
            continue
        stack = [i]
        comp[i] = n
        while stack:
            x = stack.pop()
            for d in range(4):
                j = nbr[x][d]
                if j >= 0 and j not in removed and comp[j] < 0:
                    comp[j] = n
                    stack.append(j)
        n += 1
    return comp


def solve_level(lines):
    L = parse_level(lines)
    if L["player"] < 0 or not L["boxes"]:
        return ""
    nbr = build_neighbours(L)
    nbr = restrict_to_reachable(L, nbr)
    dead = compute_dead(L, nbr)
    size = L["W"] * L["H"]
    all_floor = bytearray(size)
    all_cells = set()
    for i in range(size):
        if not L["wall"][i]:
            all_floor[i] = 1
            all_cells.add(i)

    any_dead = any(dead[b] for b in L["boxes"])
    chokes = find_chokes(L, nbr)
    solved = None

    comp = components_without(L, nbr, chokes) if chokes else None
    # a box sitting on a choke cell breaks the room model: bail out
    box_on_choke = comp is not None and any(comp[b] < 0 for b in L["boxes"])
    if not any_dead and chokes and not box_on_choke:
        comp_boxes = {}
        for b in L["boxes"]:
            comp_boxes.setdefault(comp[b], []).append(b)
        player_comp = comp[L["player"]]
        rooms = []
        decomposition_ok = True
        for c, boxes in comp_boxes.items():
            cells = [i for i in range(size) if comp[i] == c]
            doors = set()
            for i in cells:
                for d in range(4):
                    j = nbr[i][d]
                    if j >= 0 and j in chokes:
                        doors.add(j)
            if c != player_comp and len(doors) != 1:
                decomposition_ok = False
                break
            if c == player_comp and len(doors) > 1:
                decomposition_ok = False
                break
            rooms.append({
                "comp": c, "cells": cells, "boxes": boxes,
                "door": next(iter(doors)) if doors else -1,
                "is_player_room": c == player_comp,
            })
        if decomposition_ok and rooms:
            rooms.sort(key=lambda r: (0 if r["is_player_room"] else 1, r["door"]))
            cur_player = L["player"]
            cur_boxes = set(L["boxes"])
            parts = []
            failed = False
            for room in rooms:
                box_cells = set(room["cells"])
                walk_cells = bytearray(size)
                for i in room["cells"]:
                    walk_cells[i] = 1
                if room["door"] >= 0:
                    walk_cells[room["door"]] = 1
                entry = cur_player
                if not room["is_player_room"]:
                    w = walk_path(cur_player, room["door"], cur_boxes, nbr, all_floor)
                    if w is None:
                        failed = True
                        break
                    parts.append(w)
                    entry = room["door"]
                room_boxes = [b for b in room["boxes"] if b in cur_boxes]
                res = solve_area(L, nbr, dead, box_cells, walk_cells,
                                 entry, room_boxes)
                if res is None:
                    failed = True
                    break
                parts.append(res["moves"])
                for b in room_boxes:
                    cur_boxes.discard(b)
                for b in res["boxes"]:
                    cur_boxes.add(b)
                cur_player = res["player"]
            if not failed:
                solved = "".join(parts)

    if solved is None and not any_dead:
        res = solve_area(L, nbr, dead, all_cells, all_floor,
                         L["player"], L["boxes"])
        if res is not None:
            solved = res["moves"]
    return "" if solved is None else solved


def read_input():
    for arg in sys.argv[1:]:
        if os.path.isfile(arg):
            with open(arg, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    if os.path.isfile("level.txt"):
        with open("level.txt", "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return sys.stdin.read()


def main():
    text = read_input()
    blocks = split_levels(text)
    out = [solve_level(b) for b in blocks]
    if out:
        sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
