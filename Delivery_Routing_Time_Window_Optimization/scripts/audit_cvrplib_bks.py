"""Auditoria independente: parser Solomon + verificador de factibilidade das BKS CVRPLIB."""
import math, pathlib, sys, json

root = pathlib.Path(sys.argv[1])
inst_dir, bks_dir = root / "instances", root / "bks"


def parse(path):
    lines = [l.strip() for l in path.read_text().splitlines()]
    name = lines[0]
    K, Q = map(int, lines[4].split())
    nodes = []
    for l in lines[9:]:
        if not l:
            continue
        v = list(map(int, l.split()))
        assert len(v) == 7, (path, l)
        nodes.append(dict(id=v[0], x=v[1], y=v[2], q=v[3], a=v[4], b=v[5], s=v[6]))
    assert [n["id"] for n in nodes] == list(range(len(nodes))), path
    return name, K, Q, nodes


def parse_sol(path):
    routes, cost = [], None
    for l in path.read_text().splitlines():
        if l.startswith("Route"):
            routes.append(list(map(int, l.split(":")[1].split())))
        elif l.startswith("Cost"):
            cost = float(l.split()[1])
    return routes, cost


def d_exact(n, m):
    return math.hypot(n["x"] - m["x"], n["y"] - m["y"])


def d_trunc(n, m):
    return math.floor(d_exact(n, m) * 10) / 10


def check(nodes, Q, routes, dist):
    N = len(nodes) - 1
    seen = [c for r in routes for c in r]
    viol = []
    if sorted(seen) != list(range(1, N + 1)):
        viol.append("visitas")
    total = 0.0
    for r in routes:
        load = sum(nodes[c]["q"] for c in r)
        if load > Q:
            viol.append(f"cap {load}>{Q}")
        t, prev = 0.0, nodes[0]
        for c in r:
            n = nodes[c]
            t += dist(prev, n)
            total += dist(prev, n)
            if t > n["b"] + 1e-9:
                viol.append(f"tw {c}")
            t = max(t, n["a"]) + n["s"]
            prev = n
        total += dist(prev, nodes[0])
        if t + dist(prev, nodes[0]) > nodes[0]["b"] + 1e-9:
            viol.append("deposito")
    return total, viol


rows = []
for p in sorted(inst_dir.glob("*.txt")):
    name, K, Q, nodes = parse(p)
    routes, ref = parse_sol(bks_dir / f"{p.stem}.sol")
    tot_t, v_t = check(nodes, Q, routes, d_trunc)
    tot_e, v_e = check(nodes, Q, routes, d_exact)
    ints = all(isinstance(n["x"], int) for n in nodes)
    rows.append(dict(inst=name, n=len(nodes) - 1, K=K, Q=Q, depot_b=nodes[0]["b"],
                     s=sorted({n["s"] for n in nodes[1:]}), routes=len(routes), ref=ref,
                     trunc=round(tot_t, 1), exact=round(tot_e, 2),
                     feas_trunc=not v_t, feas_exact=not v_e, viol_exact=v_e[:3]))
for r in rows:
    print(r)
print("todas factiveis (trunc):", all(r["feas_trunc"] for r in rows))
print("custo bate (trunc):", all(abs(r["trunc"] - r["ref"]) < 0.051 for r in rows))
print("infactiveis em distancia exata:", [r["inst"] for r in rows if not r["feas_exact"]])
json.dump(rows, open(sys.argv[2], "w"), indent=1)
