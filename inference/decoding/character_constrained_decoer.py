import math, random
random.seed(0)

VOCAB = list('{"}:,abcdefghijklmnopqrstuvwxyz')  # allowed chars

def logits(prefix: str):
    r = random.Random(hash(prefix) & 0xffffffff)
    return [r.uniform(-2.0, 2.0) for _ in VOCAB]

def sample_masked(logits, allowed):
    probs = [math.exp(l) if c in allowed else 0.0 for c, l in zip(VOCAB, logits)]
    s = sum(probs)
    if s == 0: raise RuntimeError(f"Dead end, allowed={allowed}")
    x, acc = random.random() * s, 0.0
    for c, p in zip(VOCAB, probs):
        acc += p
        if acc >= x: return c
    return VOCAB[-1]

HEAD = '{"city":"'
MID  = '","unit":"'
TAIL = '"}'

def constrained_decode(max_steps=500):
    out, state, i = "", "HEAD", 0
    city_len = 0
    unit_opts = ["celsius", "fahrenheit"]
    unit_typed = ""

    for _ in range(max_steps):
        if state == "HEAD":
            allowed = {HEAD[i]}
        elif state == "CITY":
            # 1–20 lowercase letters only, then close quote
            allowed = set("abcdefghijklmnopqrstuvwxyz")
            if city_len >= 1: allowed.add('"')     # may close after >=1 char
            if city_len >= 20: allowed = {'"'}     # must close at 20
        elif state == "MID":
            allowed = {MID[i]}
        elif state == "UNIT":
            # constrain to prefixes of allowed literals; after full literal -> must close quote
            allowed = set()
            for lit in unit_opts:
                if lit.startswith(unit_typed):
                    if len(unit_typed) == len(lit): allowed.add('"')
                    else: allowed.add(lit[len(unit_typed)])
        elif state == "TAIL":
            allowed = {TAIL[i]}
        else:
            raise RuntimeError("bad state")

        ch = sample_masked(logits(out), allowed)
        out += ch

        # state transitions
        if state == "HEAD":
            i += 1
            if i == len(HEAD): state, i = "CITY", 0
        elif state == "CITY":
            if ch == '"': state, i = "MID", 0
            else: city_len += 1
        elif state == "MID":
            i += 1
            if i == len(MID): state, i = "UNIT", 0
        elif state == "UNIT":
            if ch == '"': state, i = "TAIL", 0
            else: unit_typed += ch
        elif state == "TAIL":
            i += 1
            if i == len(TAIL): return out

    raise RuntimeError("did not terminate")

print(constrained_decode())
