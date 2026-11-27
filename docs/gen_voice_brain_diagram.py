"""Generate brain_architecture.excalidraw (+ SVG previews per frame)."""
import json, random, sys, textwrap
from xml.sax.saxutils import escape

OUT = sys.argv[1]
PREVIEW_DIR = sys.argv[2]
random.seed(7)
EL = []
FRAMES = []  # (name, x, y, w, h)
CW = 0.56  # avg char width factor (Helvetica)
LH = 1.25

C = dict(
    blue="#a5d8ff", yellow="#ffec99", green="#b2f2bb", purple="#d0bfff", red="#ffc9c9",
    teal="#96f2d7", orange="#ffd8a8", gray="#e9ecef", white="#ffffff", pink="#fcc2d7",
    dgreen="#8ce99a", lred="#ffa8a8", lyellow="#fff3bf", caller="#f1f3f5", agent="#d3f9d8",
)


def _id():
    return "e%08x" % random.getrandbits(32)


def _base(t, x, y, w, h, **kw):
    e = dict(id=_id(), type=t, x=x, y=y, width=w, height=h, angle=0,
             strokeColor=kw.get("stroke", "#1e1e1e"), backgroundColor=kw.get("bg", "transparent"),
             fillStyle="solid", strokeWidth=kw.get("sw", 1), strokeStyle=kw.get("style", "solid"),
             roughness=0, opacity=100, groupIds=[], frameId=None,
             roundness={"type": 3} if t == "rectangle" and kw.get("round", True) else None,
             seed=random.randint(1, 2**30), version=1, versionNonce=random.randint(1, 2**30),
             isDeleted=False, boundElements=[], updated=1, link=None, locked=False)
    EL.append(e)
    return e


def wrap(s, width_px, fs):
    maxc = max(8, int(width_px / (fs * CW)))
    out = []
    for para in s.split("\n"):
        out += textwrap.wrap(para, maxc) or [""]
    return out


def text(x, y, s, fs=16, w=None, align="left", color="#1e1e1e", bold=False):
    lines = wrap(s, w, fs) if w else s.split("\n")
    tw = w if w else max(len(l) for l in lines) * fs * CW
    th = len(lines) * fs * LH
    t = "\n".join(lines)
    e = _base("text", x, y, tw, th, stroke=color)
    e.update(text=t, originalText=t, fontSize=fs, fontFamily=2, textAlign=align,
             verticalAlign="top", containerId=None, lineHeight=LH, autoResize=not w)
    e["_bold"] = bold
    return th


def box(x, y, w, h, title=None, body=None, bg=C["white"], fs=15, tfs=None, align="left",
        stroke="#1e1e1e", style="solid", sw=1, pad=10):
    """Rectangle with optional bold title + wrapped body. h=None → auto height."""
    tfs = tfs or fs + 2
    need = pad
    tl = wrap(title, w - 2 * pad, tfs) if title else []
    bl = wrap(body, w - 2 * pad, fs) if body else []
    need += len(tl) * tfs * LH + (6 if tl and bl else 0) + len(bl) * fs * LH + pad
    h = h or need
    _base("rectangle", x, y, w, h, bg=bg, stroke=stroke, style=style, sw=sw)
    cy = y + pad
    if title:
        cy += text(x + pad, cy, title, tfs, w - 2 * pad, "center" if align == "center" else "left", bold=True) + 6
    if body:
        text(x + pad, cy, body, fs, w - 2 * pad, align)
    return h


def arrow(pts, label=None, color="#1e1e1e", style="solid", fs=13, sw=2, both=False):
    x0, y0 = pts[0]
    rel = [[px - x0, py - y0] for px, py in pts]
    xs = [p[0] for p in rel]; ys = [p[1] for p in rel]
    e = _base("arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys), stroke=color, style=style, sw=sw)
    e.update(points=rel, lastCommittedPoint=None, startBinding=None, endBinding=None,
             startArrowhead="arrow" if both else None, endArrowhead="arrow", elbowed=False)
    if label:
        mx = sum(p[0] for p in pts[:2]) / 2; my = sum(p[1] for p in pts[:2]) / 2
        text(mx + 6, my - fs - 2, label, fs, color=color)


def frame(name, x, y, w, h):
    e = _base("frame", x, y, w, h)
    e.update(name=name, roundness=None)
    FRAMES.append((name, x, y, w, h))


# =====================================================================
# FRAME 1 — ARCHITECTURE
# =====================================================================
X, Y = 0, 0
frame("1. Architecture — Stateful Two-Speed Guarded Brain", X - 40, Y - 40, 2080, 1640)
text(X, Y, "The Brain: Stateful · Two-Speed · Guarded", 36, bold=True)
text(X, Y + 50, "One core engine for every business. Behaviour comes from config (goals, facts, policies, tools), "
     "truth comes from typed state + receipts, speed comes from a small talker with a big thinker running in parallel.", 17, 1950)

# caller
box(X, Y + 140, 200, 130, "CALLER", "PSTN / SIP\n(Exotel, Plivo…)\n8 kHz audio", C["gray"], align="center")
# perception
box(X + 250, Y + 120, 1220, 190, None, None, "#e7f5ff", style="dashed")
text(X + 262, Y + 128, "① PERCEPTION  (streaming, every 20 ms audio frame)", 18, bold=True)
pw = 230
P = [("Indic ASR", "streaming partials, n-best + per-word confidence, tuned on 8 kHz calls"),
     ("Language / code-switch ID", "hi-en, ta-en, mr… per segment → talker mirrors it"),
     ("Turn detector", "semantic + acoustic end-of-turn; barge-in vs echo vs 'hmm' backchannel"),
     ("Paralinguistics", "emotion, pace, noise level, frustration"),
     ("Entity mode", "switched on by state: digit / name / address / date grammars")]
for i, (t, b) in enumerate(P):
    box(X + 262 + i * (pw + 10), Y + 160, pw, 140, t, b, C["blue"], fs=13, tfs=15)
arrow([(X + 200, Y + 205), (X + 250, Y + 205)])

# config stack
cx = X + 1530
text(cx, Y + 120, "⑦ CONFIG LAYERS (plug & play)", 18, bold=True)
CFG = [("Tenant config", "business facts (prices, timings, doctors, menu), persona, voice, tools, escalation numbers", "#f8f9fa"),
       ("Vertical pack", "clinic / restaurant / lending…: goal graphs, vocab, compliance rules, eval sets", "#e9ecef"),
       ("Skills library", "capture_phone, book_slot, verify_identity, take_payment, handle_objection, escalate…", "#dee2e6"),
       ("Core engine", "this whole diagram — identical for every customer", "#ced4da")]
cy = Y + 155
for t, b, bg in CFG:
    cy += box(cx, cy, 470, None, t, b, bg, fs=13, tfs=15) + 8

# call state
sy = Y + 360
box(X + 250, sy, 1220, 230, None, None, C["lyellow"], sw=2)
text(X + 262, sy + 8, "② CALL STATE — typed, single source of truth (not the transcript)", 18, bold=True)
S = [("Goal progress", "active goal, step in goal graph, end-state"),
     ("Slots", "value · confidence · confirmed? · source"),
     ("Tool receipts", "BK-4471 booked, H-31 held, SMS-19 sent"),
     ("Commitments", "promises made to caller → must be fulfilled"),
     ("Caller profile", "known/new, language, sentiment trend"),
     ("Policy flags", "AI disclosed, consent, DND, call window"),
     ("Spoken ledger", "what the agent ACTUALLY said (barge-in safe)"),
     ("Allowed next moves", "computed from goal graph + policies")]
for i, (t, b) in enumerate(S):
    r, c = divmod(i, 4)
    box(X + 262 + c * 300, sy + 45 + r * 90, 290, 80, t, b, C["yellow"], fs=13, tfs=15)
arrow([(X + 860, Y + 310), (X + 860, sy)], "events + transcript")

# talker / thinker
ty = sy + 300
box(X + 250, ty, 560, 230, "③ FAST TALKER  (System 1)",
    "• small fine-tuned LLM (≈8B), self-hosted in Mumbai, first token ≈100–150 ms\n"
    "• input = compact STATE + last 2 turns (not whole transcript)\n"
    "• starts drafting on ASR partials (speculative), commits at turn end\n"
    "• owns conversational glue: empathy, acks, honest fillers, mirroring Hinglish\n"
    "• never decides facts or actions itself", C["green"], fs=14)
box(X + 910, ty, 560, 230, "④ THINKER  (System 2, parallel)",
    "• frontier-grade LLM, runs off the critical path\n"
    "• plans the next move from goal graph, detects multi-intent & self-repairs\n"
    "• calls tools ONLY via the deterministic executor\n"
    "• writes state diffs; injects 'next move' into talker\n"
    "• decides escalation (frustration, out-of-scope, repeated failure)", C["purple"], fs=14)
arrow([(X + 530, sy + 230), (X + 530, ty)], "reads state")
arrow([(X + 1190, sy + 230), (X + 1190, ty)], "reads state")
arrow([(X + 1290, ty), (X + 1290, sy + 230)], "writes diffs", color="#6741d9")
arrow([(X + 910, ty + 150), (X + 810, ty + 150)], color="#6741d9")
text(X + 818, ty + 105, "inject\nnext move", 12, color="#6741d9")
arrow([(cx, Y + 590), (X + 1470, Y + 560)], style="dashed", color="#495057")
text(X + 1478, Y + 595, "compiled at\nonboarding", 12, color="#495057")

# tools column
tx = cx
box(tx, ty - 20, 470, 75, "Tool executor (deterministic code)", "validates args, idempotency keys, timeouts → returns RECEIPTS", C["orange"], fs=13, tfs=15)
box(tx, ty + 65, 230, 105, "Fact store (typed)", "prices, timings, menu, doctors — each fact has an ID", C["orange"], fs=13, tfs=15)
box(tx + 240, ty + 65, 230, 105, "Policy engine (code)", "TRAI windows, DND, DPDP consent, 'no medical advice'", C["orange"], fs=13, tfs=15)
box(tx, ty + 180, 470, 70, "Business systems", "calendar / HIS · POS / table mgmt · CRM · UPI · SMS / WhatsApp", "#ffe8cc", fs=13, tfs=15)
arrow([(X + 1470, ty + 40), (tx, ty + 15)], color="#e8590c")
text(X + 1476, ty + 45, "tool\ncalls", 12, color="#e8590c")
arrow([(tx + 235, ty + 55), (tx + 235, ty + 65)], color="#e8590c")
# human
box(X, ty + 40, 200, 150, "Human agent", "warm transfer: gets state summary, caller never repeats", C["pink"], fs=13, tfs=15, align="center")
arrow([(X + 910, ty + 200), (X + 870, ty + 260), (X + 120, ty + 260), (X + 120, ty + 190)], None, color="#c2255c", style="dashed")
text(X + 300, ty + 238, "escalate", 13, color="#c2255c")

# guards
gy = ty + 300
box(X + 250, gy, 1220, 170, None, None, "#fff5f5", style="dashed")
text(X + 262, gy + 8, "⑤ OUTPUT GUARDS — deterministic, 5–20 ms, every sentence before it is spoken", 18, bold=True)
G = [("Action-claim guard", "'booked / sent / cancelled' only if a matching receipt exists in state"),
     ("Grounding guard", "every price, date, timing, policy must map to a fact ID — else 'let me confirm'"),
     ("Compliance / safety guard", "AI disclosure, consent, PII masking, domain bans (medical / financial advice)")]
for i, (t, b) in enumerate(G):
    box(X + 262 + i * 400, gy + 45, 390, 110, t, b, C["red"], fs=13, tfs=15)
arrow([(X + 530, ty + 230), (X + 530, gy)])
text(X + 538, gy - 22, "sentence stream", 13)

# tts
vy = gy + 210
box(X + 250, vy, 1220, 90, "⑥ EXPRESSIVE INDIC TTS",
    "talker emits text + style tags (warm / apologetic / brisk) · Indian number & name normalisation ('₹1.5 lakh' → 'dedh lakh') · native code-switch · "
    "controlled natural pauses · stops in <100 ms on barge-in and reports exactly what was spoken to the Spoken ledger", C["teal"], fs=13)
arrow([(X + 860, gy + 170), (X + 860, vy)])
arrow([(X + 250, vy + 45), (X + 228, vy + 45), (X + 228, Y + 300), (X + 100, Y + 300), (X + 100, Y + 270)], color="#0c8599")
text(X + 120, Y + 304, "audio out", 13, color="#0c8599")

# learning loop
ly = vy + 140
text(X, ly, "⑧ LEARNING LOOP (the moat — runs after every call)", 18, bold=True)
L = [("Call log + state trace", "every turn: state diff, receipts, guard hits"),
     ("Auto-scoring", "task success, guard blocks, latency, sentiment, escalations"),
     ("Failure clustering", "'caller says I already checked → generic reply'"),
     ("Sim-caller regression", "Indian personas, noise, code-mix, barge-ins — gates every change"),
     ("Ship fix", "config / skill / fine-tune update in hours, to all tenants of the vertical")]
for i, (t, b) in enumerate(L):
    box(X + i * 400, ly + 35, 360, 90, t, b, "#e6fcf5", fs=13, tfs=15)
    if i < 4:
        arrow([(X + i * 400 + 360, ly + 80), (X + i * 400 + 400, ly + 80)])

# latency budget
by = ly + 170
text(X, by, "Latency budget on the critical path (target p50 ≈ 700 ms mouth-to-ear; thinker is NOT on this path unless a tool result is needed → honest filler)", 16, 1990, bold=True)
segs = [("turn detect 250", 250, C["blue"]), ("talker TTFT 120", 120, C["green"]), ("guards 15", 15, C["red"]),
        ("TTS first audio 180", 180, C["teal"]), ("network + telco 150", 150, C["gray"])]
bx = X
for lab, ms, col in segs:
    w = ms * 2.0
    _base("rectangle", bx, by + 50, w, 44, bg=col, round=False)
    if ms > 20:
        text(bx + 6, by + 62, lab, 14)
    else:
        text(bx - 10, by + 100, lab, 13)
    bx += w
text(bx + 15, by + 60, "= ~715 ms", 20, bold=True)

# =====================================================================
# SIM-CALL FRAMES (2, 3)
# =====================================================================
COLS = [("Dialogue (what is heard)", 470), ("① Perception", 250), ("② Call-state diff", 320),
        ("③ Fast talker", 250), ("④ Thinker + tools", 330), ("⑤ Guards", 280)]
COLBG = [None, C["blue"], C["yellow"], C["green"], C["purple"], C["red"]]


def sim_frame(fx, fy, title, subtitle, rows, fname):
    text(fx, fy, title, 32, bold=True)
    text(fx, fy + 45, subtitle, 16, 1950)
    y = fy + 110
    x = fx
    for (h, w), bg in zip(COLS, COLBG):
        box(x, y, w, 40, h, None, bg or C["gray"], tfs=15, align="center")
        x += w + 10
    y += 50
    for label, dialogue, cells in rows:
        fs = 13
        need = 0
        dl = []
        for who, line in dialogue:
            dl.append(wrap(f"{who}: {line}", COLS[0][1] - 20, fs))
        dh = sum(len(l) for l in dl) * fs * LH + 8 * len(dl) + 40
        for (h, w), c in zip(COLS[1:], cells):
            need = max(need, len(wrap(c, w - 20, fs)) * fs * LH + 20)
        rh = max(dh, need, 70)
        # dialogue cell: separate bubbles
        _base("rectangle", fx, y, COLS[0][1], rh, bg="#ffffff")
        text(fx + 8, y + 6, label, 13, color="#868e96", bold=True)
        cy = y + 26
        for (who, line), lines in zip(dialogue, dl):
            bh = len(lines) * fs * LH + 6
            _base("rectangle", fx + 6, cy - 3, COLS[0][1] - 12, bh, bg=C["agent"] if who == "AGENT" else C["caller"], stroke="transparent")
            text(fx + 10, cy, "\n".join(lines), fs, color="#1e1e1e")
            cy += bh + 6
        x = fx + COLS[0][1] + 10
        for (h, w), c, bg in zip(COLS[1:], cells, COLBG[1:]):
            _base("rectangle", x, y, w, rh, bg=bg if c.strip() not in ("—", "") else "#f8f9fa")
            text(x + 10, y + 8, c, fs, w - 20)
            x += w + 10
        y += rh + 12
    return y


CLINIC = [
    ("PRE-CALL  (t = −200 ms)", [("SYSTEM", "Inbound call rings — Arogya Dental Clinic, Pune. Mon 19 Oct, 18:40.")],
     ["Telco caller-ID: +91 98xxx (not in CRM → new patient).",
      "tenant = arogya_dental\nvertical = clinic\ngoals loaded: book · reschedule · cancel · clinic_info · triage\nflags: ai_disclosed = false",
      "Greeting is a template — no LLM call, 0 ms thinking.",
      "Pre-fetch: today + tomorrow calendar into cache; CRM lookup (miss).",
      "Compliance: turn 1 MUST contain AI disclosure + recording notice."]),
    ("TURN 1  (t = 0.0 s → 6.1 s)", [
        ("AGENT", "Namaste, Arogya Dental Clinic se main Priya, clinic ki AI assistant. Call quality ke liye record ho sakti hai. Bataiye, kaise madad karun?"),
        ("CALLER", "Haan hello… mujhe Dr. Mehta ka appointment chahiye kal ke liye, daant mein bahut dard hai yaar.")],
     ["ASR partials stream in.\nlang = hi-en (code-mixed)\nemotion = distress (pitch ↑, pace ↑)\nTurn end predicted 220 ms after 'yaar' (complete sentence + falling pitch).",
      "ai_disclosed = true ✔\ngoal = book_appointment\ndoctor = 'Dr. Mehta' → fact D-2: Dr. Anil Mehta, Endodontist (conf .93)\ndate = 'kal' → 2026-10-20\nsymptom = severe_pain\nsentiment = distressed",
      "Speculatively drafted on partial '…appointment chahiye', committed at turn end. Leads with empathy; uses honest filler because a tool is running.",
      "Plan: calendar.get_slots(D-2, 20-Oct) → 240 ms → [11:30, 17:00].\nVertical rule: severe pain → triage Q (swelling / fever?).\nInject: offer both slots + triage Q.",
      "Filler 'ek second… dekh rahi hoon' ✔ allowed: get_slots is genuinely in flight.\nNo claims yet."]),
    ("TURN 2  (barge-in)", [
        ("AGENT", "Arre, sorry, dard mein hain aap. Ek second, main kal ke slots dekh rahi hoon… Kal Dr. Mehta ke paas subah 11:30 aur shaam 5 baje hai. Aur kya sujan ya bukhaar bhi—"),
        ("CALLER", "(interrupts) Nahi nahi, sujan nahi hai. Shaam wala de do.")],
     ["Barge-in = caller speech over TTS (echo-cancelled, not a 'hmm' backchannel) → TTS stopped in 90 ms.",
      "Spoken ledger: slots + 'sujan ya bukhaar' WERE heard → don't repeat.\nswelling = no; urgent = false\ntime = 17:00 ✔",
      "Next required slots from goal graph: name + phone → asks both in one natural line.",
      "calendar.hold(D-2, 20-Oct 17:00, 5 min) → receipt H-882 (no double booking while we collect details).",
      "Only a HOLD receipt exists → 'booked' is still forbidden."]),
    ("TURN 3  (entity capture)", [
        ("AGENT", "Theek hai, kal shaam 5 baje rakhti hoon. Aapka poora naam aur mobile number bata dijiye?"),
        ("CALLER", "Rohit Verma. Number hai nine eight seven six five, four three two zero one.")],
     ["ENTITY MODE on (state expects name + 10-digit mobile).\nDigit grammar n-best:\n 9876543201 (.71)\n 9876543210 (.22)\nname 'Rohit Verma' (.95)",
      "name = Rohit Verma (conf .95 → no confirm)\nphone = 9876543201 (conf .71 < .85 → needs_confirmation)",
      "Confirm ONLY the low-confidence slot, chunked like a human reads numbers (5 + 5).",
      "Validator: 10 digits, valid Indian mobile prefix ✔",
      "PII guard: number masked in logs (98xxxxx201)."]),
    ("TURN 4  (multi-intent + guard block)", [
        ("AGENT", "Number confirm kar loon — 98765, 43201?"),
        ("CALLER", "Haan sahi hai. Aur fees kitni hai? Aur tab tak koi painkiller le loon kya?")],
     ["3 intents in one turn: confirm + ask_fee + ask_medication.",
      "phone confirmed ✔ → all required slots filled → book precondition met\nopen questions: fee, medication",
      "Draft: 'Booking ho gayi. Fees ₹500 hai. Aap ibuprofen le sakte hain.'",
      "calendar.book(H-882, Rohit, 98…201) → receipt BK-4471 ✔\nsms.send(confirmation) → SMS-19 ✔\nfact F-12: consultation ₹500 (RCT extra)",
      "Action-claim 'booking ho gayi' ✔ (BK-4471)\nGrounding '₹500' ✔ (F-12)\nPOLICY BLOCK ✘ 'ibuprofen' — no medical advice → swapped for safe template + emergency number"]),
    ("TURN 5  (close)", [
        ("AGENT", "Aapki appointment confirm ho gayi — kal, 20 October, shaam 5 baje, Dr. Anil Mehta ke saath. Consultation ₹500 hai. Dawai ke baare mein main salah nahi de sakti, doctor hi batayenge — dard ya sujan badhe to emergency line 020-xxxx pe call kijiye. SMS bhej diya hai."),
        ("CALLER", "Ok, thank you ji.")],
     ["Closing intent; sentiment distressed → calm.",
      "goal = book_appointment → DONE (BK-4471)\ncommitments: SMS ✔ sent\nend_state = success",
      "Warm closing template (no LLM needed).",
      "Post-call: CRM.create_patient + summary; scorer: success, 1 guard block, p50 latency 680 ms.",
      "End check: 0 unfulfilled commitments ✔"]),
]

RESTO = [
    ("PRE-CALL", [("SYSTEM", "Inbound call — Spice Route, Bandra (Mumbai). Mon 19 Oct, 16:05.")],
     ["Caller-ID verified by telco: +91 99xxx.",
      "tenant = spice_route\nvertical = restaurant\ngoals: reserve · modify · cancel · takeaway · info\nfacts: F-7 Jain menu; F-9 outside cake ₹300 fee\npolicy P-3: party ≥ 8 → ₹2,000 deposit",
      "Template greeting.",
      "Pre-fetch: this week's table availability.",
      "AI disclosure required in turn 1."]),
    ("TURN 1  (3 intents, language switch)", [
        ("AGENT", "Hi, Spice Route Bandra, this is Aarav, the restaurant's AI assistant. How can I help?"),
        ("CALLER", "Hi, I want a table this Saturday, 8:30, for 8 people. Ek-do Jain bhi honge, and can we bring our own cake? Birthday hai.")],
     ["lang switch en → hi-en mid-utterance.\nIntents: reserve_table + dietary + policy question (cake) + occasion.",
      "goal = reserve_table\ndate = Sat 24-Oct, time = 20:30, party = 8\ndietary = [jain ×2]\noccasion = birthday\nopen_q: outside_cake",
      "Mirrors caller's Hinglish. Leads with the problem + alternatives, then answers the questions.",
      "pos.availability(24-Oct 20:30, 8) → none; alternatives 19:45, 21:15.\nF-7 Jain ✔ · F-9 cake ✔ · P-3 deposit applies.",
      "Grounding: Jain ✔F-7, cake fee ✔F-9 → all claims cite facts."]),
    ("TURN 2  (self-repair invalidates state)", [
        ("AGENT", "Happy birthday in advance! Saturday 8:30 pe 8 logon ki table full hai, but 7:45 ya 9:15 available hai. Jain menu hai, aur apna cake la sakte hain — ₹300 cutting charge lagta hai."),
        ("CALLER", "Hmm, 9:15 chalega. Oh wait — actually 10 log ho jayenge.")],
     ["'actually' + number → self-correction (slot repair, not a new intent).",
      "time = 21:15\nparty 8 → 10 (repair)\n⚠ availability receipt was for 8 → marked STALE",
      "Filler while re-checking: 'Ek second, 10 ke liye check karta hoon.'",
      "Re-plan: pos.availability(21:15, 10) → none; 21:30 with joined tables T12+T13 ✔",
      "BLOCKS stale claim '9:15 confirmed' — receipt doesn't cover party=10."]),
    ("TURN 3  (deposit consent + hallucination caught)", [
        ("AGENT", "10 logon ke liye 9:15 nahi ho payega, but 9:30 pe combined table mil jayegi. 8+ guests ke liye ₹2,000 deposit hai, bill mein adjust ho jayega. Chalega?"),
        ("CALLER", "Ok fine. Aur parking hai kya wahan?")],
     ["'Ok fine' = explicit affirmative → consent captured.",
      "time = 21:30 ✔\ndeposit_consent = yes (timestamped, DPDP-style log)\nopen_q: parking",
      "Draft: 'Haan, valet parking available hai.'  ← invented",
      "pos.hold(T12+T13, 21:30) → H-31\npayments.upi_link(₹2,000) → PAY-77\nfact lookup 'parking' → NOT FOUND",
      "GROUNDING BLOCK ✘ 'valet parking' has no fact → honest unknown + creates follow-up task (manager to SMS)."]),
    ("TURN 4  (caller-ID shortcut)", [
        ("AGENT", "Parking ke baare mein mere paas confirmed info nahi hai — manager se confirm karwa ke aapko SMS kar dunga. Booking kis naam pe karun, aur number yahi rakhun?"),
        ("CALLER", "Neha Kapoor, haan isi number pe.")],
     ["'isi number pe' → resolve to caller-ID. No digits to recognise → zero entity risk.",
      "name = Neha Kapoor\nphone = caller_id (telco-verified, conf 1.0)\ncommitments += parking_sms",
      "Short, warm ack.",
      "whatsapp.send(PAY-77 link) → WA-5 ✔\nP-3: booking final ONLY after payment webhook.",
      "May say 'link bhej diya' (WA-5) ✔ — may NOT say 'table confirmed'."]),
    ("TURN 5  (close with open commitments)", [
        ("AGENT", "Thanks Neha! WhatsApp pe ₹2,000 ka payment link bhej diya hai. Payment hote hi table confirm ho jayegi aur message aa jayega. Aur kuch?"),
        ("CALLER", "Nahi bas, thanks!")],
     ["Closing.",
      "goal = reserve_table → PENDING_PAYMENT (H-31 expires in 30 min)\nopen commitments: payment confirm, parking SMS",
      "Closing template.",
      "Post-call: webhook → pos.confirm → WhatsApp confirmation. Unpaid at 25 min → outbound reminder skill. Parking task → manager.",
      "Commitments ledger: 2 open → tracked until done."]),
]

FY2 = 1700
frame("2. Simulated call — Clinic (Hinglish, booking + triage)", -40, FY2 - 40, 2080, 1500)
end2 = sim_frame(0, FY2, "Simulated call #1 — Arogya Dental Clinic (inbound booking)",
                 "Read each row left → right: what was heard, what each component did in that same turn. Shows: honest fillers, barge-in, confidence-gated entity confirm, multi-intent, receipt-gated claims, a blocked medical-advice hallucination.",
                 CLINIC, "clinic")
FY3 = end2 + 120
frame("3. Simulated call — Restaurant (reservation, self-repair, deposit)", -40, FY3 - 40, 2080, 1450)
end3 = sim_frame(0, FY3, "Simulated call #2 — Spice Route, Bandra (table reservation)",
                 "Same core engine, different vertical pack + tenant config. Shows: multi-intent with language switch, self-repair invalidating a previous tool result, consent capture, a blocked parking hallucination, open commitments carried past the call.",
                 RESTO, "resto")
# fix frame heights
for e in EL:
    if e["type"] == "frame" and e["name"].startswith("2."):
        e["height"] = end2 - FY2 + 60
    if e["type"] == "frame" and e["name"].startswith("3."):
        e["height"] = end3 - FY3 + 60
FRAMES[1] = (FRAMES[1][0], -40, FY2 - 40, 2080, end2 - FY2 + 60)
FRAMES[2] = (FRAMES[2][0], -40, FY3 - 40, 2080, end3 - FY3 + 60)

# =====================================================================
# FRAME 4 — THREE BRAINS SIDE BY SIDE
# =====================================================================
FY4 = end3 + 140
text(0, FY4, "Three brains, same caller sentence", 32, bold=True)
text(0, FY4 + 45, "Input (restaurant):  \"Saturday 8:30 ke liye 8 log, ek-do Jain bhi, and can we bring our own cake?\"  …then 2 turns later: \"actually 10 log ho jayenge\"", 17, 1990)
colw = 630
ARCH = [
    ("A. Hybrid: NLU intent + slot extractor + LLM", "#fff4e6", [
        ("ASR (1-best text)", C["blue"]),
        ("NLU intent classifier → top-1 intent", C["orange"]),
        ("Slot extractor (NER / regex / CRF) per intent schema", C["orange"]),
        ("Dialogue manager: rules / state machine per intent", C["orange"]),
        ("Templates  |  LLM fallback for out-of-schema turns", C["purple"]),
        ("TTS", C["teal"])],
     "intent = book_table ✔ (top-1 only)\nslots: day ✔ time ✔ party ✔\n'ek-do Jain' ✘ no slot in schema → dropped\ncake question ✘ second intent ignored, OR sent to LLM fallback that has no grounding → may invent 'yes, free'\nRepair 8→10: works only if a 'modify_party' intent was trained\n→ Fast and cheap, but the caller has to repeat themselves; feels robotic"),
    ("B. Pure LLM: one prompt + tools", "#f3f0ff", [
        ("ASR (text)", C["blue"]),
        ("Big system prompt (persona + all business info + rules)", C["purple"]),
        ("+ full transcript every turn (grows)", C["purple"]),
        ("LLM decides what to say AND which tool to call", C["purple"]),
        ("(no checks)", C["gray"]),
        ("TTS", C["teal"])],
     "Understands all 3 parts ✔, sounds natural ✔\nCake: answers correctly only if the fee is in the prompt; otherwise confidently invents\nRepair 8→10: may keep earlier 'available' belief and say 'done, 9:15 for 10' without re-checking\nMay say 'booked' without calling the tool (τ-Voice failure)\nFrontier model each turn → 1.2–2 s pauses; cost grows with call length"),
    ("C. Stateful two-speed guarded brain", "#ebfbee", [
        ("Perception: ASR n-best + lang + turn + entity mode", C["blue"]),
        ("Typed CALL STATE (goals, slots, receipts, commitments)", C["yellow"]),
        ("Fast talker (small LLM)  ‖  Thinker (frontier LLM)", C["green"]),
        ("Deterministic tool executor + fact store + policy engine", C["orange"]),
        ("Output guards: action-claim · grounding · compliance", C["red"]),
        ("Expressive Indic TTS + spoken ledger", C["teal"])],
     "All 3 parts handled ✔ (thinker = frontier-grade understanding)\nJain → fact F-7, cake fee → fact F-9 (grounded) ✔\nRepair 8→10 marks old receipt STALE → re-check forced by state, not by luck ✔\n'Confirmed' only after payment receipt ✔\n~0.7 s: talker is small + local; tool waits covered by honest fillers\nNLU from (A) can live INSIDE as a cheap fast-path router"),
]
for i, (title, bg, steps, outcome) in enumerate(ARCH):
    x = i * (colw + 30)
    y = FY4 + 100
    _base("rectangle", x, y, colw, 730, bg=bg)
    text(x + 15, y + 12, title, 20, colw - 30, bold=True)
    sy2 = y + 70
    for j, (s, sc) in enumerate(steps):
        box(x + 40, sy2, colw - 80, 58, None, s, sc, fs=15, align="center", pad=8)
        if j < len(steps) - 1:
            arrow([(x + colw / 2, sy2 + 58), (x + colw / 2, sy2 + 78)])
        sy2 += 78
    text(x + 15, sy2 + 10, "Outcome on the test sentence", 17, bold=True)
    box(x + 15, sy2 + 40, colw - 30, None, None, outcome, "#ffffff", fs=14)
frame("4. Three brains side by side", -40, FY4 - 40, 3 * colw + 60 + 80, 890)

# =====================================================================
# FRAME 5 — COMPARISON MATRIX
# =====================================================================
FY5 = FY4 + 960
text(0, FY5, "Head-to-head comparison", 32, bold=True)
text(0, FY5 + 45, "Green = strong · Yellow = acceptable · Red = weak.  ₹ / latency numbers are engineering estimates for India-hosted stacks, not vendor quotes; ASR + TTS + telephony (~₹1–2/min) are the same for all three and excluded.", 16, 1990)
G_, Y_, R_ = "#d3f9d8", "#fff3bf", "#ffe3e3"
M = [
    ("Latency (p50 mouth-to-ear)", (G_, "0.6–0.9 s on template path; 1.3–2 s when it falls back to the LLM"),
     (R_, "1.2–2 s+ with a frontier model each turn; fast only if you use a small (dumber) model"),
     (G_, "≈0.7 s target: small local talker on the critical path, thinker in parallel, honest fillers for tool waits")),
    ("Understanding (open-ended, multi-intent, code-mixed)", (R_, "top-1 intent, fixed slot schema; breaks on 'Jain + cake + birthday' turns"),
     (G_, "excellent — the LLM sees everything"), (G_, "excellent — thinker is frontier-grade; talker handles glue")),
    ("Action truthfulness", (G_, "actions are code paths"), (R_, "can claim actions it never took (τ-Voice)"),
     (G_, "receipt-gated: impossible to say 'booked' without BK-id")),
    ("Hallucinated facts / policy", (Y_, "safe on templates, unsafe on LLM fallback"), (R_, "prompt-only grounding; invents fees, parking, refund windows"),
     (G_, "every number/policy must cite a fact ID or becomes 'let me confirm'")),
    ("State tracking over long calls", (Y_, "good inside one intent; weak across intents & repairs"),
     (R_, "transcript = state → facts decay, stale tool results reused"), (G_, "typed state; repairs invalidate dependent receipts")),
    ("Entity capture (phone, name, address)", (Y_, "regex/grammars good, but on ASR 1-best only"),
     (R_, "14–41% exact success in τ-Elicitation"), (G_, "entity mode + n-best + confirm only when confidence is low")),
    ("Naturalness / human-likeness", (R_, "templated, repetitive, IVR-like"), (G_, "natural"),
     (G_, "natural + mirrors Hinglish, style-tagged prosody")),
    ("Generic: many businesses / verticals", (R_, "new intents + training data + slot schemas per client"),
     (Y_, "fast to start (write a prompt) but no reuse; quality = prompt author"), (G_, "skills reused across verticals; vertical packs; onboarding compiler")),
    ("Time to onboard a new business", (R_, "2–6 weeks"), (Y_, "1–3 days, then endless prompt tuning"),
     (G_, "1–3 days in a known vertical; 1–2 weeks for a new vertical pack")),
    ("Brain cost per minute (est.)", (G_, "lowest: ≈₹0.1–0.4"), (R_, "≈₹0.5–2; grows with call length (whole transcript re-sent)"),
     (Y_, "≈₹0.2–0.6: small self-hosted talker every turn, thinker only when needed, cached prefixes")),
    ("Scalability (concurrency)", (G_, "CPU NLU scales trivially"), (Y_, "bound by 3rd-party rate limits; TTFT degrades under load"),
     (Y_, "talker GPUs scale horizontally; thinker via multi-provider fallback; more moving parts")),
    ("Debuggability / audit", (G_, "deterministic"), (R_, "'why did it say that?' — opaque"),
     (G_, "per-turn state diff, receipts, guard log")),
    ("Compliance (TRAI, DPDP, RBI windows)", (G_, "hardcoded"), (R_, "asked nicely in the prompt"),
     (G_, "policy-as-code + consent logs + disclosure guard")),
    ("Regression safety when changing things", (Y_, "retraining can shift other intents"), (R_, "one prompt edit changes everything"),
     (G_, "sim-caller regression gate; per-skill blast radius")),
    ("Speed of improvement", (R_, "slow: label data, retrain"), (Y_, "fast but risky"), (G_, "fast and safe: cluster → fix → regression → ship")),
    ("Engineering effort to build", (Y_, "medium"), (G_, "low — demo in a day"), (R_, "high — the honest cost of this design")),
    ("Vendor lock-in", (G_, "low"), (R_, "high on one LLM provider"), (G_, "low: every block swappable (incl. future speech-to-speech talker)")),
]
cw = [380, 530, 530, 530]
y = FY5 + 110
x = 0
for h, w in zip(["Dimension", "A. Hybrid NLU + slots + LLM", "B. Pure LLM (prompt + tools)", "C. Stateful two-speed guarded"], cw):
    box(x, y, w, 44, h, None, C["gray"], tfs=17, align="center")
    x += w + 8
y += 52
for dim, *cells in M:
    fs = 14
    rh = max(len(wrap(c[1], cw[i + 1] - 20, fs)) for i, c in enumerate(cells)) * fs * LH + 22
    rh = max(rh, len(wrap(dim, cw[0] - 20, 15)) * 15 * LH + 22)
    box(0, y, cw[0], rh, None, dim, "#f8f9fa", fs=15)
    x = cw[0] + 8
    for i, (col, t) in enumerate(cells):
        box(x, y, cw[i + 1], rh, None, t, col, fs=fs)
        x += cw[i + 1] + 8
    y += rh + 6
y += 20
verdict = ("VERDICT —  A (Hybrid NLU) wins only on narrow, scripted, very high-volume flows (OTP, payment reminders, surveys): cheapest and deterministic, but it cannot be the brain "
           "of a smart, generic agent.  B (Pure LLM) wins on time-to-demo and naturalness, and loses on everything that costs money in production: truthfulness, hallucination, latency, compliance.  "
           "C is the only design that is smart AND truthful AND fast AND multi-tenant; its price is engineering effort.  Migration: keep your current NLU + slot extractor as a cheap fast-path "
           "router and entity validator INSIDE C — it becomes a component, not the brain.")
vh = box(0, y, sum(cw) + 24, None, None, verdict, "#e7f5ff", fs=17, sw=2)
frame("5. Comparison matrix", -40, FY5 - 40, sum(cw) + 24 + 80, y + vh - FY5 + 80)

# assign elements to frames
for e in EL:
    if e["type"] == "frame":
        continue
    for name, fx, fy, fw, fh in FRAMES:
        if fx <= e["x"] <= fx + fw and fy <= e["y"] <= fy + fh:
            fid = next(f["id"] for f in EL if f["type"] == "frame" and f["name"] == name)
            e["frameId"] = fid
            break

clean = []
for e in EL:
    e2 = {k: v for k, v in e.items() if not k.startswith("_")}
    clean.append(e2)
# frames must come before children? Excalidraw is fine either way, but put frames first.
clean.sort(key=lambda e: 0 if e["type"] == "frame" else 1)
doc = dict(type="excalidraw", version=2, source="https://excalidraw.com", elements=clean,
           appState=dict(viewBackgroundColor="#ffffff", gridSize=None), files={})
json.dump(doc, open(OUT, "w"), ensure_ascii=False, indent=1)


# ---------------- SVG previews ----------------
def svg_for(fx, fy, fw, fh):
    fw = fh = max(fw, fh)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{fw}" height="{fh}" viewBox="{fx} {fy} {fw} {fh}">',
           '<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z"/></marker></defs>',
           f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" fill="#fff"/>']
    for e in EL:
        if not (fx <= e["x"] <= fx + fw and fy <= e["y"] <= fy + fh) or e["type"] == "frame":
            continue
        if e["type"] == "rectangle":
            bg = e["backgroundColor"] if e["backgroundColor"] != "transparent" else "none"
            st = e["strokeColor"] if e["strokeColor"] != "transparent" else "none"
            da = ' stroke-dasharray="8 6"' if e["strokeStyle"] == "dashed" else ""
            out.append(f'<rect x="{e["x"]}" y="{e["y"]}" width="{e["width"]}" height="{e["height"]}" rx="8" fill="{bg}" stroke="{st}" stroke-width="{e["strokeWidth"]}"{da}/>')
        elif e["type"] == "text":
            fs = e["fontSize"]
            wgt = "bold" if e.get("_bold") else "normal"
            for i, l in enumerate(e["text"].split("\n")):
                out.append(f'<text x="{e["x"]}" y="{e["y"] + fs * (i + 1) * LH - fs * 0.25}" font-family="Helvetica" font-size="{fs}" font-weight="{wgt}" fill="{e["strokeColor"]}">{escape(l)}</text>')
        elif e["type"] == "arrow":
            pts = " ".join(f'{e["x"] + p[0]},{e["y"] + p[1]}' for p in e["points"])
            out.append(f'<polyline points="{pts}" fill="none" stroke="{e["strokeColor"]}" stroke-width="2" marker-end="url(#a)"/>')
    out.append("</svg>")
    return "\n".join(out)


for i, (name, fx, fy, fw, fh) in enumerate(FRAMES):
    open(f"{PREVIEW_DIR}/frame{i + 1}.svg", "w").write(svg_for(fx, fy, fw, fh))
print("elements", len(clean), "frames", [(n, int(h)) for n, *_, h in FRAMES])
