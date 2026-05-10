// Single-page client for the FastAPI detector. No build step required.
// Override API base by setting window.AI_DETECTOR_API or ?api=... in the URL.

const params = new URLSearchParams(window.location.search);
const API_BASE =
  params.get("api") ||
  window.AI_DETECTOR_API ||
  (location.protocol.startsWith("http") && location.host
    ? `${location.protocol}//${location.host}`
    : "http://127.0.0.1:8000");

document.getElementById("api-url").textContent = API_BASE;

const $ = (id) => document.getElementById(id);
const codeEl = $("code");
const langEl = $("language");
const problemEl = $("problem-id");
const gutterEl = $("gutter");
const linesEl = $("meta-lines");
const charsEl = $("meta-chars");
const analyzeBtn = $("analyze-btn");
const sampleBtn = $("sample-btn");
const clearBtn = $("clear-btn");
const decisionBadge = $("decision-badge");
const riskScoreEl = $("risk-score");
const gaugeArc = $("gauge-arc");
const gaugeNeedle = $("gauge-needle");
const componentsList = $("components");
const signalsList = $("signals");
const apiStatus = $("api-status");
const apiError = $("api-error");

const SAMPLES = {
  python: `def solve():\n    n = int(input())\n    arr = list(map(int, input().split()))\n    arr.sort()\n    total = 0\n    for i, x in enumerate(arr):\n        total += x * (i + 1)\n    print(total)\n\nif __name__ == "__main__":\n    solve()\n`,
  cpp: `#include <bits/stdc++.h>\nusing namespace std;\nint main(){\n    int n; cin >> n;\n    vector<long long> a(n);\n    for(auto& x : a) cin >> x;\n    sort(a.begin(), a.end());\n    long long ans = 0;\n    for(int i = 0; i < n; ++i) ans += a[i] * (long long)(i + 1);\n    cout << ans << endl;\n    return 0;\n}\n`,
  java: `import java.util.*;\npublic class Main{\n    public static void main(String[] args){\n        Scanner sc = new Scanner(System.in);\n        int n = sc.nextInt();\n        long[] a = new long[n];\n        for(int i = 0; i < n; i++) a[i] = sc.nextLong();\n        Arrays.sort(a);\n        long ans = 0;\n        for(int i = 0; i < n; i++) ans += a[i] * (i + 1);\n        System.out.println(ans);\n    }\n}\n`,
};

function syncMeta() {
  const text = codeEl.value;
  const lines = text.length === 0 ? 1 : text.split("\n").length;
  linesEl.textContent = lines;
  charsEl.textContent = text.length;
  gutterEl.textContent = Array.from({ length: lines }, (_, i) => i + 1).join("\n");
}
codeEl.addEventListener("input", syncMeta);
codeEl.addEventListener("scroll", () => {
  gutterEl.scrollTop = codeEl.scrollTop;
});

sampleBtn.addEventListener("click", () => {
  codeEl.value = SAMPLES[langEl.value] || SAMPLES.python;
  syncMeta();
});
clearBtn.addEventListener("click", () => {
  codeEl.value = "";
  syncMeta();
  resetResult();
});

function setLoading(on) {
  analyzeBtn.classList.toggle("is-loading", on);
  analyzeBtn.disabled = on;
}

function decisionLabel(d) {
  switch (d) {
    case "accept":
      return ["Accept", "badge--accept"];
    case "review":
      return ["Flag for review", "badge--review"];
    case "hold":
      return ["Hold (likely AI)", "badge--hold"];
    default:
      return [d || "—", "badge--idle"];
  }
}

function colorForScore(s) {
  if (s == null || s < 0.4) return "var(--accept)";
  if (s < 0.7) return "var(--review)";
  return "var(--hold)";
}

function glowForScore(s) {
  if (s == null || s < 0.4) return "var(--glow-green)";
  if (s < 0.7) return "0 0 20px rgba(245,158,11,0.30)";
  return "var(--glow-magenta)";
}

function setGauge(score) {
  const clamped = Math.max(0, Math.min(1, score));
  const arcLen = 282;
  gaugeArc.setAttribute("stroke-dasharray", `${arcLen * clamped} ${arcLen}`);
  gaugeArc.style.stroke = colorForScore(clamped);
  gaugeArc.style.filter = `drop-shadow(0 0 6px ${colorForScore(clamped)})`;
  const angle = -90 + clamped * 180;
  gaugeNeedle.style.transform = `rotate(${angle}deg)`;
  riskScoreEl.textContent = clamped.toFixed(2);
  riskScoreEl.style.color = colorForScore(clamped);
  riskScoreEl.style.textShadow = glowForScore(clamped).replace("box-shadow", "text-shadow");
  // pulse glow on result card
  const rc = document.getElementById("result-card");
  if (rc) rc.style.boxShadow = glowForScore(clamped);
}

function setComponents(comp) {
  for (const li of componentsList.querySelectorAll("li.bar")) {
    const key = li.dataset.key;
    const value = comp ? comp[key] : null;
    const fill = li.querySelector(".bar__fill");
    const valEl = li.querySelector(".bar__value");
    if (value == null) {
      fill.style.width = "0%";
      valEl.textContent = "—";
      li.style.opacity = 0.55;
    } else {
      fill.style.width = `${(value * 100).toFixed(0)}%`;
      valEl.textContent = value.toFixed(2);
      li.style.opacity = 1;
    }
  }
}

function setSignals(signals, decision) {
  signalsList.innerHTML = "";
  if (!signals || signals.length === 0) {
    const li = document.createElement("li");
    li.className = "signals__empty";
    li.textContent =
      decision === "accept"
        ? "No suspicious signals detected."
        : "No detection signals returned.";
    signalsList.appendChild(li);
    return;
  }
  for (const s of signals) {
    const li = document.createElement("li");
    li.textContent = s;
    if (decision === "hold") li.classList.add("is-bad");
    else if (decision === "review") li.classList.add("is-warn");
    signalsList.appendChild(li);
  }
}

function resetResult() {
  setGauge(0);
  riskScoreEl.textContent = "—";
  riskScoreEl.style.color = "";
  decisionBadge.className = "badge badge--idle";
  decisionBadge.textContent = "awaiting input";
  const rc = document.getElementById("result-card");
  if (rc) rc.style.boxShadow = "";
  setComponents(null);
  setSignals([], null);
  apiError.hidden = true;
  apiError.textContent = "";
}

async function analyze() {
  const code = codeEl.value.trim();
  if (!code) {
    apiError.hidden = false;
    apiError.textContent = "Please paste some code first.";
    return;
  }
  apiError.hidden = true;
  setLoading(true);
  const subCard = document.getElementById("submission-card");
  const resCard = document.getElementById("result-card");
  if (subCard) subCard.classList.add("is-scanning");
  if (resCard) resCard.classList.add("is-scanning");
  try {
    const resp = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code,
        language: langEl.value,
        problem_id: problemEl.value || null,
      }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text || resp.statusText}`);
    }
    const data = await resp.json();
    setGauge(data.risk_score ?? 0);
    const [label, badgeCls] = decisionLabel(data.decision);
    decisionBadge.className = `badge ${badgeCls}`;
    decisionBadge.textContent = label;
    setComponents(data.component_scores || {});
    setSignals(data.signals || [], data.decision);
  } catch (err) {
    apiError.hidden = false;
    apiError.textContent = `Analyze failed: ${err.message}`;
  } finally {
    setLoading(false);
    if (subCard) subCard.classList.remove("is-scanning");
    if (resCard) resCard.classList.remove("is-scanning");
  }
}

analyzeBtn.addEventListener("click", analyze);

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    analyze();
  }
});

async function pingHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    if (!r.ok) throw new Error(String(r.status));
    const data = await r.json();
    const loaded = Object.values(data.models_loaded || {}).filter(Boolean).length;
    apiStatus.className = "status status--ok";
    apiStatus.textContent = `online · ${loaded} model${loaded === 1 ? "" : "s"} loaded`;
  } catch {
    apiStatus.className = "status status--down";
    apiStatus.textContent = "offline";
  }
}

resetResult();
syncMeta();
pingHealth();
setInterval(pingHealth, 15000);
