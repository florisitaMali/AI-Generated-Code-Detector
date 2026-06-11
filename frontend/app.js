// Single-page client for the FastAPI detector. No build step required.
// Override API base by setting window.AI_DETECTOR_API or ?api=... in the URL.

const params = new URLSearchParams(window.location.search);
const API_BASE =
  params.get("api") ||
  window.AI_DETECTOR_API ||
  (location.protocol.startsWith("http") && location.host
    ? `${location.protocol}//${location.host}`
    : "http://127.0.0.1:8000");

const apiUrlEl = document.getElementById("api-url");
if (apiUrlEl) apiUrlEl.textContent = API_BASE;

const $ = (id) => document.getElementById(id);
const codeEl = $("code");
const langEl = $("language");
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

let detectionMode = "ensemble";

/** Policy thresholds — loaded from GET /health; match backend THRESHOLD_* env vars. */
let thresholdAccept = 0.4;
let thresholdReview = 0.7;

function fmtThreshold(n) {
  const s = Number(n).toFixed(2);
  return s.replace(/\.?0+$/, "") || "0";
}

function applyThresholds(accept, review) {
  if (accept != null && !Number.isNaN(Number(accept))) thresholdAccept = Number(accept);
  if (review != null && !Number.isNaN(Number(review))) thresholdReview = Number(review);
  const la = $("legend-accept");
  const lr = $("legend-review");
  const lh = $("legend-hold");
  if (la) {
    la.innerHTML = `<i style="background:var(--accept)"></i>0–${fmtThreshold(thresholdAccept)} accept`;
  }
  if (lr) {
    lr.innerHTML = `<i style="background:var(--review)"></i>${fmtThreshold(thresholdAccept)}–${fmtThreshold(thresholdReview)} review`;
  }
  if (lh) {
    lh.innerHTML = `<i style="background:var(--hold)"></i>${fmtThreshold(thresholdReview)}–1.0 hold`;
  }
}

const AUTH_TOKEN_KEY = "tracecoder_token";
const AUTH_USER_KEY = "tracecoder_user";

function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

function authHeaders(extra = {}) {
  const token = getAuthToken();
  const headers = { ...extra };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

function setAuthSession(token, user) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
  updateAuthUI();
  loadHistory();
}

function clearAuthSession() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
  updateAuthUI();
  clearHistoryList();
}

function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem(AUTH_USER_KEY) || "null");
  } catch {
    return null;
  }
}

const authGuest = $("auth-guest");
const authUser = $("auth-user");
const userLabel = $("user-label");
const userAvatar = $("user-avatar");
const historySection = $("history-section");
const historyList = $("history-list");
const historyEmpty = $("history-empty");
const googleSigninSlot = $("google-signin-slot");
let supabaseClient = null;

// #region agent log
function agentLog(location, message, data, hypothesisId) {
  fetch("http://127.0.0.1:7607/ingest/5ca97c2d-d639-4e8b-9f2c-89ae596606ba", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Debug-Session-Id": "fba3a1" },
    body: JSON.stringify({
      sessionId: "fba3a1",
      runId: "pre-fix",
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now(),
    }),
  }).catch(() => {});
}
// #endregion

function updateAuthUI() {
  const user = getStoredUser();
  const loggedIn = Boolean(user && getAuthToken());
  if (authGuest) {
    authGuest.hidden = loggedIn;
    authGuest.style.display = loggedIn ? "none" : "";
  }
  if (authUser) {
    authUser.hidden = !loggedIn;
    authUser.style.display = loggedIn ? "" : "none";
  }
  if (historySection) historySection.hidden = !loggedIn;
  if (user && userAvatar && user.avatarUrl) {
    userAvatar.src = user.avatarUrl;
    userAvatar.alt = user.username || "Profile";
    userAvatar.hidden = false;
    if (userLabel) userLabel.hidden = true;
  } else if (user && userLabel) {
    userLabel.textContent = user.username || "Signed in";
    userLabel.hidden = false;
    if (userAvatar) userAvatar.hidden = true;
  } else {
    if (userLabel) userLabel.hidden = true;
    if (userAvatar) userAvatar.hidden = true;
  }
  // #region agent log
  agentLog(
    "app.js:updateAuthUI",
    "auth ui state",
    {
      loggedIn,
      guestHidden: authGuest?.hidden,
      userHidden: authUser?.hidden,
      guestDisplay: authGuest ? getComputedStyle(authGuest).display : null,
      userDisplay: authUser ? getComputedStyle(authUser).display : null,
      hasToken: Boolean(getAuthToken()),
    },
    "H1"
  );
  // #endregion
}

function clearHistoryList() {
  if (historyList) historyList.innerHTML = "";
  if (historyEmpty) historyEmpty.hidden = false;
}

function formatHistoryDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function riskClass(decision) {
  if (decision === "accept") return "history__risk--accept";
  if (decision === "review") return "history__risk--review";
  return "history__risk--hold";
}

function codePreview(text) {
  const oneLine = (text || "").trim().replace(/\s+/g, " ");
  if (!oneLine) return "";
  return oneLine.length <= 120 ? oneLine : `${oneLine.slice(0, 119)}…`;
}

function prependHistoryOptimistic({ code, language, risk_score, detection_mode }) {
  if (!historyList || !historyEmpty) return;
  historyEmpty.hidden = true;
  const li = document.createElement("li");
  li.className = "history__item history__item--pending";
  li.dataset.id = `pending-${Date.now()}`;
  li.dataset.pending = "1";
  const mode = detection_mode || detectionMode;
  const decision = decisionFromScore(risk_score ?? 0);
  li.innerHTML = `
      <span class="history__meta">${language} · ${MODE_LABELS[mode] || mode} · just now</span>
      <span class="history__risk ${riskClass(decision)}">${Number(risk_score).toFixed(2)}</span>
      <button type="button" class="history__delete" title="Delete" aria-label="Delete" disabled>✕</button>
      <span class="history__preview">${escapeHtml(codePreview(code))}</span>
    `;
  historyList.prepend(li);
}

let historySyncTimer = null;
function scheduleHistorySync() {
  if (historySyncTimer) clearTimeout(historySyncTimer);
  historySyncTimer = setTimeout(() => {
    historySyncTimer = null;
    loadHistory();
  }, 600);
}

async function loadHistory() {
  if (!getAuthToken() || !historyList) return;
  try {
    const resp = await fetch(`${API_BASE}/auth/history`, { headers: authHeaders() });
    if (resp.status === 401) {
      // #region agent log
      agentLog("app.js:loadHistory", "history 401 — keeping supabase session", {}, "H4");
      // #endregion
      return;
    }
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    renderHistoryList(data.entries || []);
  } catch (err) {
    console.warn("History load failed:", err);
  }
}

function renderHistoryList(entries) {
  historyList.innerHTML = "";
  if (!entries.length) {
    historyEmpty.hidden = false;
    return;
  }
  historyEmpty.hidden = true;
  for (const entry of entries) {
    const li = document.createElement("li");
    li.className = "history__item";
    li.dataset.id = entry.id;
    li.innerHTML = `
      <span class="history__meta">${entry.language} · ${MODE_LABELS[entry.detection_mode] || entry.detection_mode} · ${formatHistoryDate(entry.created_at)}</span>
      <span class="history__risk ${riskClass(entry.decision)}">${Number(entry.risk_score).toFixed(2)}</span>
      <button type="button" class="history__delete" title="Delete" aria-label="Delete">✕</button>
      <span class="history__preview">${escapeHtml(entry.code_preview || "")}</span>
    `;
    li.addEventListener("click", (e) => {
      if (e.target.closest(".history__delete")) return;
      loadHistoryEntry(entry.id);
    });
    li.querySelector(".history__delete")?.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteHistoryEntry(entry.id);
    });
    historyList.appendChild(li);
  }
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadHistoryEntry(id) {
  try {
    const resp = await fetch(`${API_BASE}/auth/history/${id}`, { headers: authHeaders() });
    if (!resp.ok) throw new Error(await resp.text());
    const entry = await resp.json();
    codeEl.value = entry.code || "";
    langEl.value = entry.language || "python";
    detectionMode = entry.detection_mode || "ensemble";
    document.querySelectorAll(".model-card").forEach((b) => {
      b.classList.toggle("is-selected", (b.dataset.mode || "ensemble") === detectionMode);
    });
    syncMeta();
    setGauge(entry.risk_score ?? 0);
    const decisionUsed = decisionFromScore(entry.risk_score ?? 0);
    const [label, badgeCls] = decisionLabel(decisionUsed);
    decisionBadge.className = `badge ${badgeCls}`;
    decisionBadge.textContent = label;
    setComponents(entry.component_scores || {}, detectionMode);
    setSignals(entry.signals || [], decisionUsed);
    updateModeBadge(detectionMode);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    apiError.hidden = false;
    apiError.textContent = `Could not load history: ${err.message}`;
  }
}

async function deleteHistoryEntry(id) {
  const li = historyList?.querySelector(`.history__item[data-id="${CSS.escape(id)}"]`);
  if (li?.dataset.pending === "1") {
    li.remove();
    if (historyList && !historyList.children.length && historyEmpty) historyEmpty.hidden = false;
    return;
  }
  if (li) {
    li.classList.add("history__item--removing");
    li.remove();
    if (historyList && !historyList.children.length && historyEmpty) historyEmpty.hidden = false;
  }
  try {
    const resp = await fetch(`${API_BASE}/auth/history/${id}`, {
      method: "DELETE",
      headers: authHeaders(),
    });
    if (!resp.ok) throw new Error(await resp.text());
  } catch (err) {
    await loadHistory();
    apiError.hidden = false;
    apiError.textContent = `Delete failed: ${err.message}`;
  }
}

async function syncSessionFromSupabase() {
  if (!supabaseClient) return;
  const { data, error } = await supabaseClient.auth.getSession();
  // #region agent log
  agentLog(
    "app.js:syncSessionFromSupabase",
    "session sync",
    {
      hasError: Boolean(error),
      hasSession: Boolean(data?.session?.access_token),
      userIdPrefix: data?.session?.user?.id?.slice(0, 8) || null,
    },
    "H2"
  );
  // #endregion
  if (error || !data?.session?.access_token) {
    clearAuthSession();
    return;
  }
  const user = data.session.user || {};
  setAuthSession(data.session.access_token, {
    id: user.id || "",
    username: user.email || "supabase-user",
    avatarUrl: user.user_metadata?.avatar_url || user.user_metadata?.picture || "",
  });
}

async function startSupabaseGoogleSignin() {
  if (!supabaseClient) return;
  const redirectTo = `${window.location.origin}${window.location.pathname}`;
  const { error } = await supabaseClient.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo },
  });
  if (error) {
    apiError.hidden = false;
    apiError.textContent = `Supabase login failed: ${error.message}`;
  }
}

async function logoutWithSupabase() {
  // #region agent log
  agentLog("app.js:logoutWithSupabase", "logout clicked", { hasClient: Boolean(supabaseClient) }, "H3");
  // #endregion
  if (supabaseClient) {
    const { error } = await supabaseClient.auth.signOut();
    // #region agent log
    agentLog(
      "app.js:logoutWithSupabase",
      "signOut result",
      { hasError: Boolean(error), errorMsg: error?.message || null },
      "H3"
    );
    // #endregion
  }
  clearAuthSession();
}

function renderSignInButton() {
  if (!googleSigninSlot) return;
  googleSigninSlot.innerHTML = "";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn btn--primary btn--sm";
  btn.textContent = "Sign in with Google";
  btn.addEventListener("click", startSupabaseGoogleSignin);
  googleSigninSlot.appendChild(btn);
}

async function initSupabaseAuth() {
  try {
    const resp = await fetch(`${API_BASE}/auth/supabase/config`);
    if (!resp.ok) throw new Error("Supabase config unavailable");
    const cfg = await resp.json();
    if (!cfg.enabled || !cfg.url || !cfg.anon_key) {
      googleSigninSlot.textContent = "Supabase auth unavailable — save .env and restart API";
      googleSigninSlot.style.fontSize = "12px";
      googleSigninSlot.style.color = "var(--muted)";
      return;
    }
    if (!window.supabase?.createClient) {
      googleSigninSlot.textContent = "Supabase script failed to load";
      return;
    }
    supabaseClient = window.supabase.createClient(cfg.url, cfg.anon_key);
    renderSignInButton();
    supabaseClient.auth.onAuthStateChange((event, session) => {
      // #region agent log
      agentLog(
        "app.js:onAuthStateChange",
        "auth event",
        { event, hasSession: Boolean(session?.access_token) },
        "H2"
      );
      // #endregion
      if (session?.access_token) {
        const u = session.user || {};
        setAuthSession(session.access_token, {
          id: u.id || "",
          username: u.email || "supabase-user",
          avatarUrl: u.user_metadata?.avatar_url || u.user_metadata?.picture || "",
        });
      } else if (event === "SIGNED_OUT") {
        clearAuthSession();
      }
    });
    await syncSessionFromSupabase();
  } catch (err) {
    console.warn("Supabase auth init failed:", err);
  }
}

$("logout-btn")?.addEventListener("click", logoutWithSupabase);
$("history-refresh")?.addEventListener("click", loadHistory);

/** Which component score rows are relevant per detection_mode (must match API). */
const MODE_COMPONENTS = {
  ensemble: ["statistical", "codebert", "llm_judge"],
  fusion: ["statistical", "codebert"],
  stylometric: ["statistical"],
  randomforest: ["random_forest"],
  logisticregression: ["logistic_regression"],
  codebert: ["codebert"],
  graphcodebert: ["graphcodebert"],
  unixcoder: ["unixcoder"],
  llm: ["llm_judge"],
};

const MODE_LABELS = {
  ensemble: "Full ensemble",
  fusion: "Fusion (stat + neural)",
  stylometric: "Stylometric only",
  randomforest: "Random Forest",
  logisticregression: "Logistic Regression",
  codebert: "CodeBERT",
  graphcodebert: "GraphCodeBERT",
  unixcoder: "UniXcoder",
  llm: "LLM judge only",
};

const componentsHeading = document.querySelector(".components h3");
const modeBadge = document.getElementById("mode-badge");

function allowedKeysForMode(mode) {
  return new Set(MODE_COMPONENTS[mode] || MODE_COMPONENTS.ensemble);
}

function updateModeBadge(mode = detectionMode) {
  if (modeBadge) {
    modeBadge.textContent = MODE_LABELS[mode] || mode;
    modeBadge.hidden = false;
  }
  if (componentsHeading) {
    const single = MODE_COMPONENTS[mode]?.length === 1;
    componentsHeading.textContent = single ? "Detector score" : "Component scores";
  }
}

function updateComponentVisibility(mode = detectionMode) {
  const allowed = allowedKeysForMode(mode);
  for (const li of componentsList.querySelectorAll("li.bar")) {
    const key = li.dataset.key;
    const show = allowed.has(key);
    li.hidden = !show;
    if (!show) {
      li.querySelector(".bar__fill").style.width = "0%";
      li.querySelector(".bar__value").textContent = "—";
      li.style.opacity = "0.55";
    }
  }
  updateModeBadge(mode);
}

document.querySelectorAll(".model-card").forEach((btn) => {
  btn.addEventListener("click", () => {
    detectionMode = btn.dataset.mode || "ensemble";
    document.querySelectorAll(".model-card").forEach((b) => b.classList.remove("is-selected"));
    btn.classList.add("is-selected");
    updateComponentVisibility(detectionMode);
    resetResult();
  });
});

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

function decisionFromScore(score) {
  if (score < thresholdAccept) return "accept";
  if (score < thresholdReview) return "review";
  return "hold";
}

function colorForScore(s) {
  if (s == null || s < thresholdAccept) return "var(--accept)";
  if (s < thresholdReview) return "var(--review)";
  return "var(--hold)";
}

function glowForScore(s) {
  if (s == null || s < thresholdAccept) return "var(--glow-green)";
  if (s < thresholdReview) return "0 0 20px rgba(245,158,11,0.30)";
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

function setComponents(comp, mode = detectionMode) {
  const allowed = allowedKeysForMode(mode);
  updateComponentVisibility(mode);

  for (const li of componentsList.querySelectorAll("li.bar")) {
    const key = li.dataset.key;
    if (!allowed.has(key)) continue;

    const value = comp ? comp[key] : null;
    const fill = li.querySelector(".bar__fill");
    const valEl = li.querySelector(".bar__value");
    if (value == null || value === undefined) {
      fill.style.width = "0%";
      valEl.textContent = "—";
      li.style.opacity = 0.55;
    } else {
      fill.style.width = `${(value * 100).toFixed(0)}%`;
      valEl.textContent = Number(value).toFixed(2);
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
  setComponents(null, detectionMode);
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
    const requestedMode = detectionMode;
    const resp = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        code,
        language: langEl.value,
        problem_id: null,
        detection_mode: requestedMode,
      }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status}: ${text || resp.statusText}`);
    }
    const data = await resp.json();
    let modeUsed = data.detection_mode || requestedMode;
    let gaugeRisk = data.risk_score ?? 0;
    const comp = data.component_scores || {};

    // Backward-compatibility fallback: some running servers may ignore detection_mode.
    if (!data.detection_mode) {
      if (requestedMode === "stylometric" && comp.statistical != null) {
        gaugeRisk = Number(comp.statistical);
      } else if (requestedMode === "codebert" && comp.codebert != null) {
        gaugeRisk = Number(comp.codebert);
      } else if (requestedMode === "llm" && comp.llm_judge != null) {
        gaugeRisk = Number(comp.llm_judge);
      } else if (requestedMode === "fusion") {
        const parts = [comp.statistical, comp.codebert].filter((v) => v != null).map(Number);
        if (parts.length) gaugeRisk = parts.reduce((a, b) => a + b, 0) / parts.length;
      }
      modeUsed = requestedMode;
    }
    if (data.detection_mode && data.detection_mode !== detectionMode) {
      console.warn(
        `API used detection_mode=${data.detection_mode} but UI selected ${detectionMode}. Restart uvicorn if this persists.`
      );
    }
    setGauge(gaugeRisk);
    const decisionUsed = decisionFromScore(gaugeRisk);
    const [label, badgeCls] = decisionLabel(decisionUsed);
    decisionBadge.className = `badge ${badgeCls}`;
    decisionBadge.textContent = label;
    setComponents(data.component_scores || {}, modeUsed);
    setSignals(data.signals || [], decisionUsed);
    updateModeBadge(modeUsed);
    // #region agent log
    agentLog(
      "app.js:analyze",
      "analyze done",
      { hasToken: Boolean(getAuthToken()), status: resp.status },
      "H4"
    );
    // #endregion
    if (getAuthToken()) {
      prependHistoryOptimistic({
        code,
        language: langEl.value,
        risk_score: gaugeRisk,
        detection_mode: modeUsed,
      });
      scheduleHistorySync();
    }
  } catch (err) {
    // #region agent log
    agentLog("app.js:analyze", "analyze error", { message: String(err.message).slice(0, 120) }, "H5");
    // #endregion
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
    applyThresholds(data.threshold_auto_accept, data.threshold_flag_review);
    apiStatus.className = "status status--ok";
    apiStatus.textContent = `online · ${loaded} model${loaded === 1 ? "" : "s"} loaded`;
  } catch {
    apiStatus.className = "status status--down";
    apiStatus.textContent = "offline";
  }
}

updateComponentVisibility(detectionMode);
applyThresholds(thresholdAccept, thresholdReview);
resetResult();
syncMeta();
updateAuthUI();
if (getAuthToken()) loadHistory();
initSupabaseAuth();
pingHealth();
setInterval(pingHealth, 15000);
