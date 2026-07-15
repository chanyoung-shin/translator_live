/* ============ LiveBridge 프론트엔드 ============ */
"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  statusPill: $("statusPill"), statusDot: $("statusDot"), statusText: $("statusText"),
  sourceSelect: $("sourceSelect"), startBtn: $("startBtn"), exportBtn: $("exportBtn"),
  summaryBtn: $("summaryBtn"),
  settingsBtn: $("settingsBtn"), feed: $("feed"), feedWrap: $("feedWrap"),
  emptyState: $("emptyState"), scrollDownBtn: $("scrollDownBtn"),
  livePartialRow: $("livePartialRow"), livePartialText: $("livePartialText"),
  liveTransRow: $("liveTransRow"), liveTransText: $("liveTransText"),
  liveIdle: $("liveIdle"), liveChip: $("liveChip"),
  drawer: $("drawer"), drawerBackdrop: $("drawerBackdrop"), drawerClose: $("drawerClose"),
  fontSize: $("fontSize"), fontSizeVal: $("fontSizeVal"),
  showSource: $("showSource"), livePartialTrans: $("livePartialTrans"),
  recordToggle: $("recordToggle"),
  themeSeg: $("themeSeg"), engineSelect: $("engineSelect"),
  asrSelect: $("asrSelect"), silenceSeg: $("silenceSeg"),
  modelSection: $("modelSection"), modelSelect: $("modelSelect"),
  clearBtn: $("clearBtn"), sysInfoBody: $("sysInfoBody"),
};

// ---------- 상태 ----------
let ws = null;
let wsRetry = 0;
let running = false;        // listening 또는 loading (버튼이 '중지'로 동작)
let autoScroll = true;
let lastPartialId = null;   // 지금 라이브 바에 떠 있는 세그먼트 id
let partialStaleTimer = null; // 버려진 세그먼트가 라이브 바에 남지 않게 하는 타이머
const cards = new Map();            // id -> card element
const pendingTranslations = new Map(); // final 카드보다 번역이 먼저 도착한 경우 보관

// ---------- 설정 (localStorage) ----------
const settings = {
  fontSize: 110,
  showSource: true,
  livePartialTrans: true,
  recording: false,
  theme: "dark",
  engine: "local",
  model: "qwen3-4b",
  asrModel: "small",
  silenceMs: 600,
  source: "system",
};

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("livebridge-settings") || "{}");
    Object.assign(settings, saved);
  } catch (e) { /* 무시 */ }
  applySettings();
}

function saveSettings() {
  localStorage.setItem("livebridge-settings", JSON.stringify(settings));
}

function applySettings() {
  document.documentElement.style.setProperty("--caption-scale", settings.fontSize / 100);
  els.fontSize.value = settings.fontSize;
  els.fontSizeVal.textContent = settings.fontSize + "%";
  document.body.classList.toggle("hide-source", !settings.showSource);
  els.showSource.checked = settings.showSource;
  els.livePartialTrans.checked = settings.livePartialTrans;
  els.recordToggle.checked = settings.recording;
  document.documentElement.setAttribute("data-theme", settings.theme);
  els.themeSeg.querySelectorAll(".seg-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.themeVal === settings.theme));
  els.engineSelect.value = settings.engine;
  els.modelSelect.value = settings.model;
  els.modelSection.classList.toggle("hidden", settings.engine !== "local");
  els.asrSelect.value = settings.asrModel;
  els.silenceSeg.querySelectorAll(".seg-btn").forEach(b =>
    b.classList.toggle("active", Number(b.dataset.silence) === settings.silenceMs));
  els.sourceSelect.value = settings.source;
}

// ---------- WebSocket ----------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    wsRetry = 0;
    setStatus("idle", "대기 중");
    send({ type: "hello" });
    // 서버 상태와 로컬 설정 동기화 (서버 재시작/다른 브라우저 대비)
    send({ type: "options", live_translation: settings.livePartialTrans,
           engine: settings.engine, model: settings.model, recording: settings.recording,
           asr_model: settings.asrModel, silence_ms: settings.silenceMs });
  };

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    handleMessage(msg);
  };

  ws.onclose = () => {
    setStatus("error", "서버 연결 끊김 — 재연결 중…");
    running = false;
    updateStartBtn();
    const delay = Math.min(1000 * Math.pow(1.5, wsRetry++), 8000);
    setTimeout(connect, delay);
  };

  ws.onerror = () => ws.close();
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// ---------- 서버 메시지 처리 ----------
function handleMessage(msg) {
  switch (msg.type) {
    case "status": {
      const s = msg.state;
      running = (s === "listening" || s === "loading"); // 로딩 중에도 버튼은 '중지'
      updateStartBtn();
      if (s === "idle") { setStatus("idle", "대기 중"); showLiveIdle(); }
      else if (s === "loading") setStatus("loading", msg.detail || "모델 불러오는 중…");
      else if (s === "listening") setStatus("live", msg.detail || "듣는 중");
      else if (s === "error") setStatus("error", msg.detail || "오류");
      if (msg.sysinfo) els.sysInfoBody.textContent = msg.sysinfo;
      break;
    }
    case "partial": {
      if (!msg.stable && !msg.tentative) {
        // 서버가 세그먼트를 버렸음 (너무 짧음/환각 필터 등) → 라이브 바 정리
        if (!msg.id || msg.id === lastPartialId) clearPartial();
        break;
      }
      lastPartialId = msg.id || null;
      showPartial(msg.stable || "", msg.tentative || "", msg.source);
      // 파티셜이 3초간 갱신되지 않으면(정상이면 0.7초마다 옴) 버려진 것 → 정리
      clearTimeout(partialStaleTimer);
      partialStaleTimer = setTimeout(clearPartial, 3000);
      break;
    }
    case "partial_translation": {
      // 지금 표시 중인 세그먼트의 번역만 (지나간 문장의 늦은 번역 무시)
      if (settings.livePartialTrans && msg.text &&
          msg.id === lastPartialId &&
          !els.livePartialRow.classList.contains("hidden")) {
        els.liveTransText.textContent = msg.text;
        els.liveTransRow.classList.remove("hidden");
      }
      break;
    }
    case "final": {
      addFinalCard(msg);
      clearPartial();
      break;
    }
    case "final_update": {
      // 쪼개진 문장이 직전 카드에 병합됨 — 원문 갱신 + 번역 다시 대기
      const card = cards.get(msg.id);
      if (card) {
        card.querySelector(".utt-src").textContent = msg.text;
        const dst = card.querySelector(".utt-dst");
        dst.textContent = "";
        dst.classList.add("pending");
      }
      clearPartial();
      maybeScroll();
      break;
    }
    case "translation": {
      attachTranslation(msg.id, msg.text);
      break;
    }
    case "error_toast": {
      setStatus("error", msg.detail || "오류가 발생했어요");
      break;
    }
    case "summary": {
      renderSummary(msg.text);
      break;
    }
    case "models": {
      rebuildModelOptions(msg.options || []);
      break;
    }
    case "notice": {
      // 상태 필에 잠깐 표시 후 원래 상태 문구로 복귀
      const text = msg.detail || "";
      setStatus(running ? "live" : "idle", text);
      setTimeout(() => {
        if (els.statusText.textContent === text) {
          setStatus(running ? "live" : "idle", running ? "듣는 중" : "대기 중");
        }
      }, 4000);
      break;
    }
  }
}

// ---------- 상태 표시 ----------
function setStatus(kind, text) {
  els.statusDot.className = "status-dot" +
    (kind === "live" ? " live" : kind === "loading" ? " loading" : kind === "error" ? " error" : "");
  els.statusText.textContent = text;
}

function updateStartBtn() {
  const label = els.startBtn.querySelector("span");
  const icon = els.startBtn.querySelector("svg");
  if (running) {
    els.startBtn.classList.add("recording");
    label.textContent = "중지";
    icon.innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>';
  } else {
    els.startBtn.classList.remove("recording");
    label.textContent = "시작";
    icon.innerHTML = '<polygon points="6 4 20 12 6 20" fill="currentColor" stroke="none"/>';
  }
}

// ---------- 라이브(부분 인식) 표시 ----------
// stable: 연속 인식에서 변하지 않은 확실한 부분(진하게), tentative: 아직 흔들리는 부분(흐리게)
function showPartial(stable, tentative, source) {
  if (!stable && !tentative) return;
  els.liveIdle.classList.add("hidden");
  els.livePartialRow.classList.remove("hidden");
  els.livePartialText.textContent = "";
  if (stable) {
    const s = document.createElement("span");
    s.className = "live-stable";
    s.textContent = stable + " ";
    els.livePartialText.appendChild(s);
  }
  if (tentative) {
    const t = document.createElement("span");
    t.className = "live-tentative";
    t.textContent = tentative;
    els.livePartialText.appendChild(t);
  }
  els.liveChip.textContent = source === "mic" ? "🎙️ 나" : "듣는 중";
}

function clearPartial() {
  clearTimeout(partialStaleTimer);
  lastPartialId = null;
  els.livePartialRow.classList.add("hidden");
  els.liveTransRow.classList.add("hidden");
  els.livePartialText.textContent = "";
  els.liveTransText.textContent = "";
  if (!running) showLiveIdle();
}

function showLiveIdle() {
  els.liveIdle.classList.remove("hidden");
  els.livePartialRow.classList.add("hidden");
  els.liveTransRow.classList.add("hidden");
}

// ---------- 발화 카드 ----------
function fmtTime(ts) {
  const d = ts ? new Date(ts * 1000) : new Date();
  return d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function addFinalCard(msg) {
  if (cards.has(msg.id)) return; // 재연결 시 서버가 기록을 다시 보내므로 중복 방지
  els.emptyState.classList.add("hidden");

  const card = document.createElement("article");
  const isMe = msg.source === "mic";
  card.className = "utt" + (isMe ? " me" : "");
  card.dataset.id = msg.id;

  const meta = document.createElement("div");
  meta.className = "utt-meta";
  const chip = document.createElement("span");
  chip.className = "chip" + (isMe ? " me" : "");
  chip.textContent = isMe ? "🎙️ 나" : "🔊 상대방";
  const time = document.createElement("span");
  time.textContent = fmtTime(msg.time);
  const copyBtn = document.createElement("button");
  copyBtn.className = "utt-copy";
  copyBtn.textContent = "복사";
  copyBtn.onclick = () => {
    const src = card.querySelector(".utt-src")?.textContent || "";
    const dst = card.querySelector(".utt-dst")?.textContent || "";
    navigator.clipboard.writeText(dst + (src ? "\n" + src : ""));
    copyBtn.textContent = "복사됨 ✓";
    setTimeout(() => (copyBtn.textContent = "복사"), 1200);
  };
  meta.append(chip, time, copyBtn);

  const src = document.createElement("div");
  src.className = "utt-src";
  src.textContent = msg.text;

  const dst = document.createElement("div");
  dst.className = "utt-dst pending";

  card.append(meta, src, dst);
  els.feed.appendChild(card);
  cards.set(msg.id, card);

  // 번역이 카드보다 먼저 도착해 있었다면 즉시 부착
  if (pendingTranslations.has(msg.id)) {
    attachTranslation(msg.id, pendingTranslations.get(msg.id));
    pendingTranslations.delete(msg.id);
  }

  // 카드 수 제한 (성능)
  if (cards.size > 500) {
    const firstKey = cards.keys().next().value;
    cards.get(firstKey)?.remove();
    cards.delete(firstKey);
  }

  maybeScroll();
}

function attachTranslation(id, text) {
  const card = cards.get(id);
  if (!card) {
    // 전송 순서 역전 대비: 카드가 아직 없으면 보관해 두었다가 카드 생성 시 부착
    pendingTranslations.set(id, text);
    if (pendingTranslations.size > 50) {
      pendingTranslations.delete(pendingTranslations.keys().next().value);
    }
    return;
  }
  const dst = card.querySelector(".utt-dst");
  dst.classList.remove("pending");
  dst.textContent = text;
  maybeScroll();
}

// ---------- 회의 요약 카드 ----------
function renderSummary(text) {
  if (!text) return;
  els.emptyState.classList.add("hidden");
  let card = document.getElementById("summaryCard");
  if (!card) {
    card = document.createElement("article");
    card.id = "summaryCard";
    card.className = "utt summary-card";
    els.feed.prepend(card);
  }
  card.innerHTML = "";
  const meta = document.createElement("div");
  meta.className = "utt-meta";
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = "📝 회의 요약";
  const copyBtn = document.createElement("button");
  copyBtn.className = "utt-copy";
  copyBtn.textContent = "복사";
  copyBtn.onclick = () => {
    navigator.clipboard.writeText(text);
    copyBtn.textContent = "복사됨 ✓";
    setTimeout(() => (copyBtn.textContent = "복사"), 1200);
  };
  meta.append(chip, copyBtn);
  const body = document.createElement("div");
  body.className = "summary-body";
  // 최소 마크다운: **굵게** 와 줄바꿈만 (XSS 방지 위해 먼저 이스케이프)
  const esc = text.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  body.innerHTML = esc.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
  card.append(meta, body);
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

els.summaryBtn.onclick = () => send({ type: "summarize" });

// ---------- 모델 목록 (서버가 models/ 폴더 스캔 결과 전송) ----------
function rebuildModelOptions(options) {
  if (!options.length) return;
  els.modelSelect.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o.value;
    opt.textContent = o.label;
    els.modelSelect.appendChild(opt);
  }
  if (![...els.modelSelect.options].some(o => o.value === settings.model)) {
    settings.model = "qwen3-4b";
    saveSettings();
  }
  els.modelSelect.value = settings.model;
}

// ---------- 자동 스크롤 ----------
function maybeScroll() {
  if (autoScroll) {
    // 반드시 instant: smooth 애니메이션은 중간 scroll 이벤트가
    // autoScroll 판정(80px 근접)과 싸워 자동 스크롤을 영구히 꺼버린다
    els.feedWrap.scrollTo({ top: els.feedWrap.scrollHeight, behavior: "instant" });
  } else {
    els.scrollDownBtn.classList.remove("hidden");
  }
}

els.feedWrap.addEventListener("scroll", () => {
  const nearBottom = els.feedWrap.scrollHeight - els.feedWrap.scrollTop - els.feedWrap.clientHeight < 80;
  autoScroll = nearBottom;
  if (nearBottom) els.scrollDownBtn.classList.add("hidden");
});

els.scrollDownBtn.onclick = () => {
  autoScroll = true;
  els.feedWrap.scrollTo({ top: els.feedWrap.scrollHeight, behavior: "smooth" });
  els.scrollDownBtn.classList.add("hidden");
};

// ---------- 컨트롤 ----------
els.startBtn.onclick = () => {
  if (running) {
    send({ type: "stop" });
  } else {
    send({ type: "start", source: settings.source, engine: settings.engine, model: settings.model });
    setStatus("loading", "시작하는 중…");
  }
};

els.sourceSelect.onchange = () => {
  settings.source = els.sourceSelect.value;
  saveSettings();
  if (running) send({ type: "start", source: settings.source, engine: settings.engine,
                      model: settings.model }); // 소스 변경 시 재시작
};

els.exportBtn.onclick = async () => {
  try {
    const res = await fetch("/export");
    if (!res.ok) throw new Error();
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    a.download = `회의록_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) {
    setStatus("error", "저장할 내용이 없어요");
    setTimeout(() => setStatus(running ? "live" : "idle", running ? "듣는 중" : "대기 중"), 2000);
  }
};

// ---------- 설정 드로어 ----------
function openDrawer() { els.drawer.classList.remove("hidden"); els.drawerBackdrop.classList.remove("hidden"); }
function closeDrawer() { els.drawer.classList.add("hidden"); els.drawerBackdrop.classList.add("hidden"); }
els.settingsBtn.onclick = openDrawer;
els.drawerClose.onclick = closeDrawer;
els.drawerBackdrop.onclick = closeDrawer;

els.fontSize.oninput = () => {
  settings.fontSize = Number(els.fontSize.value);
  applySettings(); saveSettings();
};
els.showSource.onchange = () => { settings.showSource = els.showSource.checked; applySettings(); saveSettings(); };
els.livePartialTrans.onchange = () => {
  settings.livePartialTrans = els.livePartialTrans.checked;
  saveSettings();
  send({ type: "options", live_translation: settings.livePartialTrans });
  if (!settings.livePartialTrans) els.liveTransRow.classList.add("hidden");
};
els.recordToggle.onchange = () => {
  settings.recording = els.recordToggle.checked;
  saveSettings();
  send({ type: "options", recording: settings.recording });
};
els.themeSeg.onclick = (e) => {
  const btn = e.target.closest(".seg-btn");
  if (!btn) return;
  settings.theme = btn.dataset.themeVal;
  applySettings(); saveSettings();
};
els.engineSelect.onchange = () => {
  settings.engine = els.engineSelect.value;
  applySettings(); saveSettings();
  send({ type: "options", engine: settings.engine });
};
els.modelSelect.onchange = () => {
  settings.model = els.modelSelect.value;
  saveSettings();
  send({ type: "options", model: settings.model });
};
els.asrSelect.onchange = () => {
  settings.asrModel = els.asrSelect.value;
  saveSettings();
  send({ type: "options", asr_model: settings.asrModel });
};
els.silenceSeg.onclick = (e) => {
  const btn = e.target.closest(".seg-btn");
  if (!btn) return;
  settings.silenceMs = Number(btn.dataset.silence);
  applySettings(); saveSettings();
  send({ type: "options", silence_ms: settings.silenceMs });
};
els.clearBtn.onclick = () => {
  if (!confirm("지금까지의 기록을 모두 지울까요?")) return;
  els.feed.querySelectorAll(".utt").forEach(el => el.remove());
  cards.clear();
  els.emptyState.classList.remove("hidden");
  send({ type: "clear" });
  closeDrawer();
};

// ---------- 시작 ----------
loadSettings();
connect();
