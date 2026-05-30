/* app.js — 阿瓦隆前端逻辑：轮询状态 + 渲染 + 真人操作。 */

const PHASE_CN = {
  TEAM: "组队阶段", DISCUSS: "发言/指控", VOTE: "投票阶段", MISSION: "任务执行",
  ASSASSIN: "刺杀阶段", OVER: "游戏结束",
};

// 角色 -> 纹章 symbol id（仅在规则允许展示时使用，绝不泄露隐藏身份）
const ROLE_EMBLEM = {
  Merlin: "em-merlin", Percival: "em-percival", LoyalServant: "em-loyal",
  Assassin: "em-assassin", Morgana: "em-morgana", Oberon: "em-oberon",
};
function svgUse(id) { return `<svg class="emblem"><use href="#${id}"></use></svg>`; }

let state = null;
let selectedTeam = new Set();   // 真人作为队长选人时的暂存
let polling = null;
let setupChoice = null;         // 开局弹窗中暂选的人数

// 各人数（5~10）局的阵营构成（与后端 roles.ROLE_SETUPS 保持一致），仅用于弹窗展示
const SETUP_INFO = {
  5:  "好人3（梅林·派西维尔·忠臣×1） vs 坏人2（刺客·莫甘娜）",
  6:  "好人4（梅林·派西维尔·忠臣×2） vs 坏人2（刺客·莫甘娜）",
  7:  "好人4（梅林·派西维尔·忠臣×2） vs 坏人3（刺客·莫甘娜·奥伯伦）",
  8:  "好人5（梅林·派西维尔·忠臣×3） vs 坏人3（刺客·莫甘娜·莫德雷德）",
  9:  "好人6（梅林·派西维尔·忠臣×4） vs 坏人3（刺客·莫甘娜·莫德雷德）",
  10: "好人6（梅林·派西维尔·忠臣×4） vs 坏人4（刺客·莫甘娜·莫德雷德·奥伯伦）",
};

const $ = (id) => document.getElementById(id);

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2600);
}

async function api(path, method = "GET", body = null) {
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  const res = await fetch(path, opt);
  const data = await res.json();
  if (!res.ok) { toast(data.error || "操作失败"); return null; }
  return data;
}

async function refresh() {
  const data = await api("/api/state");
  if (data) { state = data; render(); }
}

async function act(path, body) {
  const data = await api(path, "POST", body);
  if (data) { state = data; selectedTeam.clear(); render(); }
}

/* ---------------- 渲染 ---------------- */
function render() {
  if (!state) return;
  const s = state;
  const nLabel = s.n ? `${s.n}人局 · ` : "";
  $("phase-badge").textContent =
    nLabel + PHASE_CN[s.phase] + (s.phase !== "OVER" ? ` · 第${s.round + 1}轮` : "");
  if (s.llm) {
    const b = $("llm-badge");
    b.textContent = s.llm.enabled ? `🤖 LLM: ${s.llm.model}` : "🤖 启发式 AI";
    b.style.color = s.llm.enabled ? "var(--ok)" : "var(--muted)";
  }

  renderPlayers();
  renderMissions();
  renderPrivate();
  renderAction();
  renderChat();
  renderVoteHistory();
  renderLog();
}

function renderChat() {
  const wrap = $("chat");
  if (!state.chat || !state.chat.length) {
    wrap.innerHTML = `<div class="hint">本局还没有发言。组队后进入发言/指控阶段。</div>`;
    return;
  }
  wrap.innerHTML = state.chat.map((c) => {
    const mine = c.pid === state.private.id;
    const acc = (c.accuse !== null && c.accuse !== undefined)
      ? `<div class="accuse">⮞ 指控 ${playerName(c.accuse)}</div>` : "";
    return `<div class="bubble ${mine ? "me" : ""}">
      <div class="av">${c.pid + 1}</div>
      <div class="body">
        <div class="who">${c.name}<span class="ctx">第${c.round + 1}轮·第${c.attempt}次</span></div>
        <div class="txt">${escapeHtml(c.text)}</div>${acc}
      </div></div>`;
  }).join("");
  wrap.scrollTop = wrap.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function playerName(pid) {
  return state.players[pid] ? state.players[pid].name : `#${pid}`;
}

function renderPlayers() {
  const ul = $("players");
  ul.innerHTML = "";
  const priv = state.private;
  const pickingTeam = state.waiting === "pick_team";
  const pickingAssassin = state.waiting === "assassinate";

  state.players.forEach((p) => {
    const li = document.createElement("li");
    li.className = "player";
    if (pickingTeam) li.classList.add("selectable");
    if (pickingTeam && selectedTeam.has(p.id)) li.classList.add("selected");
    if (pickingAssassin && p.id !== priv.id) li.classList.add("assassin-pick");

    const tags = [];
    if (p.is_human) tags.push(`<span class="tag human">真人 · 你</span>`);
    if (p.is_leader) tags.push(`<span class="tag leader">队长</span>`);
    // 真人合法可见信息高亮
    if (priv.known_evil.includes(p.id)) tags.push(`<span class="tag evil-known">坏人</span>`);
    if (priv.merlin_candidates.includes(p.id)) tags.push(`<span class="tag merlin-cand">梅林候选</span>`);
    // 当前提议队伍标记
    if (state.proposed_team.includes(p.id) && (state.phase === "VOTE" || state.phase === "MISSION"))
      tags.push(`<span class="tag">⚔ 出征</span>`);
    // 结束后公开角色
    if (p.role) {
      const cls = p.alignment === "good" ? "role-good" : "role-evil";
      tags.push(`<span class="tag ${cls}">${p.role_cn}</span>`);
    }

    // 纹章头像：游戏结束按真实角色显示；进行中默认通用骑士剪影，
    // 并依据真人“合法可知”信息给边框上色（坏人=红、梅林候选=紫）。
    const medCls = ["medallion"];
    let emblemId = "em-figure";
    if (p.role) { emblemId = ROLE_EMBLEM[p.role] || "em-figure"; medCls.push(p.alignment === "good" ? "side-good" : "side-evil"); }
    else {
      if (priv.known_evil.includes(p.id)) medCls.push("mark-evil");
      else if (priv.merlin_candidates.includes(p.id)) medCls.push("mark-cand");
    }
    if (p.is_human) medCls.push("is-human");
    if (p.is_leader) medCls.push("is-leader");
    const crown = p.is_leader ? `<svg class="crown"><use href="#crown"></use></svg>` : "";

    li.innerHTML = `
      <div class="${medCls.join(" ")}">
        ${crown}${svgUse(emblemId)}
        <span class="seat">${p.id + 1}</span>
      </div>
      <div class="pinfo">
        <div class="pname">${p.name}</div>
        <div class="tags">${tags.join("")}</div>
      </div>`;

    if (pickingTeam) li.onclick = () => toggleTeam(p.id);
    if (pickingAssassin && p.id !== priv.id) li.onclick = () => assassinate(p.id);
    ul.appendChild(li);
  });
}

function renderMissions() {
  const wrap = $("missions");
  wrap.innerHTML = "";
  const results = {};
  state.missions.forEach((m) => (results[m.round] = m));
  state.mission_config.forEach((cfg, i) => {
    const div = document.createElement("div");
    div.className = "quest";
    const r = results[i];
    if (r) div.classList.add(r.success ? "ok" : "fail");
    else if (state.phase !== "OVER" && i === state.round) div.classList.add("current");
    const center = r ? (r.success ? "✔" : "✘") : cfg.size;
    div.innerHTML = `
      <div class="quest-shield"><svg class="emblem"><use href="#shield"></use></svg>
        <span class="num">${center}</span></div>
      <div class="cap">第${i + 1}轮</div>
      <div class="sub">${cfg.size}人${cfg.fails_required > 1 ? " · 需2票" : ""}${r ? `<br>${r.fails} 张失败票` : ""}</div>`;
    wrap.appendChild(div);
  });

  $("score").innerHTML =
    `<span class="g">好人成功：${state.score.good} / 3</span>` +
    `<span class="e">坏人破坏：${state.score.evil} / 3</span>`;
  $("vote-track").textContent =
    state.phase === "OVER" ? "" : `本轮连续否决：${state.vote_track} / 5（满 5 坏人获胜）`;
}

function renderPrivate() {
  const p = state.private;
  const side = p.alignment === "good" ? "good" : "evil";
  const sideCn = p.alignment === "good" ? "⚜ 好人阵营" : "☠ 坏人阵营";
  const emblemId = ROLE_EMBLEM[p.role] || "em-figure";
  const known = [];
  p.known_evil.forEach((id) =>
    known.push(`<div class="know-chip evil">${svgUse("em-assassin")}<span>${playerName(id)} · 坏人</span></div>`));
  p.merlin_candidates.forEach((id) =>
    known.push(`<div class="know-chip cand">${svgUse("em-merlin")}<span>${playerName(id)} · 梅林候选</span></div>`));

  $("private").innerHTML = `
    <div class="role-card side-${side}">
      <div class="role-roundel">${svgUse(emblemId)}</div>
      <div class="role-name">${p.role_cn}<small>${p.role}</small></div>
      <span class="role-side ${side}">${sideCn}</span>
      <div class="info">${p.info_text.join("<br>")}</div>
      ${known.length ? `<div class="known">${known.join("")}</div>` : ""}
    </div>`;
}

function renderAction() {
  const body = $("action-body");
  const title = $("action-title");
  body.innerHTML = "";

  if (state.phase === "OVER") { renderResult(body, title); return; }

  switch (state.waiting) {
    case "pick_team": return renderPickTeam(body, title);
    case "speak": return renderSpeak(body, title);
    case "vote": return renderVote(body, title);
    case "mission": return renderMission(body, title);
    case "assassinate": return renderAssassin(body, title);
    default: {
      title.textContent = "等待中…";
      let extra = `当前队长：<b>${playerName(state.leader)}</b>。`;
      if (state.phase === "DISCUSS" && state.current_speaker !== null)
        extra = `正在发言：<b>${playerName(state.current_speaker)}</b>…`;
      body.innerHTML = `
        <div class="waiting-spinner"><span class="dot"></span>
        AI 正在思考 / 结算中。${extra}</div>`;
    }
  }
}

function renderSpeak(body, title) {
  title.textContent = "💬 轮到你发言 / 指控";
  const team = state.proposed_team.map(playerName).join("、");
  const rnd = (state.discuss_round || 0) + 1;
  const maxR = state.max_discuss_rounds || 3;
  body.innerHTML = `
    <div class="hint">第 <b>${rnd}/${maxR}</b> 轮讨论 · 队长 <b>${playerName(state.leader)}</b> 提议队伍：<b>${team}</b>。<br>
      你可以<b>连续多次发言</b>；说完点“结束发言”让给下一位，或直接“提前进入投票”。</div>
    <textarea id="speak-text" class="speak-box" maxlength="200"
      placeholder="发表看法、辩护或指控……可多次发送"></textarea>
    <div class="btn-row">
      <button id="speak-send" class="btn btn-primary">发送发言</button>
      <button id="speak-done" class="btn btn-ghost">结束发言（让过）</button>
      <button id="speak-tovote" class="btn btn-ghost">提前进入投票</button>
    </div>`;
  $("speak-send").onclick = () => {
    const t = $("speak-text").value.trim();
    if (!t) { toast("发言不能为空（不想发言请点“结束发言”）"); return; }
    act("/api/speak", { text: t });
  };
  $("speak-done").onclick = () => act("/api/end_speak", {});
  $("speak-tovote").onclick = () => act("/api/end_discussion", {});
}

function renderPickTeam(body, title) {
  title.textContent = "👑 你是队长 · 选择任务队员";
  const need = state.team_size;
  body.innerHTML = `
    <div class="hint">请在左侧玩家列表中点击选择 <b>${need}</b> 名队员
      （已选 <b id="sel-count">${selectedTeam.size}</b> / ${need}）。</div>
    <div class="btn-row">
      <button id="confirm-team" class="btn btn-primary" ${selectedTeam.size === need ? "" : "disabled"}>确认队伍</button>
    </div>`;
  $("confirm-team").onclick = () => {
    if (selectedTeam.size !== need) return;
    act("/api/team", { team: [...selectedTeam] });
  };
}

function toggleTeam(pid) {
  if (selectedTeam.has(pid)) selectedTeam.delete(pid);
  else {
    if (selectedTeam.size >= state.team_size) { toast(`最多选择 ${state.team_size} 人`); return; }
    selectedTeam.add(pid);
  }
  render();
}

function renderVote(body, title) {
  title.textContent = "🗳️ 对当前队伍投票";
  const team = state.proposed_team.map(playerName).join("、");
  body.innerHTML = `
    <div class="hint">队长 <b>${playerName(state.leader)}</b> 提议队伍：<b>${team}</b>。<br>是否同意该队伍出任务？</div>
    <div class="btn-row">
      <button id="v-yes" class="btn btn-ok">👍 赞成</button>
      <button id="v-no" class="btn btn-bad">👎 反对</button>
    </div>`;
  const lock = () => disableRow(body, "已提交，等待结算…");
  $("v-yes").onclick = () => { lock(); act("/api/vote", { approve: true }); };
  $("v-no").onclick = () => { lock(); act("/api/vote", { approve: false }); };
}

// 点击后立即禁用该区域按钮并提示，给出即时反馈、避免重复提交
function disableRow(body, msg) {
  body.querySelectorAll("button").forEach((b) => (b.disabled = true));
  const row = body.querySelector(".btn-row");
  if (row) row.insertAdjacentHTML("afterend", `<div class="hint">⏳ ${msg}</div>`);
}

function renderMission(body, title) {
  title.textContent = "⚔️ 执行任务";
  const isGood = state.private.alignment === "good";
  body.innerHTML = `
    <div class="hint">你是任务队员。${isGood ? "好人只能提交【成功】。" : "你是坏人，可选择成功或失败。"}</div>
    <div class="btn-row">
      <button id="m-success" class="btn btn-ok">✅ 成功</button>
      <button id="m-fail" class="btn btn-bad" ${isGood ? "disabled" : ""}>❌ 失败</button>
    </div>`;
  const lock = () => disableRow(body, "已出牌，等待结算…");
  $("m-success").onclick = () => { lock(); act("/api/mission", { action: "success" }); };
  if (!isGood) $("m-fail").onclick = () => { lock(); act("/api/mission", { action: "fail" }); };
}

function renderAssassin(body, title) {
  title.textContent = "🗡️ 刺杀阶段 · 你是刺客";
  body.innerHTML = `
    <div class="hint">好人已完成 3 个任务。请在左侧点击你认为是 <b>梅林</b> 的玩家进行刺杀。<br>
    刺中梅林则坏人获胜，刺错则好人获胜。</div>`;
}

function assassinate(pid) {
  if (!confirm(`确定刺杀 ${playerName(pid)}？`)) return;
  act("/api/assassinate", { target: pid });
}

function renderResult(body, title) {
  title.textContent = "终局";
  const good = state.winner === "good";
  let html = `
    <div class="result ${state.winner}">
      <div class="result-crest">${svgUse(good ? "em-percival" : "em-assassin")}</div>
      <div class="result-title">${good ? "正义之光 · 好人获胜" : "黑暗降临 · 坏人获胜"}</div>
    </div>`;
  if (state.assassin_target !== null && state.assassin_target !== undefined) {
    html += `<div class="hint" style="text-align:center">刺客刺向了 <b>${playerName(state.assassin_target)}</b>
             （${state.assassinated_role}）。</div>`;
  }
  html += `<div class="btn-row" style="justify-content:center"><button id="ng2" class="btn btn-primary">⚔ 再战一局</button></div>`;
  body.innerHTML = html;
  $("ng2").onclick = () => openSetup(false);
}

function renderVoteHistory() {
  const wrap = $("vote-history");
  wrap.innerHTML = "";
  if (!state.proposals.length) { wrap.innerHTML = `<div class="hint">暂无投票记录。</div>`; return; }
  [...state.proposals].reverse().forEach((p) => {
    const div = document.createElement("div");
    div.className = "vrow";
    const team = p.team.map(playerName).join("、");
    const votes = state.players.map((pl) => {
      const v = p.votes[String(pl.id)];
      return `<span class="vtag ${v ? "va" : "vr"}">${pl.name.replace("（真人）", "")}${v ? "✓" : "✗"}</span>`;
    }).join("");
    div.innerHTML = `
      <div class="vtitle">第${p.round + 1}轮·第${p.attempt}次 · 队长 ${playerName(p.leader)} · 队伍[${team}]
        · <span class="${p.approved ? "approved" : "rejected"}">${p.approved ? "通过" : "否决"}</span></div>
      <div>${votes}</div>`;
    wrap.appendChild(div);
  });
}

function renderLog() {
  const wrap = $("log");
  wrap.innerHTML = state.log.map((l) => `<div class="lrow">${l}</div>`).join("");
  wrap.scrollTop = wrap.scrollHeight;
}

/* ---------------- 控制 ---------------- */
// 开局：弹出人数选择遮罩。firstTime=true 表示首屏尚无对局，隐藏"取消"（必须先选人数）。
function openSetup(firstTime = false) {
  $("setup-cancel").hidden = !!firstTime;
  setupChoice = (state && state.n) || 7;
  const wrap = $("setup-counts");
  wrap.innerHTML = [5, 6, 7, 8, 9, 10].map((n) =>
    `<button class="btn setup-num${n === setupChoice ? " active" : ""}" data-n="${n}">${n} 人</button>`
  ).join("");
  wrap.querySelectorAll(".setup-num").forEach((b) => {
    b.onclick = () => {
      setupChoice = parseInt(b.dataset.n, 10);
      wrap.querySelectorAll(".setup-num").forEach((x) =>
        x.classList.toggle("active", parseInt(x.dataset.n, 10) === setupChoice));
      $("setup-detail").textContent = SETUP_INFO[setupChoice] || "";
      $("setup-start").disabled = false;
    };
  });
  $("setup-detail").textContent = SETUP_INFO[setupChoice] || "";
  $("setup-start").disabled = false;
  $("setup-overlay").hidden = false;
}

function closeSetup() { $("setup-overlay").hidden = true; }

async function startGame(n) {
  const data = await api("/api/new_game", "POST", { n_players: n });
  if (data) { state = data; selectedTeam.clear(); closeSetup(); render(); startPolling(); }
}

$("new-game").onclick = () => openSetup(false);
$("setup-cancel").onclick = closeSetup;
$("setup-start").onclick = () => startGame(setupChoice || 7);

// 轮询：等待 AI / 结算时自动刷新；轮到真人操作时不打断
function startPolling() {
  if (polling) return;                       // 幂等：仅在首局开始后启动一次
  polling = setInterval(() => {
    if (!state) return;                      // 尚未开始对局（仍在选人数）时不轮询
    const humanTurn = ["pick_team", "speak", "vote", "mission", "assassinate"].includes(state.waiting);
    if (!humanTurn && state.phase !== "OVER") refresh();
  }, 1500);
}

// 首屏：直接弹出人数选择，由玩家选定后才真正开局（不预先拉取默认 7 人局，省去无谓的 AI 推进）
openSetup(true);
