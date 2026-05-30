"""
memory.py
=========
Agent 的**分层记忆系统**。把原先单一的自由文本笔记(notes)升级为三层结构 + 结构化信念，
对标 MemGPT / Generative Agents 的"内存分层"，并引入带证据与加锁的**信念修正**机制。

三层记忆（生命周期由短到长）：
  - working（工作记忆）：当前这次决策的一句私有心证，用完即弃，不跨回合。
  - episodic（情节记忆）：append-only 的心证事件流——**只增不改**，从根上避免"全量覆盖丢信息"。
  - semantic（语义记忆）：对每名其他玩家的**结构化信念**——坏人概率 evil、梅林概率 merlin、
    一句证据 note。新证据到来时做**增量修正(revise)**而非整体覆盖；由角色合法认知播种，
    "确知坏人"被加锁(locked)，模型无法把确证事实改坏，抑制幻觉累积。

设计要点：
  - 严格私有：记忆只活在服务端的 Agent 实例上，绝不下发前端、绝不共享给别的 Agent。
  - 概率而非标签：信念用 0~1 概率表达，天然可量化、可可视化（见 snapshot()，可供调试/前端热力图）。
  - 预算可控：render() 只渲染"有判断"的玩家与最近若干条心证，控制注入上下文的 token。
"""

from dataclasses import dataclass, field
from typing import Any


def _clamp01(x, default):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return 0.0 if v < 0 else 1.0 if v > 1 else v


@dataclass
class Belief:
    """对单名玩家的结构化信念。evil/merlin 为 0~1 概率；note 为最近一条证据。"""
    evil: float = 0.5          # P(他是坏人)；0.5=中立未知
    merlin: float = 0.0        # P(他是梅林)
    note: str = ""             # 最近一条支撑证据（≤40字）
    updates: int = 0           # 被修正过几次（体现证据积累）
    round: int = -1            # 最后一次修正发生的轮次
    locked: bool = False       # 由合法认知确证，禁止模型改写（如已知坏人）


class AgentMemory:
    def __init__(self, pid, names, known_evil=(), merlin_candidates=()):
        self.pid = pid
        self.names = names
        # ---- semantic：每名其他玩家一条结构化信念，由合法认知播种 ----
        self.beliefs: dict[int, Belief] = {}
        for p in range(len(names)):
            if p == pid:
                continue
            self.beliefs[p] = Belief()
        for p in known_evil:                       # 确知坏人：evil=1，加锁
            if p in self.beliefs:
                self.beliefs[p] = Belief(evil=1.0, note="确知坏人", locked=True)
        for p in merlin_candidates:                # 梅林候选：先验各 50% 是真梅林
            if p in self.beliefs and not self.beliefs[p].locked:
                self.beliefs[p].merlin = 0.5
                self.beliefs[p].note = "梅林候选"
        # ---- episodic：append-only 心证流 ----
        self.episodes: list[dict[str, Any]] = []
        # ---- working：本次决策的临时心证 ----
        self.working: str = ""

    def _name(self, p):
        return f"{p + 1}号({self.names[p]})"

    # ---------------- 写入：信念修正 + 情节追加 ----------------
    def update(self, belief_updates, memo, rnd):
        """一次决策后回写记忆：修正结构化信念(增量) + 追加情节心证。"""
        self.revise(belief_updates, rnd)
        memo = (memo or "").strip()[:40]
        self.working = memo                        # 工作记忆：仅保留本次
        if memo:                                   # 情节记忆：append-only，不覆盖历史
            self.episodes.append({"round": rnd, "text": memo})
            if len(self.episodes) > 40:            # 兜底上限，防无界增长
                self.episodes = self.episodes[-40:]

    def revise(self, belief_updates, rnd):
        """把模型给出的 belief_updates 增量并入既有信念；加锁项与非法项被忽略。"""
        if not isinstance(belief_updates, list):
            return
        for u in belief_updates:
            if not isinstance(u, dict):
                continue
            seat = u.get("seat")
            try:
                p = int(seat) - 1
            except (TypeError, ValueError):
                continue
            b = self.beliefs.get(p)
            if b is None or p == self.pid or b.locked:   # 自己/未知座位/确证事实：不可改
                continue
            if "evil" in u:
                b.evil = _clamp01(u.get("evil"), b.evil)
            if "merlin" in u:
                b.merlin = _clamp01(u.get("merlin"), b.merlin)
            note = str(u.get("note", "")).strip()[:40]
            if note:
                b.note = note
            b.updates += 1
            b.round = rnd

    # ---------------- 读取：渲染进决策上下文 ----------------
    def render(self):
        """把三层记忆组织成注入 LLM 的文本块。只渲染'有判断'的玩家与近期心证以控预算。"""
        lines = ["【结构化信念档案】(我的私有判断，概率0~1：坏人越高越可疑，梅林越高越像梅林)"]
        shown = 0
        for p, b in sorted(self.beliefs.items()):
            # 跳过纯默认(中立、无梅林倾向、无证据)的玩家，减少噪声
            if abs(b.evil - 0.5) < 1e-9 and b.merlin == 0.0 and not b.note:
                continue
            tag = "🔒" if b.locked else ""
            ev = f"⟨{b.note}⟩" if b.note else ""
            lines.append(f"  {self._name(p)}{tag} 坏人{b.evil:.2f} 梅林{b.merlin:.2f} {ev}")
            shown += 1
        if shown == 0:
            lines.append("  （尚在观察，暂无明确判断）")
        if self.episodes:
            lines.append("【情节记忆·近期心证】")
            for e in self.episodes[-6:]:
                lines.append(f"  R{e['round'] + 1} {e['text']}")
        return "\n".join(lines)

    # ---------------- 调试/前端：结构化快照（可驱动嫌疑度热力图）----------------
    def snapshot(self):
        """返回可序列化的信念快照，供调试或前端可视化（不含任何越权信息）。"""
        return {
            "pid": self.pid,
            "beliefs": {p: {"evil": round(b.evil, 3), "merlin": round(b.merlin, 3),
                            "note": b.note, "updates": b.updates, "locked": b.locked}
                        for p, b in self.beliefs.items()},
            "episodes": list(self.episodes[-8:]),
            "working": self.working,
        }
