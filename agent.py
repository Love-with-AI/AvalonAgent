"""
agent.py
========
阿瓦隆智能体（Agent）。本项目的核心技术栈：每名 AI 玩家 = 一个自治 Agent，
其"感知 → 推理 → 行动"的决策闭环用 **LangGraph** 建模为一张状态图（StateGraph）。

每个 Agent 具备：
  - 身份与**合法认知**（仅自己角色规则允许知道的隐藏信息；严格信息隔离）。
  - **分层记忆**（memory.py）：working/episodic/semantic 三层 + 结构化信念；每次决策后由模型做
    **增量信念修正**（而非整体覆盖），并向情节记忆追加心证，形成可积累、抗遗忘的连续心智。
  - **感知**：把"公开信息（含全部发言转录、投票/任务历史）+ 自己的合法私有信息"渲染成上下文。
  - **推理 + 行动**：调用 LLM（经 llm.py / LangChain ChatOpenAI），以"工具(JSON Schema)"形式产出
    结构化决策（含私有 reasoning）。
  - **优雅降级**：未配置 LLM（无 Key/未装库/调用失败/输出非法）时，自动回退到 ai.py 的启发式策略。

LangGraph 编排：每个 Agent 持有一张编译后的状态图，三个节点串成决策管线——
    perceive（构造 system/user/tool 上下文） → reason（调 LLM 取结构化 JSON） → act（校验/回退/落库）
"好人执行任务恒为成功"等无需推理的分支，由 perceive 直接短路到 END。

决策接口与启发式 ai.py 对齐：decide_team / decide_vote / decide_mission / decide_speech / decide_assassination。
对外只暴露"行动结果"；Agent 的私有 reasoning 仅留在服务端（self.last_reasoning），绝不下发，避免泄密。

座位约定：对 LLM 一律用 1-based「N号」表示玩家（pid+1），更直观、更少出错；内部再换算回 pid。
"""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import ai
import llm
from memory import AgentMemory
from roles import ROLE_CN


# ---- 角色策略说明（写进 system 提示，帮助 LLM 进入角色）----
ROLE_BRIEF = {
    "Merlin": "你是【梅林】(好人)。你看得见坏人（**莫德雷德除外**，若他在场你看不到他），但**绝不能暴露你看得见**"
              "——一旦坏人察觉你是梅林，终局刺客会刺杀你而坏人获胜。要用'感觉/推理'的口吻含蓄地引导好人，避免精准点名所有坏人。",
    "Percival": "你是【派西维尔】(好人)。你看到两名'梅林候选'，其一是真梅林、其一是莫甘娜伪装。"
                "要保护真梅林、识别并提防莫甘娜，必要时可假装自己是梅林替身吸引刺客火力。",
    "LoyalServant": "你是【忠臣】(好人)，没有特殊信息。靠投票、任务结果与发言推理找出坏人，并配合好人。",
    "Assassin": "你是【刺客】(坏人)。与莫甘娜互知(不含奥伯伦)。要伪装好人、破坏任务、误导好人；"
                "全程记下谁最像梅林——好人三胜后由你刺杀，刺中梅林则坏人翻盘获胜。"
                "**与队友的配合只能藏在公开动作里（你们之间没有私聊）**：任务里若已知队友同在队中，"
                "通常只需一人出失败牌，免得失败票数直接暴露坏人人数；投票时可暗助队友进队，"
                "但别和队友步调太一致而被看穿。每个配合动作都要权衡'推进胜利'与'暴露身份'。",
    "Morgana": "你是【莫甘娜】(坏人)。与刺客互知(不含奥伯伦)。你在派西维尔眼中是'梅林候选'，"
               "应**伪装成梅林**、释放假信息、推动任务失败。"
               "**护队友也只能靠公开发言/投票（你们之间没有私聊）**：可替队友辩护、把火力引向好人，"
               "或主动吸引刺杀火力替真队友打掩护；但替队友洗白要自然，过度护短反而暴露你们的关系。",
    "Oberon": "你是【奥伯伦】(坏人)，但你不认识其他坏人、他们也不认识你，更无从私聊。"
              "只能从任务失败票数、投票与发言中**推断**谁可能是队友：出失败牌前先掂量本队是否可能已有其他坏人，"
              "免得双雷把坏人人数暴露。独立行动、伪装好人、伺机破坏任务。",
    "Mordred": "你是【莫德雷德】(坏人)。与刺客/莫甘娜互知(不含奥伯伦)。你的关键优势：**梅林看不见你**，"
               "所以你可以更大胆地伪装成好人、甚至假装自己'看得见坏人'来争取信任、误导好人。"
               "护队友也只能靠公开发言/投票（你们之间没有私聊）；破坏任务、引导好人投错队伍，"
               "但配合队友时仍要权衡'推进胜利'与'暴露身份'。",
}


def _seat(pid):
    return pid + 1


# ---- 记忆写入字段：所有决策工具共用的结构化记忆 schema（方案1分层 + 方案3结构化信念）----
# belief_updates 让模型做"增量信念修正"，memo 追加进情节记忆；二者都可空。
_MEM_SCHEMA = {
    "belief_updates": {
        "type": "array", "items": {"type": "object"},
        "description": ("（可空数组）本回合你想更新判断的玩家。每个元素形如 "
                        '{"seat":3,"evil":0.8,"merlin":0.1,"note":"理由≤15字"}：'
                        "seat=座位号(从1起)，evil=他是坏人的概率0~1，merlin=他是梅林的概率0~1，"
                        "note=简短依据。只列有新判断的玩家，没有就给空数组[]。"),
    },
    "memo": {
        "type": "string",
        "description": "（可空）本回合一句话私有心证，≤25字，会写入你的情节记忆供后续回合参考。",
    },
}


# ---------------------------------------------------------------------------
# LangGraph 状态：一次决策在三个节点间流转所携带的全部信息
# ---------------------------------------------------------------------------
class DecisionState(TypedDict, total=False):
    decision: str            # team / vote / mission / speech / assassination
    pub: dict                # 公开信息
    team: Optional[list]     # 当前被提名队伍（vote/mission/speech 用）
    discuss_round: int       # 发言阶段的轮次（0-based）；>=1 时允许选择沉默
    # perceive 产出
    system: str
    user: str
    tool: dict
    skip_llm: bool           # 无需推理的短路分支（如好人执行任务恒成功）
    # reason 产出
    raw: Optional[dict]      # 解析后的 LLM 结构化输出
    # act 产出
    result: Any              # 最终行动结果（pid 列表 / bool / 字符串 / (文本,指控) / pid）
    reasoning: str           # 本次私有推理（仅服务端）
    belief_updates: list     # 本次的结构化信念修正（[]=无修正）
    memo: str                # 本次的情节心证（""=不追加）
    used_fallback: bool      # 是否走了启发式回退


class AvalonAgent:
    def __init__(self, pid, know, names, rules_text):
        self.pid = pid
        self.know = know                       # 合法认知（含 role 枚举、known_evil、merlin_candidates）
        self.names = names                     # 公开名字 [name0..name_{n-1}]
        self.n = len(names)                    # 本局总人数
        self.rules = rules_text                # 本局规则提示（随人数/配置而变）
        self.role = know["role"]
        self.role_cn = ROLE_CN[self.role]
        self.alignment = know["alignment"]
        self.memory = AgentMemory(             # 分层记忆 + 结构化信念（由合法认知播种）
            pid, names, know["known_evil"], know["merlin_candidates"])
        self.last_reasoning = ""               # 最近一次私有推理（仅服务端）
        self._graph = self._build_graph()      # 编译后的 LangGraph 决策图

    # ---------------- 感知：渲染上下文 ----------------
    def _who(self, pid):
        return f"{_seat(pid)}号({self.names[pid]})"

    def _identity_block(self):
        lines = [f"你是 {self._who(self.pid)}，角色：{self.role_cn}（{self.alignment}阵营）。",
                 ROLE_BRIEF.get(str(self.role), "")]
        if self.know["known_evil"]:
            ke = "、".join(self._who(p) for p in sorted(self.know["known_evil"]))
            lines.append(f"你**确知的坏人**：{ke}（这是你的隐藏信息，切勿直白说出）。")
        if self.know["merlin_candidates"]:
            mc = "、".join(self._who(p) for p in sorted(self.know["merlin_candidates"]))
            lines.append(f"你看到的【梅林候选】两人：{mc}（其一真梅林、其一莫甘娜）。")
        return "\n".join(x for x in lines if x)

    def _public_block(self, pub, team=None):
        L = []
        roster = "、".join(self._who(p) for p in pub["player_ids"])
        L.append(f"玩家：{roster}。当前队长：{self._who(pub['leader'])}。")
        L.append(f"第{pub['round']+1}轮任务：需 {pub['team_size']} 人，"
                 f"{'需2张失败票' if pub['fails_required']>1 else '1张失败票即失败'}。"
                 f"本轮已连续被否 {pub['vote_track']} 次（满5坏人胜）。")
        if pub["missions"]:
            ms = "；".join(f"第{m['round']+1}轮[{'成功' if m['success'] else '失败'}"
                          f",{m['fails']}张失败票,队伍{'/'.join(str(_seat(t)) for t in m['team'])}]"
                          for m in pub["missions"])
            L.append("任务结果：" + ms)
        if pub["proposals"]:
            ps = []
            for p in pub["proposals"][-8:]:
                yes = [str(_seat(int(k))) for k, v in p["votes"].items() if v]
                no = [str(_seat(int(k))) for k, v in p["votes"].items() if not v]
                ps.append(f"第{p['round']+1}轮第{p['attempt']}次:队长{_seat(p['leader'])}"
                          f"提名[{'/'.join(str(_seat(t)) for t in p['team'])}],"
                          f"赞成{'/'.join(yes) or '无'},反对{'/'.join(no) or '无'}→"
                          f"{'通过' if p['approved'] else '否决'}")
            L.append("近期组队/投票：" + "；".join(ps))
        if pub.get("chat"):
            cs = []
            for c in pub["chat"][-16:]:
                acc = f"[指控{_seat(c['accuse'])}号]" if c.get("accuse") is not None else ""
                cs.append(f"{_seat(c['pid'])}号({c['name']}){acc}:“{c['text']}”")
            L.append("发言记录：\n" + "\n".join(cs))
        if team is not None:
            L.append("当前被提名的队伍：" + "、".join(self._who(t) for t in team) + "。")
        return "\n".join(L)

    def _system(self):
        return (self.rules + "\n\n你的身份与机密信息：\n" + self._identity_block() +
                "\n\n硬性要求：1)只能使用本提示提供的信息，不得编造你并不知道的身份事实；"
                "2)发言用简体中文、1~2句、口语化、符合你的角色与阵营利益；"
                "3)始终为你的阵营争取胜利；4)务必通过给定的工具返回结构化结果；"
                "5)你与任何玩家之间都没有私下沟通渠道——与队友的一切配合只能通过公开的提名/投票/发言/任务牌"
                "来暗中体现，且需自行权衡'配合收益'与'暴露身份'的风险。")

    def _ctx(self, pub, team=None, ask=""):
        return (self._public_block(pub, team) +
                "\n\n你的私有记忆（仅你可见）：\n" + self.memory.render() +
                f"\n\n{ask}")

    def _to_pid(self, seat):
        try:
            s = int(seat)
        except (TypeError, ValueError):
            return None
        return s - 1 if 1 <= s <= self.n else None

    # ---------------- 各决策的工具(JSON Schema) + 提问语 ----------------
    def _spec(self, decision, pub, discuss_round=0):
        """返回 (tool, ask)：该决策对应的工具 schema 与给模型的提问。"""
        if decision == "team":
            size = pub["team_size"]
            return ({
                "name": "propose_team", "description": "作为队长提名本轮任务队伍。",
                "input_schema": {"type": "object", "properties": {
                    "reasoning": {"type": "string", "description": "你的私下推理（不会公开）"},
                    "team": {"type": "array", "items": {"type": "integer"},
                             "description": f"恰好{size}个玩家座位号(1-{self.n})，可含自己"},
                    **_MEM_SCHEMA},
                    "required": ["reasoning", "team"]}},
                f"轮到你当队长，请提名 {size} 名队员上车。")
        if decision == "vote":
            return ({
                "name": "cast_vote", "description": "对当前被提名队伍投赞成或反对。",
                "input_schema": {"type": "object", "properties": {
                    "reasoning": {"type": "string"},
                    "approve": {"type": "boolean", "description": "true=赞成该队伍出任务"},
                    **_MEM_SCHEMA},
                    "required": ["reasoning", "approve"]}},
                "请对该队伍投票（注意：同一轮连续5次否决坏人直接获胜）。")
        if decision == "mission":
            return ({
                "name": "play_mission", "description": "在任务中秘密出成功或失败牌。",
                "input_schema": {"type": "object", "properties": {
                    "reasoning": {"type": "string"},
                    "action": {"type": "string", "enum": ["success", "fail"]},
                    **_MEM_SCHEMA},
                    "required": ["reasoning", "action"]}},
                "你在任务队伍中。决定出'成功'还是'失败'（出失败会推进坏人，但可能暴露身份）。")
        if decision == "speech":
            later = discuss_round >= 1
            props = {
                "reasoning": {"type": "string", "description": "私下推理（不公开）"},
                "message": {"type": "string", "description": "公开发言，简体中文1~2句；本轮无话可留空"},
                "accuse": {"type": "integer", "description": f"指控的玩家座位号(1-{self.n})，无指控填0"},
                **_MEM_SCHEMA}
            required = ["reasoning"]
            if later:
                props["speak"] = {"type": "boolean",
                    "description": "本轮是否发言。现已是第2轮及以后讨论，若无新观点/无需回应，可填false保持沉默。"}
            else:
                required.append("message")
            ask = ("轮到你发言。可表态、辩护、引导或指控，为本阵营争取选票。" if not later else
                   f"这是第{discuss_round + 1}轮讨论，你已听过此前所有发言。"
                   "有新观点/要回应/要指控就发言(speak=true 并给 message)，否则选择沉默(speak=false)，不必为说而说。")
            return ({"name": "make_statement",
                     "description": "在发言阶段公开发言，可选指控某人；第2轮起可选择沉默。",
                     "input_schema": {"type": "object", "properties": props, "required": required}},
                    ask)
        if decision == "assassination":
            return ({
                "name": "assassinate", "description": "好人已三胜，指认并刺杀你认为的梅林。",
                "input_schema": {"type": "object", "properties": {
                    "reasoning": {"type": "string", "description": "你判断谁是梅林的依据"},
                    "target": {"type": "integer", "description": f"刺杀目标座位号(1-{self.n})"},
                    **_MEM_SCHEMA},
                    "required": ["reasoning", "target"]}},
                "好人完成3个任务。复盘全程发言/投票，指认最可能是梅林的玩家并刺杀。")
        raise ValueError(f"未知决策类型: {decision}")

    # ---------------- LangGraph 节点：感知 / 推理 / 行动 ----------------
    def _node_perceive(self, state: DecisionState) -> dict:
        decision, pub, team = state["decision"], state["pub"], state.get("team")
        # 好人执行任务恒为成功：无需推理，直接短路（与原实现一致，不消耗 LLM）
        if decision == "mission" and self.alignment == "good":
            return {"skip_llm": True, "result": "success", "belief_updates": [],
                    "memo": "", "reasoning": "", "used_fallback": False}
        tool, ask = self._spec(decision, pub, state.get("discuss_round", 0))
        return {"skip_llm": False, "system": self._system(),
                "user": self._ctx(pub, team, ask), "tool": tool}

    def _node_reason(self, state: DecisionState) -> dict:
        if not llm.is_available():
            return {"raw": None}
        return {"raw": llm.decide(state["system"], state["user"], state["tool"])}

    def _node_act(self, state: DecisionState) -> dict:
        decision, pub, team = state["decision"], state["pub"], state.get("team")
        discuss_round = state.get("discuss_round", 0)
        raw = state.get("raw")
        # 先取出私有推理与记忆字段（即便后续动作字段非法也保留，与原 _commit 行为一致）
        reasoning, belief_updates, memo = "", [], ""
        if raw:
            reasoning = str(raw.get("reasoning", ""))[:1000]
            bu = raw.get("belief_updates")
            if isinstance(bu, list):
                belief_updates = bu
            m = raw.get("memo")
            if isinstance(m, str):
                memo = m
        result, fellback = self._extract_action(decision, pub, team, raw, discuss_round)
        return {"result": result, "reasoning": reasoning, "belief_updates": belief_updates,
                "memo": memo, "used_fallback": fellback}

    def _extract_action(self, decision, pub, team, raw, discuss_round=0):
        """校验 LLM 结构化输出 -> 行动结果；非法/缺失则回退 ai.py 启发式。返回 (result, used_fallback)。"""
        if decision == "team":
            if raw:
                pids = [self._to_pid(s) for s in raw.get("team", [])]
                pids = [p for p in dict.fromkeys(pids) if p is not None]
                if len(pids) == pub["team_size"]:
                    return pids, False
            return ai.decide_team(self.pid, self.know, pub), True

        if decision == "vote":
            if raw and isinstance(raw.get("approve"), bool):
                return raw["approve"], False
            return ai.decide_vote(self.pid, self.know, pub, team), True

        if decision == "mission":
            if raw and raw.get("action") in ("success", "fail"):
                return raw["action"], False
            return ai.decide_mission(self.pid, self.know, pub, team), True

        if decision == "speech":
            if raw:
                if discuss_round >= 1 and raw.get("speak") is False:
                    return None, False                # 第2轮起：主动选择沉默(pass)
                msg = raw.get("message")
                if isinstance(msg, str) and msg.strip():
                    accuse = self._to_pid(raw.get("accuse", 0))
                    if accuse == self.pid:
                        accuse = None
                    return (msg.strip()[:200], accuse), False
            if discuss_round >= 1:
                return None, False                    # 后续轮无有效发言 -> 沉默（启发式亦由此快速收敛）
            return ai.decide_speech(self.pid, self.know, pub, team), True   # 首轮必须发言

        if decision == "assassination":
            if raw:
                t = self._to_pid(raw.get("target"))
                if t is not None and t != self.pid and t not in self.know["known_evil"]:
                    return t, False
            return ai.decide_assassination(self.pid, self.know, pub), True

        raise ValueError(f"未知决策类型: {decision}")

    def _build_graph(self):
        g = StateGraph(DecisionState)
        g.add_node("perceive", self._node_perceive)
        g.add_node("reason", self._node_reason)
        g.add_node("act", self._node_act)
        g.add_edge(START, "perceive")
        # 短路分支（如好人执行任务）直接结束；否则进入 推理 → 行动
        g.add_conditional_edges("perceive",
                                lambda s: "end" if s.get("skip_llm") else "reason",
                                {"reason": "reason", "end": END})
        g.add_edge("reason", "act")
        g.add_edge("act", END)
        return g.compile()

    # ---------------- 决策入口：运行 LangGraph，回写记忆 ----------------
    def _run(self, decision, pub, team=None, discuss_round=0):
        out = self._graph.invoke({"decision": decision, "pub": pub, "team": team,
                                  "discuss_round": discuss_round})
        # 回写私有推理与分层记忆（仅服务端）：增量信念修正 + 追加情节心证
        if out.get("reasoning"):
            self.last_reasoning = out["reasoning"]
        self.memory.update(out.get("belief_updates") or [],
                           out.get("memo") or "", pub.get("round", 0))
        return out["result"]

    def decide_team(self, pub):
        return self._run("team", pub)

    def decide_vote(self, pub, team):
        return self._run("vote", pub, team)

    def decide_mission(self, pub, team):
        return self._run("mission", pub, team)

    def decide_speech(self, pub, team, discuss_round=0):
        return self._run("speech", pub, team, discuss_round)

    def decide_assassination(self, pub):
        return self._run("assassination", pub)
