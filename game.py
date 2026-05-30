"""
game.py
=======
集中式游戏状态管理 + 规则引擎 + 流程状态机（5~10 人局阿瓦隆）。

阶段 (phase):
  TEAM     队长选人
  VOTE     全体对队伍投票
  MISSION  入选成员执行任务
  ASSASSIN 好人三胜后，刺客刺杀
  OVER     结束

核心循环 advance()：自动驱动所有 AI 行动，直到"需要真人输入"或"游戏结束"为止。
真人每次操作后调用 advance()，即可推进到下一次需要真人的节点。
"""

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import ai
import llm
from agent import AvalonAgent
from roles import (
    ROLE_SETUPS, Role, ROLE_CN, alignment_of, build_knowledge, describe_setup,
)

# 各人数（5~10）局的任务配置：每轮出任务人数 / 该轮判定失败所需的失败票数（标准阿瓦隆规则）。
MISSION_CONFIG = {
    5:  {"sizes": [2, 3, 2, 3, 3], "fails": [1, 1, 1, 1, 1]},
    6:  {"sizes": [2, 3, 4, 3, 4], "fails": [1, 1, 1, 1, 1]},
    7:  {"sizes": [2, 3, 3, 4, 4], "fails": [1, 1, 1, 2, 1]},
    8:  {"sizes": [3, 4, 4, 5, 5], "fails": [1, 1, 1, 2, 1]},
    9:  {"sizes": [3, 4, 4, 5, 5], "fails": [1, 1, 1, 2, 1]},
    10: {"sizes": [3, 4, 4, 5, 5], "fails": [1, 1, 1, 2, 1]},
}
MIN_PLAYERS, MAX_PLAYERS = 5, 10
DEFAULT_PLAYERS = 7

# 7 人局任务配置（向后兼容别名）
MISSION_TEAM_SIZES = MISSION_CONFIG[DEFAULT_PLAYERS]["sizes"]
MISSION_FAILS_REQUIRED = MISSION_CONFIG[DEFAULT_PLAYERS]["fails"]
N_PLAYERS = DEFAULT_PLAYERS
# 发言/指控阶段最多进行的轮数（硬上限，防止讨论无限循环）。每轮每位玩家至多发言一次、可沉默。
MAX_DISCUSS_ROUNDS = 3

# 名字库：按座位 pid 直接索引（names[pid]），故需 >= MAX_PLAYERS 个，真人那一席的名字不使用。
AI_NAMES = ["Aria", "Borin", "Cleo", "Doran", "Elga",
            "Finn", "Gwen", "Hale", "Iris", "Jora"]


def _rules_text(n, sizes, fails, roles):
    """生成给 LLM 的本局规则提示（随人数/配置而变）。"""
    sizes_str = "/".join(str(s) for s in sizes)
    fail_rounds = [str(i + 1) for i, f in enumerate(fails) if f > 1]
    fail_note = (f"第{('、'.join(fail_rounds))}轮需2张失败票才算失败，其余轮1张即失败"
                 if fail_rounds else "任意一轮只需1张失败票即失败")
    return (
        f"《阿瓦隆》{n} 人局规则：{describe_setup(roles)}。"
        f"五轮任务，出任务人数依次 {sizes_str}；{fail_note}。"
        "每轮：队长提名队伍→全体发言→全体公投(过半通过,平票否决)→入选者秘密出'成功/失败'(好人只能成功,坏人可失败)。"
        "好人完成3个任务→进入刺杀,刺客指认梅林,刺中则坏人胜、否则好人胜；坏人破坏3个任务直接胜；同一轮连续5次组队被否,坏人胜。"
    )


class Game:
    def __init__(self, n_players=DEFAULT_PLAYERS):
        self.new_game(n_players)

    # ------------------------------------------------------------------ setup
    def new_game(self, n_players=DEFAULT_PLAYERS):
        n = int(n_players)
        if n not in MISSION_CONFIG:
            raise ValueError(f"人数必须在 {MIN_PLAYERS}~{MAX_PLAYERS} 之间")
        self.n = n
        self.team_sizes = MISSION_CONFIG[n]["sizes"]
        self.fails_required = MISSION_CONFIG[n]["fails"]
        # 串行化所有会改变状态的操作（advance / submit_*）。view 读取不加锁，保证 GET 始终快返。
        self._lock = threading.RLock()
        # 专给"投票/任务出牌"用的小锁：真人写自己那张票时只抢它，**不**抢上面的大锁，
        # 从而不会被后台 worker 正在跑的 AI 出票（LLM，约 10s，持大锁）卡住。
        # 投票/任务本就是同时秘密进行的，真人可随时提交，AI 票稍后到齐再结算。
        self._ballot_lock = threading.Lock()
        roles = list(ROLE_SETUPS[n])
        random.shuffle(roles)

        human_id = random.randrange(n)              # 真人随机入座、随机角色
        names = list(AI_NAMES)
        random.shuffle(names)

        self.players = []
        for pid in range(n):
            self.players.append({
                "id": pid,
                "name": "你（真人）" if pid == human_id else names[pid],
                "is_human": pid == human_id,
                "role": roles[pid],
            })
        self.human_id = human_id

        # 规则允许的认知（信息隔离的关键）
        self.knowledge = build_knowledge(self.players)

        # 本局规则提示（随人数/配置而变），注入每个 Agent 的 system 提示
        rules_text = _rules_text(n, self.team_sizes, self.fails_required, roles)

        # 为每个 AI 玩家构建一个自治 Agent（仅注入其本人合法认知，保持信息隔离）
        names = [p["name"] for p in self.players]
        self.agents = {
            p["id"]: AvalonAgent(p["id"], self.knowledge[p["id"]], names, rules_text)
            for p in self.players if not p["is_human"]
        }

        # 流程状态
        self.round = 0                               # 当前任务轮次 idx 0..4
        self.leader = random.randrange(n)            # 首任队长随机
        self.phase = "TEAM"
        self.vote_track = 0                          # 本轮连续被否次数

        self.proposed_team = []                      # 当前提议的队伍
        self.votes = {}                              # pid -> bool（本次投票）
        self.mission_actions = {}                    # pid -> "success"/"fail"

        # 发言/指控阶段（多轮：每位玩家每轮至多发言一次，真人可连续多发；
        # 达到轮数上限或整轮无人发言则进入投票）
        self.discuss_order = []                      # 发言顺序（座位序，从队长起）
        self.discuss_round = 0                       # 当前发言轮（0-based，上限 MAX_DISCUSS_ROUNDS）
        self.discuss_idx = 0                         # 当前轮内的发言指针
        self.round_spoke = False                     # 本轮是否有人实际发言（全员沉默则提前结束）
        self.discussion = []                         # 本次提议的发言列表

        # 公开历史
        self.proposals = []                          # 含投票的组队记录
        self.missions = []                           # 任务结果
        self.chat = []                               # 全程公开发言记录（聊天面板）
        self.accuse_counts = {pid: 0 for pid in range(n)}  # 各玩家被指控次数
        self.log = []                                # 不泄密的日志

        # 结局
        self.winner = None
        self.assassin_target = None
        self.assassinated_role = None

        self._log(f"游戏开始：{n} 人局（{describe_setup(roles)}）。"
                  f"首任队长：{self.players[self.leader]['name']}。")
        # 不在此处推进；由服务器的后台 worker（或测试的 advance）驱动 AI。

    # ------------------------------------------------------------- public api
    def score(self):
        s = sum(1 for m in self.missions if m["success"])
        f = sum(1 for m in self.missions if not m["success"])
        return s, f

    def _public_info(self):
        """组装给 AI 的公开信息（不含任何隐藏身份）。"""
        return {
            "n": self.n,
            "player_ids": [p["id"] for p in self.players],
            "names": [p["name"] for p in self.players],
            "round": self.round,
            "team_size": self.team_sizes[self.round],
            "fails_required": self.fails_required[self.round],
            "leader": self.leader,
            "vote_track": self.vote_track,
            "proposals": self.proposals,
            "missions": self.missions,
            "chat": self.chat,
            "accuse_counts": dict(self.accuse_counts),
            "proposed_team": list(self.proposed_team),
        }

    def _log(self, msg):
        self.log.append(msg)

    def _run_agents(self, pids, fn):
        """对多个 Agent 执行同一决策；LLM 模式下并发（降低多智能体调用延迟），否则顺序。"""
        if not pids:
            return {}
        if llm.is_available() and len(pids) > 1:
            with ThreadPoolExecutor(max_workers=min(6, len(pids))) as ex:
                return dict(zip(pids, ex.map(fn, pids)))
        return {pid: fn(pid) for pid in pids}

    # ------------------------------------------------------------ state machine
    def advance(self):
        """驱动 AI 行动直至需要真人输入或游戏结束。（单飞：同一时刻只应有一个 advance 在跑）"""
        with self._lock:
            self._advance_locked()

    def _advance_locked(self):
        guard = 0
        while True:
            guard += 1
            if guard > 1000:
                break
            if self.phase == "TEAM":
                if self.players[self.leader]["is_human"]:
                    return                            # 等真人选人
                self._ai_pick_team()
                continue
            if self.phase == "DISCUSS":
                n = len(self.discuss_order)
                if self.discuss_idx >= n:             # 本轮发言已走完一圈
                    if not self.round_spoke:          # 整轮无人发言 -> 结束讨论，进入投票
                        self._start_vote()
                        continue
                    self.discuss_round += 1           # 还有人想说，进入下一轮讨论
                    if self.discuss_round >= MAX_DISCUSS_ROUNDS:
                        self._start_vote()            # 达到轮数上限 -> 投票（防无限讨论）
                        continue
                    self.discuss_idx = 0
                    self.round_spoke = False
                    continue
                if self.players[self.discuss_order[self.discuss_idx]]["is_human"]:
                    return                            # 轮到真人，等其发言/让过（真人本回合可多次发言）
                # 取"从当前位置起、到下一个真人之前"的一段连续 AI 发言者，并发生成（提速核心）。
                # 同一批内彼此看不到对方发言，但都能看到此前已记录的全部发言（含真人）。
                # decide_speech 在第 2 轮起可能返回 None，表示该 AI 本轮选择沉默（无新观点）。
                batch = []
                idx = self.discuss_idx
                while idx < n and not self.players[self.discuss_order[idx]]["is_human"]:
                    batch.append(self.discuss_order[idx])
                    idx += 1
                info, team, rnd = self._public_info(), self.proposed_team, self.discuss_round
                if llm.is_available() and len(batch) > 1:
                    # 并发生成，谁先返回就先记录 -> 前端轮询可看到发言陆续浮现（提升观感）
                    with ThreadPoolExecutor(max_workers=min(6, len(batch))) as ex:
                        futs = {ex.submit(self.agents[pid].decide_speech, info, team, rnd): pid
                                for pid in batch}
                        for fut in as_completed(futs):
                            self._apply_speech(futs[fut], fut.result())
                else:
                    for pid in batch:                 # 启发式：顺序即可（无网络延迟）
                        self._apply_speech(pid, self.agents[pid].decide_speech(info, team, rnd))
                self.discuss_idx += len(batch)        # 这批 AI 的发言回合统一推进
                continue
            if self.phase == "VOTE":
                self._ai_fill_votes()
                if self.human_id not in self.votes:
                    return                            # 等真人投票
                self._resolve_votes()
                continue
            if self.phase == "MISSION":
                self._ai_fill_mission()
                if self.human_id in self.proposed_team and self.human_id not in self.mission_actions:
                    return                            # 等真人执行任务
                self._resolve_mission()
                continue
            if self.phase == "ASSASSIN":
                assassin = self._assassin_id()
                if self.players[assassin]["is_human"]:
                    return                            # 等真人刺杀
                target = self.agents[assassin].decide_assassination(self._public_info())
                self.do_assassinate(assassin, target)
                return
            return  # OVER

    # ----- TEAM ----------------------------------------------------------
    def _ai_pick_team(self):
        leader = self.leader
        team = self.agents[leader].decide_team(self._public_info())
        self._set_team(team)

    def _set_team(self, team):
        size = self.team_sizes[self.round]
        team = list(dict.fromkeys(int(x) for x in team))   # 去重保序
        if len(team) != size:
            raise ValueError(f"队伍人数必须为 {size}")
        if any(t < 0 or t >= self.n for t in team):
            raise ValueError("非法玩家")
        self.proposed_team = team
        self.votes = {}
        # 进入发言/指控阶段：发言顺序为座位序，从队长开始；多轮讨论从第 0 轮起
        self.phase = "DISCUSS"
        self.discuss_order = [(self.leader + i) % self.n for i in range(self.n)]
        self.discuss_round = 0
        self.discuss_idx = 0
        self.round_spoke = False
        self.discussion = []
        names = "、".join(self.players[t]["name"] for t in team)
        self._log(f"第{self.round + 1}轮 第{self.vote_track + 1}次组队："
                  f"队长 {self.players[self.leader]['name']} 提名 [{names}]，进入发言。")

    def _record_speech(self, pid, text, accuse=None):
        """记录一条发言（公开）；指控则累加被指控次数。发言指针由调用方推进。"""
        entry = {
            "round": self.round,
            "attempt": self.vote_track + 1,
            "pid": pid,
            "name": self.players[pid]["name"],
            "text": text,
            "accuse": accuse,
        }
        self.discussion.append(entry)
        self.chat.append(entry)
        if accuse is not None and 0 <= accuse < self.n and accuse != pid:
            self.accuse_counts[accuse] += 1
        self.round_spoke = True                       # 本轮已有人发言（用于全员沉默时提前收场）

    def _apply_speech(self, pid, result):
        """处理一个 AI 的发言决策：(text, accuse) 则记录；None 或空文本=本轮沉默(pass)，跳过。"""
        if not result:
            return
        text, accuse = result
        if not (text or "").strip():
            return
        self._record_speech(pid, text, accuse)

    def _start_vote(self):
        """结束发言阶段，进入投票。"""
        self.phase = "VOTE"
        self.votes = {}

    def _is_human_turn_to_speak(self, pid):
        return (self.phase == "DISCUSS"
                and self.discuss_idx < len(self.discuss_order)
                and self.discuss_order[self.discuss_idx] == pid)

    def submit_speech(self, pid, text):
        """真人发言：记录但**不**让出发言权——真人可在本回合内连续多次发言。"""
        with self._lock:
            if self.phase != "DISCUSS":
                raise ValueError("当前不是发言阶段")
            if not self._is_human_turn_to_speak(pid):
                raise ValueError("还没轮到你发言")
            text = (text or "").strip()[:200]
            if not text:
                raise ValueError("发言不能为空（不想发言请点'结束发言'让过）")
            self._record_speech(pid, text, accuse=None)   # 真人自由发言，不做结构化指控解析

    def end_speech(self, pid):
        """真人结束本回合发言，让给下一位（推进发言指针）。"""
        with self._lock:
            if self.phase != "DISCUSS":
                raise ValueError("当前不是发言阶段")
            if not self._is_human_turn_to_speak(pid):
                raise ValueError("现在不是你的发言回合")
            self.discuss_idx += 1

    def end_discussion(self, pid):
        """真人提前结束整段讨论，直接进入投票（仅在你的发言回合可操作）。"""
        with self._lock:
            if self.phase != "DISCUSS":
                raise ValueError("当前不是发言阶段")
            if not self._is_human_turn_to_speak(pid):
                raise ValueError("现在不是你的发言回合")
            self._start_vote()

    def submit_team(self, pid, team):
        with self._lock:
            if self.phase != "TEAM" or pid != self.leader:
                raise ValueError("当前不可由你选人")
            self._set_team(team)

    # ----- VOTE ----------------------------------------------------------
    def _ai_fill_votes(self):
        info = self._public_info()
        team = self.proposed_team
        pending = [p["id"] for p in self.players
                   if not p["is_human"] and p["id"] not in self.votes]
        # LLM 调用在 ballot 锁之外完成（耗时），只在写结果时短暂加锁，避免阻塞真人投票
        results = self._run_agents(pending, lambda pid: self.agents[pid].decide_vote(info, team))
        with self._ballot_lock:
            for pid, v in results.items():
                self.votes.setdefault(pid, bool(v))   # 真人可能已先投，勿覆盖

    def submit_vote(self, pid, approve: bool):
        # 只抢 ballot 小锁：真人可随时投票，不被后台 AI 出票（持大锁）阻塞
        with self._ballot_lock:
            if self.phase != "VOTE":
                raise ValueError("当前不是投票阶段")
            self.votes[pid] = bool(approve)

    def _resolve_votes(self):
        approves = sum(1 for v in self.votes.values() if v)
        approved = approves > self.n / 2             # 需过半（平票=否决）
        self.proposals.append({
            "round": self.round,
            "attempt": self.vote_track + 1,
            "leader": self.leader,
            "team": list(self.proposed_team),
            "votes": {str(pid): bool(v) for pid, v in self.votes.items()},
            "approved": approved,
        })
        detail = "、".join(
            f"{self.players[pid]['name']}{'赞成' if self.votes[pid] else '反对'}"
            for pid in range(self.n))
        if approved:
            self._log(f"投票通过（{approves} 赞成 / {self.n - approves} 反对）。{detail}。开始执行任务。")
            self.vote_track = 0
            self.mission_actions = {}
            self.phase = "MISSION"
        else:
            self.vote_track += 1
            self._log(f"投票被否（{approves} 赞成 / {self.n - approves} 反对）。{detail}。")
            if self.vote_track >= 5:
                self._log("连续 5 次组队失败，坏人获胜！")
                self._end("evil")
                return
            self.leader = (self.leader + 1) % self.n
            self.proposed_team = []
            self.phase = "TEAM"
            self._log(f"队长轮换为 {self.players[self.leader]['name']}（第 {self.vote_track + 1} 次尝试）。")

    # ----- MISSION -------------------------------------------------------
    def _ai_fill_mission(self):
        info = self._public_info()
        team = self.proposed_team
        pending = [pid for pid in team
                   if not self.players[pid]["is_human"] and pid not in self.mission_actions]
        results = self._run_agents(pending, lambda pid: self.agents[pid].decide_mission(info, team))
        with self._ballot_lock:
            for pid, a in results.items():
                self.mission_actions.setdefault(pid, a)   # 真人可能已先出牌，勿覆盖

    def submit_mission(self, pid, action: str):
        # 同 submit_vote：只抢 ballot 小锁，真人随时可出牌，不被后台 AI 出牌阻塞
        with self._ballot_lock:
            if self.phase != "MISSION":
                raise ValueError("当前不是任务阶段")
            if pid not in self.proposed_team:
                raise ValueError("你不在任务队伍中")
            if action == "fail" and self.knowledge[pid]["alignment"] == "good":
                raise ValueError("好人只能选择成功")
            if action not in ("success", "fail"):
                raise ValueError("非法操作")
            self.mission_actions[pid] = action

    def _resolve_mission(self):
        fails = sum(1 for a in self.mission_actions.values() if a == "fail")
        required = self.fails_required[self.round]
        success = fails < required
        self.missions.append({
            "round": self.round,
            "team": list(self.proposed_team),
            "fails": fails,
            "success": success,
        })
        self._log(f"第{self.round + 1}轮任务{'成功' if success else '失败'}！"
                  f"（出现 {fails} 张失败票，需 {required} 张判负）")

        s, f = self.score()
        if s >= 3:
            self._log("好人已完成 3 个任务，进入【刺客刺杀】阶段。")
            self.phase = "ASSASSIN"
            self.proposed_team = []
            return
        if f >= 3:
            self._log("坏人破坏 3 个任务，坏人直接获胜！")
            self._end("evil")
            return
        # 下一轮
        self.round += 1
        self.vote_track = 0
        self.leader = (self.leader + 1) % self.n
        self.proposed_team = []
        self.phase = "TEAM"
        self._log(f"进入第{self.round + 1}轮。队长：{self.players[self.leader]['name']}。")

    # ----- ASSASSIN ------------------------------------------------------
    def _assassin_id(self):
        for p in self.players:
            if p["role"] == Role.ASSASSIN:
                return p["id"]
        return None

    def do_assassinate(self, pid, target):
      with self._lock:                                  # RLock：advance 内部调用也安全（可重入）
        if self.phase != "ASSASSIN":
            raise ValueError("当前不是刺杀阶段")
        if pid != self._assassin_id():
            raise ValueError("你不是刺客")
        target = int(target)
        self.assassin_target = target
        self.assassinated_role = self.players[target]["role"]
        hit = self.players[target]["role"] == Role.MERLIN
        self._log(f"刺客刺杀了 {self.players[target]['name']}"
                  f"（{ROLE_CN[self.players[target]['role']]}）。")
        if hit:
            self._log("刺中梅林！坏人获胜！")
            self._end("evil")
        else:
            self._log("没有刺中梅林，好人获胜！")
            self._end("good")

    def _end(self, winner):
        self.winner = winner
        self.phase = "OVER"

    # ------------------------------------------------------------ view
    def view_for_human(self):
        """构造发给前端的状态：公开信息 + 真人那一份合法私有信息。"""
        hid = self.human_id
        s, f = self.score()
        over = self.phase == "OVER"

        players = []
        for p in self.players:
            entry = {
                "id": p["id"],
                "name": p["name"],
                "is_human": p["is_human"],
                "is_leader": p["id"] == self.leader and self.phase != "OVER",
            }
            # 仅游戏结束时公开全部角色
            if over:
                entry["role"] = p["role"].value
                entry["role_cn"] = ROLE_CN[p["role"]]
                entry["alignment"] = alignment_of(p["role"])
            players.append(entry)

        # 真人合法私有信息
        myk = self.knowledge[hid]
        private = {
            "id": hid,
            "name": self.players[hid]["name"],
            "role": myk["role"].value,
            "role_cn": ROLE_CN[myk["role"]],
            "alignment": myk["alignment"],
            "info_text": list(myk["info_text"]),
            "known_evil": sorted(myk["known_evil"]),
            "merlin_candidates": sorted(myk["merlin_candidates"]),
        }

        # 真人当前应做什么
        waiting = self._waiting_for_human()

        return {
            "phase": self.phase,
            "n": self.n,
            "round": self.round,
            "mission_config": [
                {"size": self.team_sizes[i], "fails_required": self.fails_required[i]}
                for i in range(5)
            ],
            "missions": [
                {"round": m["round"], "success": m["success"], "fails": m["fails"]}
                for m in self.missions
            ],
            "score": {"good": s, "evil": f},
            "leader": self.leader,
            "vote_track": self.vote_track,
            "team_size": self.team_sizes[self.round],
            "fails_required": self.fails_required[self.round],
            "proposed_team": list(self.proposed_team),
            "players": players,
            "private": private,
            "proposals": self.proposals,
            "chat": self.chat,
            "accuse_counts": dict(self.accuse_counts),
            "current_speaker": (self.discuss_order[self.discuss_idx]
                                if self.phase == "DISCUSS"
                                and self.discuss_idx < len(self.discuss_order) else None),
            "discuss_round": self.discuss_round,
            "max_discuss_rounds": MAX_DISCUSS_ROUNDS,
            "log": self.log,
            "waiting": waiting,
            "winner": self.winner,
            "assassin_target": self.assassin_target,
            "assassinated_role": (ROLE_CN[self.assassinated_role]
                                  if self.assassinated_role else None),
            "human_in_team": hid in self.proposed_team,
            "human_voted": hid in self.votes,
            "human_mission_done": hid in self.mission_actions,
            "llm": {"enabled": llm.is_available(), "model": llm.model_name()},
        }

    def _waiting_for_human(self):
        hid = self.human_id
        if self.phase == "TEAM" and self.leader == hid:
            return "pick_team"
        if (self.phase == "DISCUSS" and self.discuss_idx < len(self.discuss_order)
                and self.discuss_order[self.discuss_idx] == hid):
            return "speak"
        if self.phase == "VOTE" and hid not in self.votes:
            return "vote"
        if self.phase == "MISSION" and hid in self.proposed_team and hid not in self.mission_actions:
            return "mission"
        if self.phase == "ASSASSIN" and self._assassin_id() == hid:
            return "assassinate"
        if self.phase == "OVER":
            return "over"
        return "wait"     # 轮到 AI / 等待结算
