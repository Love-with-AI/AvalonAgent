"""
ai.py
=====
AI 智能体决策逻辑（模块化）。

严格的信息边界：每个 AI 决策函数只接收
  - me:        自己的 pid
  - know:      自己角色规则允许知道的信息（roles.build_knowledge 的产物）
  - pub:       公开信息（玩家数、座位、组队/投票/任务历史、当前轮次、队长……）
绝不传入其它玩家的真实角色。

公开信息 pub 结构（由 game.py 组装）：
{
  "n": 7,
  "player_ids": [0..6],
  "round": int,                # 当前任务轮次 idx (0..4)
  "team_size": int,
  "fails_required": int,
  "leader": pid,
  "vote_track": int,           # 本轮连续被否次数 (0..4)；attempt = vote_track+1
  "proposals": [ {round,attempt,leader,team:[pid],votes:{pid:bool},approved:bool}, ... ],
  "missions":  [ {round,team:[pid],fails:int,success:bool}, ... ],
}
"""

import random


def pub_score(pub):
    """从公开任务结果统计 (成功数, 失败数)。"""
    s = sum(1 for m in pub["missions"] if m["success"])
    f = sum(1 for m in pub["missions"] if not m["success"])
    return s, f


# ----------------------------------------------------------------------------
# 推理：基于公开信息为"其他玩家"打一个"坏人嫌疑分"。好人 AI 用它做判断。
# ----------------------------------------------------------------------------
def suspicion_scores(me, know, pub):
    """返回 dict[pid] -> float，分数越高越像坏人；负数表示受信任。

    阈值约定（见 decide_vote）：>= 1.0 视为"可疑"。
    """
    scores = {pid: 0.0 for pid in pub["player_ids"]}

    # 1) 失败任务：参与过失败任务的成员是头号嫌疑（固定 +1.3 起，足以越过 1.0 阈值）
    for m in pub["missions"]:
        if m["team"] and not m["success"]:
            share = m["fails"] / len(m["team"])
            for pid in m["team"]:
                scores[pid] += 1.3 + 1.5 * share
        # 2) 成功任务：参与者获得信任（降低嫌疑），便于好人凝聚"可信核心"
        elif m["team"] and m["success"]:
            for pid in m["team"]:
                scores[pid] -= 0.8

    # 3) 投票模式：对"事后失败的队伍"投赞成 -> 加嫌疑；投反对 -> 略减
    failed_teams = [tuple(sorted(m["team"])) for m in pub["missions"] if not m["success"]]
    for p in pub["proposals"]:
        if not p["approved"]:
            continue
        if tuple(sorted(p["team"])) in failed_teams:
            for pid, v in p["votes"].items():
                scores[int(pid)] += 0.4 if v else -0.25

    # 4) 公开发言中的指控：被指控越多越可疑（小幅，且封顶，避免单凭嘴遁定罪）
    for pid, cnt in pub.get("accuse_counts", {}).items():
        scores[int(pid)] += min(0.6, 0.15 * cnt)

    # 5) 自己合法已知的坏人 -> 拉满
    for pid in know["known_evil"]:
        scores[pid] += 100.0

    scores[me] = -100.0   # 自己绝对可信
    return scores


# ----------------------------------------------------------------------------
# 决策 1：是否赞成当前提议的队伍
# ----------------------------------------------------------------------------
def decide_vote(me, know, pub, team):
    team = set(team)
    attempt = pub["vote_track"] + 1
    alignment = know["alignment"]

    # 第 5 次组队投票（vote_track 已到 4）：否决即坏人获胜
    if attempt >= 5:
        return alignment == "good"   # 好人必须赞成保命；坏人否决求胜

    if alignment == "evil":
        # 队伍中是否有"能投失败票的人"：自己 或 已知坏人同伴
        failer_on_team = (me in team) or any(t in team for t in know["known_evil"])
        if failer_on_team:
            return True               # 有人能搞砸 -> 赞成让它过
        # 一支（已知范围内）全好人的队伍 -> 多数情况下否决，但偶尔放行以隐藏
        return random.random() < 0.25

    # ---- 好人 ----
    # 梅林：明确知道坏人，倾向否决含坏人的队伍，但**刻意保留噪声**以隐藏身份，
    # 否则"每次都精准反对坏人"会被刺客一眼识破。
    if know["role"] == "Merlin" and (team & know["known_evil"]):
        n_evil = len(team & know["known_evil"])
        # 队伍里坏人越多越要否决；只有一名坏人时偶尔放行以混淆视听。
        reject_prob = 0.6 if n_evil == 1 else 0.9
        # 关键任务（已两次失败 / 需 2 票的轮次）必须否决
        _, fails = pub_score(pub)
        if fails >= 2 or pub["fails_required"] > 1:
            reject_prob = 0.95
        return random.random() >= reject_prob   # True=赞成

    scores = suspicion_scores(me, know, pub)
    THRESHOLD = 1.0
    # 队伍里有可疑玩家（打过失败任务等）-> 否决。否则放行，避免无谓拉锯逼近第 5 次。
    if any(scores[pid] >= THRESHOLD for pid in team):
        return False
    return True


# ----------------------------------------------------------------------------
# 决策 2：作为队长挑选任务队伍
# ----------------------------------------------------------------------------
def decide_team(me, know, pub):
    size = pub["team_size"]
    ids = list(pub["player_ids"])
    alignment = know["alignment"]
    fails_required = pub["fails_required"]

    if alignment == "evil":
        team = [me]                                   # 自己先上（可投失败票）
        mates = [t for t in know["known_evil"] if t != me]
        # 需要 2 张失败票的轮次，尽量带一名已知同伴
        if fails_required > 1 and mates:
            team.append(mates[0])
        # 其余用"看起来干净"的好人填充，伪装成正常队伍
        others = [p for p in ids if p not in team and p not in know["known_evil"]]
        random.shuffle(others)
        for p in others:
            if len(team) >= size:
                break
            team.append(p)
        # 兜底
        i = 0
        while len(team) < size and i < len(ids):
            if ids[i] not in team:
                team.append(ids[i])
            i += 1
        return team[:size]

    # ---- 好人 ----
    scores = suspicion_scores(me, know, pub)
    # 梅林彻底排除已知坏人
    pool = [p for p in ids if p != me]
    if know["role"] == "Merlin":
        pool = [p for p in pool if p not in know["known_evil"]]
    pool.sort(key=lambda p: (scores[p], random.random()))   # 嫌疑低者优先

    # 优先复用"打过成功任务"的可信成员，凝聚干净核心
    proven = []
    for m in pub["missions"]:
        if m["success"]:
            for pid in m["team"]:
                if pid not in proven and pid != me and scores[pid] < 1.0:
                    proven.append(pid)
    team = [me]
    for pid in proven + pool:
        if len(team) >= size:
            break
        if pid not in team and scores[pid] < 1.0:   # 不带已知/疑似坏人
            team.append(pid)
    # 兜底填满
    for pid in pool:
        if len(team) >= size:
            break
        if pid not in team:
            team.append(pid)
    return team[:size]


# ----------------------------------------------------------------------------
# 决策 3：任务中出"成功 / 失败"票
# ----------------------------------------------------------------------------
def decide_mission(me, know, pub, team):
    if know["alignment"] == "good":
        return "success"                      # 好人只能成功

    fails_required = pub["fails_required"]
    # 坏人协同：已知同伴 + 自己 在队伍中的人，按 pid 排序。
    known_group = sorted([me] + [t for t in know["known_evil"] if t in team])

    if fails_required > 1:
        # 需要 2 张失败票的轮次：若确认在场坏人不足以凑齐，则不要单独出失败票
        #（否则白白浪费且暴露身份）。奥伯伦不认识同伴，无法确认 -> 选择成功。
        if len(known_group) >= fails_required and me in known_group:
            return "fail"
        return "success"

    # 仅需 1 张失败票的轮次：协同只让最小 pid 出失败票，避免暴露过多失败票数。
    if me in known_group:
        return "fail" if known_group.index(me) == 0 else "success"
    # 奥伯伦：不认识同伴，独自出失败票推动任务失败
    return "fail"


# ----------------------------------------------------------------------------
# 决策 4：刺客刺杀（仅 AI 刺客调用）—— 在好人中推断梅林
# ----------------------------------------------------------------------------
def decide_assassination(me, know, pub):
    # 候选 = 所有非"自己已知坏人"且非自己的玩家（奥伯伦不被刺客认识，故也是候选/误伤）
    known_evil = set(know["known_evil"]) | {me}
    candidates = [p for p in pub["player_ids"] if p not in known_evil]
    if not candidates:
        return random.choice([p for p in pub["player_ids"] if p != me])

    # "梅林味"评分：梅林会规避坏人——对含已知坏人的队伍投反对、对失败队伍投反对者更像梅林
    evil_known = set(know["known_evil"]) | {me}
    failed_teams = [tuple(sorted(m["team"])) for m in pub["missions"] if not m["success"]]
    score = {c: 0.0 for c in candidates}
    for p in pub["proposals"]:
        team = set(p["team"])
        has_evil = bool(team & evil_known)
        team_failed = tuple(sorted(p["team"])) in failed_teams and p["approved"]
        for c in candidates:
            v = p["votes"].get(str(c), p["votes"].get(c))
            if v is None:
                continue
            if has_evil and not v:
                score[c] += 1.0          # 反对含坏人的队伍 -> 像梅林
            if has_evil and v:
                score[c] -= 0.5
            if team_failed and not v:
                score[c] += 0.5
    # 发言信号：公开指控过刺客已知坏人（莫甘娜/自己）的玩家，更像"看得见坏人"的梅林
    for entry in pub.get("chat", []):
        spk = entry.get("pid")
        tgt = entry.get("accuse")
        if spk in score and tgt in evil_known:
            score[spk] += 0.8
    # 刺客并非全知：用带温度的加权抽样建模其"判断 + 不确定性"，
    # 而非每次都选分数最高者（否则机械式必中梅林，失真且不好玩）。
    lo = min(score.values())
    weights = {c: (score[c] - lo + 1.5) for c in candidates}   # 平移并整体抬高 -> 分布更平
    # 40% 概率锁定头号怀疑对象，60% 概率按权重在候选中抽样（模拟犹豫/被诱骗）
    best = max(score.values())
    top = [c for c in candidates if score[c] >= best - 1e-9]
    if random.random() < 0.4:
        return random.choice(top)
    total = sum(weights.values())
    r = random.uniform(0, total)
    acc = 0.0
    for c in candidates:
        acc += weights[c]
        if r <= acc:
            return c
    return random.choice(candidates)


# ----------------------------------------------------------------------------
# 决策 5：发言/指控阶段的发言（返回 (文本, 指控目标pid或None)）
# ----------------------------------------------------------------------------
def decide_speech(me, know, pub, team):
    """根据自身角色、合法认知与公开信息生成一句发言；可附带一个结构化指控目标。

    - 坏人会伪装/甩锅（栽赃好人、为含己方的队伍站台）。
    - 梅林会含蓄点名坏人，但刻意只点"已显可疑"的，避免暴露自己全知。
    - 派西维尔围绕"梅林候选"表态。
    - 好人依据失败任务/嫌疑表态；奥伯伦只能凭公开信息。
    """
    def nm(pid):
        return f"{pid + 1}号"

    scores = suspicion_scores(me, know, pub)
    others = [p for p in pub["player_ids"] if p != me]
    on_team = me in team
    team_names = "、".join(nm(t) for t in team)
    susp_sorted = sorted(others, key=lambda p: -scores[p])
    top_susp = susp_sorted[0]
    s_done, f_done = pub_score(pub)

    align = know["alignment"]
    role = know["role"]

    # ---------------- 坏人 ----------------
    if align == "evil":
        mates = set(know["known_evil"]) | {me}
        evil_on_team = bool(mates & set(team))
        # 甩锅对象：优先挑"已被公众怀疑且不是己方"的人附和，否则栽赃一名干净好人
        non_mates = [p for p in others if p not in mates]
        plausible = [p for p in non_mates if scores[p] >= 1.0]
        if plausible:
            target = plausible[0]
            text = f"这局失败八成跟 {nm(target)} 脱不了干系，我看他很可疑。"
        else:
            non_mates.sort(key=lambda p: scores[p])   # 挑最干净的好人栽赃，搅浑水
            target = non_mates[0] if non_mates else top_susp
            text = f"我观察下来 {nm(target)} 的节奏不太对，建议大家盯紧他。"
        if evil_on_team:
            text += f" 不过这支队伍（{team_names}）我看没问题，建议通过。"
        elif on_team:
            text += " 我在队里，放心，我肯定好好做任务。"
        return text, target

    # ---------------- 梅林 ----------------
    if role == "Merlin":
        # 只点"已经显得可疑"的已知坏人，避免精准点名暴露身份
        evil_public = [e for e in know["known_evil"] if scores[e] >= 1.0]
        evil_on_team = [e for e in know["known_evil"] if e in team]
        if evil_on_team and (f_done >= 1 or pub["fails_required"] > 1) and random.random() < 0.7:
            target = evil_on_team[0]
            return f"这支队伍我有点担心，{nm(target)} 让我不太放心，慎重投票。", target
        if evil_public and random.random() < 0.6:
            target = evil_public[0]
            return f"结合任务结果，{nm(target)} 的嫌疑不小。", target
        return "先别急，多看一两轮再下结论。", None

    # ---------------- 派西维尔 ----------------
    if role == "Percival":
        cands = sorted(know["merlin_candidates"])
        if cands:
            cs = "、".join(nm(c) for c in cands)
            if scores[top_susp] >= 1.0:
                return f"我比较信任 {cs} 里的人。另外 {nm(top_susp)} 我有点怀疑。", top_susp
            return f"{cs} 这两位我倾向相信，关键票看他们。", None

    # ---------------- 忠臣 / 普通好人 ----------------
    if scores[top_susp] >= 1.0:
        return f"{nm(top_susp)} 上过失败的车，我反对再带上他。", top_susp
    if on_team:
        return f"我在这支队伍（{team_names}）里，我会出成功，请大家支持。", None
    if s_done == 0 and f_done == 0:
        return "首轮信息不足，我先观察，倾向给队长一个机会。", None
    return "目前还看不太准，我会根据任务结果继续判断。", None
