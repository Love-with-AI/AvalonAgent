"""
test_sim.py — 无头自检：让 AI 替"真人"也自动决策，跑多局完整游戏，
验证：5~10 人各人数流程能跑通到结束、规则不崩、信息隔离正确（含莫德雷德对梅林隐身）。
运行： python3 test_sim.py
"""
import random
import ai
from game import Game, MISSION_CONFIG
from roles import Role, ROLE_SETUPS, EVIL_ROLES


def auto_play(g: Game):
    """让真人席位也由 AI 代打，把一局推到 OVER。"""
    hid = g.human_id
    guard = 0
    while g.phase != "OVER":
        guard += 1
        assert guard < 500, "可能死循环"
        w = g._waiting_for_human()
        info = g._public_info()
        if w == "pick_team":
            g.submit_team(hid, ai.decide_team(hid, g.knowledge[hid], info))
        elif w == "speak":
            if g.discuss_round == 0:                  # 首轮发一次言；其余轮直接让过
                text, _ = ai.decide_speech(hid, g.knowledge[hid], info, g.proposed_team)
                g.submit_speech(hid, text)
            g.end_speech(hid)
        elif w == "vote":
            g.submit_vote(hid, ai.decide_vote(hid, g.knowledge[hid], info, g.proposed_team))
        elif w == "mission":
            g.submit_mission(hid, ai.decide_mission(hid, g.knowledge[hid], info, g.proposed_team))
        elif w == "assassinate":
            g.do_assassinate(hid, ai.decide_assassination(hid, g.knowledge[hid], info))
        else:
            g.advance()


def check_isolation(g: Game):
    """验证：knowledge 不泄露规则之外的身份（按通用阿瓦隆规则，含莫德雷德/奥伯伦）。"""
    by_role = {}
    for p in g.players:
        by_role.setdefault(p["role"], []).append(p["id"])

    def ids(role):
        return set(by_role.get(role, []))

    evil = {pid for r in EVIL_ROLES for pid in by_role.get(r, [])}
    oberon = ids(Role.OBERON)
    mordred = ids(Role.MORDRED)
    recognizing = evil - oberon                       # 互知圈
    merlin_sees = evil - mordred                      # 梅林看不见莫德雷德

    for p in g.players:
        k = g.knowledge[p["id"]]
        pid, role = p["id"], p["role"]
        if role == Role.MERLIN:
            assert k["known_evil"] == merlin_sees, "梅林应看到坏人但看不到莫德雷德"
            assert not (k["known_evil"] & mordred), "梅林不应看到莫德雷德"
        elif role == Role.PERCIVAL:
            cands = {by_role[Role.MERLIN][0], by_role[Role.MORGANA][0]}
            assert k["merlin_candidates"] == cands
            assert k["known_evil"] == set()
        elif role in (Role.ASSASSIN, Role.MORGANA, Role.MORDRED):
            assert k["known_evil"] == recognizing - {pid}, "互知圈应为全部坏人去奥伯伦再去自己"
            assert not (k["known_evil"] & oberon), "互知圈不应含奥伯伦"
        elif role == Role.OBERON:
            assert k["known_evil"] == set(), "奥伯伦不知任何队友"
        else:  # 忠臣
            assert k["known_evil"] == set() and k["merlin_candidates"] == set()


def run_for_n(n, games):
    """对人数 n 跑若干局，返回 (好人胜, 坏人胜, 进入刺杀数, 命中数)。"""
    sizes = MISSION_CONFIG[n]["sizes"]
    good_wins = evil_wins = reached = hit = 0
    for _ in range(games):
        g = Game(n)
        assert g.n == n and len(g.players) == n
        check_isolation(g)
        auto_play(g)
        assert g.winner in ("good", "evil")
        if g.assassin_target is not None:
            reached += 1
            if g.assassinated_role == Role.MERLIN:
                hit += 1
        s, f = g.score()
        if g.winner == "good":
            assert s >= 3 and g.assassinated_role != "梅林"
            good_wins += 1
        else:
            evil_wins += 1
        for m in g.missions:
            assert len(m["team"]) == sizes[m["round"]]
            assert m["success"] == (m["fails"] < MISSION_CONFIG[n]["fails"][m["round"]])
    return good_wins, evil_wins, reached, hit


def main():
    random.seed(7)
    # 角色清单完整性：每个人数都恰好 n 人，且含梅林+派西维尔
    for n, roles in ROLE_SETUPS.items():
        assert len(roles) == n, f"{n} 人局角色数不符"
        assert Role.MERLIN in roles and Role.PERCIVAL in roles

    per_n = 80
    total_good = total_evil = 0
    for n in range(5, 11):
        gw, ew, reached, hit = run_for_n(n, per_n)
        total_good += gw
        total_evil += ew
        line = (f"  {n:>2} 人局：跑通 {per_n} 局，好人胜 {gw}/{per_n}（{gw/per_n:.0%}）")
        if reached:
            line += f"，进入刺杀 {reached} 局、命中梅林 {hit}（{hit/reached:.0%}）"
        print(line)

    total = total_good + total_evil
    print(f"✅ 全人数（5~10）共 {total} 局全部跑通，信息隔离校验通过。"
          f"好人总胜率 {total_good/total:.0%}。")


if __name__ == "__main__":
    main()
