"""
roles.py
========
角色定义、阵营划分，以及"每个玩家在规则允许下能知道的隐藏信息"的计算。

设计要点（信息隔离的核心）：
- 角色的真实身份只存在于后端 GameState 中。
- 每个玩家拥有一份独立的 "knowledge"（认知），里面**只**包含规则允许其知道的信息。
- AI 决策与前端展示都只读取对应玩家的 knowledge + 公开信息，绝不读取其它玩家的隐藏身份。
"""

from enum import Enum


class Role(str, Enum):
    MERLIN = "Merlin"               # 梅林：好人，知道所有坏人（莫德雷德除外）
    PERCIVAL = "Percival"           # 派西维尔：好人，看到梅林与莫甘娜，但无法区分
    LOYAL = "LoyalServant"          # 忠臣：好人，无额外信息
    ASSASSIN = "Assassin"           # 刺客：坏人，与莫甘娜/莫德雷德互知；终局负责刺杀
    MORGANA = "Morgana"             # 莫甘娜：坏人，与刺客/莫德雷德互知；对派西维尔伪装成梅林
    MORDRED = "Mordred"             # 莫德雷德：坏人，与刺客/莫甘娜互知，但**梅林看不见他**
    OBERON = "Oberon"               # 奥伯伦：坏人，但与其他坏人互不相识


# 各人数（5~10）局的角色清单。好人均含梅林+派西维尔，其余补忠臣；坏人按标准配置。
ROLE_SETUPS = {
    5:  [Role.MERLIN, Role.PERCIVAL, Role.LOYAL,
         Role.ASSASSIN, Role.MORGANA],
    6:  [Role.MERLIN, Role.PERCIVAL, Role.LOYAL, Role.LOYAL,
         Role.ASSASSIN, Role.MORGANA],
    7:  [Role.MERLIN, Role.PERCIVAL, Role.LOYAL, Role.LOYAL,
         Role.ASSASSIN, Role.MORGANA, Role.OBERON],
    8:  [Role.MERLIN, Role.PERCIVAL, Role.LOYAL, Role.LOYAL, Role.LOYAL,
         Role.ASSASSIN, Role.MORGANA, Role.MORDRED],
    9:  [Role.MERLIN, Role.PERCIVAL, Role.LOYAL, Role.LOYAL, Role.LOYAL, Role.LOYAL,
         Role.ASSASSIN, Role.MORGANA, Role.MORDRED],
    10: [Role.MERLIN, Role.PERCIVAL, Role.LOYAL, Role.LOYAL, Role.LOYAL, Role.LOYAL,
         Role.ASSASSIN, Role.MORGANA, Role.MORDRED, Role.OBERON],
}

# 7 人局角色清单（向后兼容别名）
SEVEN_PLAYER_ROLES = ROLE_SETUPS[7]

# 阵营
GOOD_ROLES = {Role.MERLIN, Role.PERCIVAL, Role.LOYAL}
EVIL_ROLES = {Role.ASSASSIN, Role.MORGANA, Role.MORDRED, Role.OBERON}

# 中文显示名
ROLE_CN = {
    Role.MERLIN: "梅林",
    Role.PERCIVAL: "派西维尔",
    Role.LOYAL: "忠臣",
    Role.ASSASSIN: "刺客",
    Role.MORGANA: "莫甘娜",
    Role.MORDRED: "莫德雷德",
    Role.OBERON: "奥伯伦",
}


def alignment_of(role: Role) -> str:
    return "good" if role in GOOD_ROLES else "evil"


def describe_setup(roles) -> str:
    """返回一局阵营构成的中文描述，如
    '好人4(梅林/派西维尔/忠臣×2) vs 坏人3(刺客/莫甘娜/奥伯伦)'。供规则提示文本复用。"""
    def side(predicate):
        counts = {}
        order = []
        for r in roles:
            if predicate(r):
                if r not in counts:
                    order.append(r)
                counts[r] = counts.get(r, 0) + 1
        total = sum(counts.values())
        parts = []
        for r in order:
            c = counts[r]
            parts.append(f"{ROLE_CN[r]}×{c}" if c > 1 else ROLE_CN[r])
        return total, "/".join(parts)

    g_total, g_desc = side(lambda r: alignment_of(r) == "good")
    e_total, e_desc = side(lambda r: alignment_of(r) == "evil")
    return f"好人{g_total}({g_desc}) vs 坏人{e_total}({e_desc})"


def build_knowledge(players):
    """
    根据全体玩家的真实角色，计算每个玩家"规则允许知道"的信息。

    players: list of dict，每项含 {"id", "role"}
    返回: dict[pid] -> {
        role, alignment,
        known_evil:        set[pid]  # 该玩家**确定**为坏人的 pid
        merlin_candidates: set[pid]  # 派西维尔看到的"梅林/莫甘娜"两人（无法区分）
        info_text:         list[str] # 给前端展示的文字提示
    }
    """
    by_role = {}
    for p in players:
        by_role.setdefault(p["role"], []).append(p["id"])

    def first(role):
        return by_role.get(role, [None])[0]

    merlin = first(Role.MERLIN)
    morgana = first(Role.MORGANA)

    all_evil = {pid for r in EVIL_ROLES for pid in by_role.get(r, [])}
    oberon_ids = set(by_role.get(Role.OBERON, []))
    mordred_ids = set(by_role.get(Role.MORDRED, []))
    # 互相认识的坏人圈：全部坏人，但奥伯伦既不认识别人、也不被别人认识。
    recognizing_evil = all_evil - oberon_ids
    # 梅林能看见的坏人：全部坏人，但莫德雷德对梅林隐身。
    merlin_sees = all_evil - mordred_ids

    knowledge = {}
    for p in players:
        pid, role = p["id"], p["role"]
        k = {
            "role": role,
            "alignment": alignment_of(role),
            "known_evil": set(),
            "merlin_candidates": set(),
            "info_text": [],
        }

        if role == Role.MERLIN:
            # 梅林看到坏人（莫德雷德除外），但看不到具体角色
            k["known_evil"] = set(merlin_sees)
            k["info_text"].append("你看到的【坏人】（不含具体身份、且看不到莫德雷德）：见高亮玩家。")

        elif role == Role.PERCIVAL:
            # 派西维尔看到梅林与莫甘娜两个名字，但无法区分谁是梅林
            cands = {pid_ for pid_ in (merlin, morgana) if pid_ is not None}
            k["merlin_candidates"] = cands
            k["info_text"].append("你看到两名【梅林候选】，其一是真梅林，另一是莫甘娜伪装。")

        elif role in (Role.ASSASSIN, Role.MORGANA, Role.MORDRED):
            # 刺客/莫甘娜/莫德雷德互知（不含奥伯伦）
            k["known_evil"] = recognizing_evil - {pid}
            if role == Role.ASSASSIN:
                k["info_text"].append("你是刺客。你的坏人同伴（不含奥伯伦）已高亮。任务全胜后由你刺杀。")
            elif role == Role.MORGANA:
                k["info_text"].append("你是莫甘娜。坏人同伴（不含奥伯伦）已高亮；你在派西维尔眼中是'梅林候选'。")
            else:  # 莫德雷德
                k["info_text"].append("你是莫德雷德。坏人同伴（不含奥伯伦）已高亮；"
                                      "**梅林看不见你**，可大胆伪装好人。")

        elif role == Role.OBERON:
            # 奥伯伦：不知队友，队友也不知他
            k["info_text"].append("你是奥伯伦（坏人），但你不认识其他坏人，他们也不认识你。")

        else:  # 忠臣
            k["info_text"].append("你是忠臣（好人），没有额外信息，靠推理找出坏人。")

        knowledge[pid] = k
    return knowledge
