"""
llm.py
======
LLM 客户端封装。为 Agent 提供"结构化决策"能力：给定 system / user 提示与一个工具(JSON Schema)，
让模型以该工具入参对应的 JSON 返回结构化结果。对外接口固定：

    is_available() -> bool
    model_name()   -> str
    provider_name()-> str
    decide(system, user, tool, max_tokens=1500) -> dict | None

后端：**OpenAI 兼容**（DeepSeek 等），通过 **LangChain 的 `ChatOpenAI`** 调用
`/chat/completions` + JSON 输出模式实现。**自动探测、优雅降级**（不可用时返回 None，上层回退
ai.py 启发式）。配置来源（优先级：环境变量 > config.py）：
     LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
   或 DeepSeek 习惯变量：DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL

说明：本文件是「模型客户端」；多智能体的**编排**（感知→推理→行动）见 agent.py，
那里用 **LangGraph** 把每个 Agent 的决策建模为一张状态图。

其它：
   AVALON_MODEL  覆盖模型名
   AVALON_LLM=0  强制关闭 LLM（即使有 Key 也走启发式）
   LLM_THINKING  on/off（默认 off）。DeepSeek v4 等是"思考型"模型，会先生成一大段思维链(reasoning_content)
                 再输出答案，既慢又容易把 token 预算耗尽导致 JSON 截断。默认关闭思考(更快更稳)；
                 设为 on 可开启思考（更"聪明"但更慢，需更大 max_tokens）。
"""

import json
import os
import threading

_lock = threading.Lock()
_init_done = False
_cfg = None          # {"provider","api_key","base_url","model"} 或 None
_models = {}         # (max_tokens, thinking_off) -> ChatOpenAI（按需缓存，避免重复建客户端）
_thinking_unsupported = False     # 若某网关不认 thinking 参数，置位后不再发送


def _from_config(name, default=None):
    """读取 config.py 中的变量（文件不存在则返回 default）。"""
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _resolve():
    """探测可用后端配置；环境变量优先于 config.py。返回配置 dict 或 None。"""
    if os.environ.get("AVALON_LLM", "").lower() in ("0", "off", "false", "no"):
        return None

    model_override = os.environ.get("AVALON_MODEL")

    # OpenAI 兼容（DeepSeek 等）
    api_key = (os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
               or _from_config("LLM_API_KEY"))
    if api_key:
        base = (os.environ.get("LLM_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
                or _from_config("LLM_BASE_URL") or "https://api.deepseek.com")
        model = (model_override or os.environ.get("LLM_MODEL")
                 or _from_config("LLM_MODEL") or "deepseek-chat")
        return {"provider": "openai", "api_key": api_key,
                "base_url": base.rstrip("/"), "model": model}

    return None


def _init():
    global _init_done, _cfg
    if _init_done:
        return
    with _lock:
        if _init_done:
            return
        _init_done = True
        _cfg = _resolve()


def is_available() -> bool:
    _init()
    return _cfg is not None


def provider_name() -> str:
    _init()
    return _cfg["provider"] if _cfg else "heuristic"


def model_name() -> str:
    _init()
    return _cfg["model"] if _cfg else "(heuristic)"


# ---------------------------------------------------------------------------
# 把工具的 JSON Schema 转成给模型的"请输出 JSON"指令（用于 JSON 输出模式）
# ---------------------------------------------------------------------------
def _schema_instruction(tool):
    props = tool.get("input_schema", {}).get("properties", {})
    required = set(tool.get("input_schema", {}).get("required", []))
    lines = []
    for key, spec in props.items():
        # JSON 模式下不索取 reasoning：它是自由长文本，极易把输出撑到 max_tokens 截断。
        # 模型仍会"隐式推理"，我们只取动作字段 + 一句短记忆。
        if key == "reasoning":
            continue
        t = spec.get("type", "string")
        if t == "array":
            it = spec.get("items", {}).get("type", "string")
            t = f"array<{it}>"
        flag = "必填" if key in required else "可选"
        desc = spec.get("description", "")
        lines.append(f'  "{key}" ({t}, {flag}): {desc}')
    return ("请只输出一个 JSON 对象（不要任何额外文字、不要代码块围栏、不要 reasoning 字段），字段如下：\n"
            + "\n".join(lines)
            + "\n注意：记忆字段(memo/belief_updates)务必从简；务必输出**完整且简短**的 JSON。")


def _minimal_instruction(tool):
    """只索取"必填动作字段"（去掉 reasoning 与记忆字段），用于截断后的极简重试。"""
    props = tool.get("input_schema", {}).get("properties", {})
    _meta = ("reasoning", "updated_notes", "belief_updates", "memo")
    req = [k for k in tool.get("input_schema", {}).get("required", [])
           if k not in _meta]
    if not req:                                   # 兜底：取第一个非 meta 字段
        req = [k for k in props if k not in _meta][:1]
    lines = []
    for k in req:
        spec = props.get(k, {})
        t = spec.get("type", "string")
        if t == "array":
            t = f"array<{spec.get('items', {}).get('type', 'string')}>"
        lines.append(f'  "{k}" ({t}): {spec.get("description", "")}')
    return ("严格只输出一个最小 JSON 对象，且**只含**以下字段，不要 reasoning、不要任何解释或多余字段：\n"
            + "\n".join(lines))


def _extract_json(text):
    """从模型文本中稳健提取首个 JSON 对象。"""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    try:
        return json.loads(s)
    except Exception:
        pass
    # 退而求其次：截取第一个 { 到最后一个 }
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# OpenAI 兼容（DeepSeek）：LangChain ChatOpenAI + /chat/completions + JSON 输出模式
# ---------------------------------------------------------------------------
def _thinking_off():
    v = (os.environ.get("LLM_THINKING") or _from_config("LLM_THINKING") or "off")
    return str(v).lower() not in ("on", "1", "true", "yes")


def _make_model(max_tokens):
    """构造（并按 max_tokens 缓存）一个 ChatOpenAI 客户端。"""
    from langchain_openai import ChatOpenAI
    key = (max_tokens, _thinking_off(), _thinking_unsupported)
    m = _models.get(key)
    if m is not None:
        return m
    kwargs = dict(
        model=_cfg["model"],
        api_key=_cfg["api_key"],
        base_url=_cfg["base_url"],
        temperature=0.6,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=0,
        # JSON 输出模式：与原实现一致，强制返回单个 JSON 对象
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    # 默认关闭"思考型"模型的思维链：更快、且不会把预算耗在 reasoning_content 上导致 JSON 截断。
    if _thinking_off() and not _thinking_unsupported:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    m = ChatOpenAI(**kwargs)
    _models[key] = m
    return m


def _call(system, user_content, max_tokens):
    """单次调用；返回 (解析后的 dict 或 None, finish_reason)。"""
    global _thinking_unsupported
    from langchain_core.messages import HumanMessage, SystemMessage
    msgs = [SystemMessage(content=system), HumanMessage(content=user_content)]
    sent_thinking = _thinking_off() and not _thinking_unsupported
    try:
        resp = _make_model(max_tokens).invoke(msgs)
    except Exception as e:
        # 某些网关不支持 thinking 参数 -> 记下并去掉该参数重试一次（兼容非 DeepSeek 端点）
        if sent_thinking and (getattr(e, "status_code", None) == 400
                              or e.__class__.__name__ == "BadRequestError"
                              or "thinking" in str(e).lower()):
            _thinking_unsupported = True
            resp = _make_model(max_tokens).invoke(msgs)
        else:
            raise
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    finish = (resp.response_metadata or {}).get("finish_reason")
    return _extract_json(content), finish


def decide(system: str, user: str, tool: dict, max_tokens: int = 1500):
    """用结构化方式拿决策；失败返回 None（上层回退启发式）。

    max_tokens 默认放宽到 1500：JSON 模式下若被截断会导致整段 JSON 不可解析，
    宁可多给预算也要保证决策字段完整。
    """
    _init()
    if not _cfg:
        return None
    try:
        out, reason = _call(system, user + "\n\n" + _schema_instruction(tool), max_tokens)
        if out is not None:
            return out
        # 极少数情况下仍失败：思考型模型即便如此也可能截断 -> 加大预算再试一次
        out, reason = _call(system, user + "\n\n" + _minimal_instruction(tool),
                            max_tokens=max(max_tokens * 2, 2000))
        if out is None:
            print(f"[llm] 输出不可解析(finish_reason={reason})，本次回退启发式。")
        return out
    except Exception as e:
        print(f"[llm] 调用失败，本次回退启发式：{e}")
        return None
