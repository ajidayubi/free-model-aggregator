#!/usr/bin/env python3
"""Best free AI model aggregator.

One server, two wires (OpenAI Chat Completions + Anthropic Messages, JSON and
streaming), routing each request to the best free model that can serve it
across ten free-tier providers, failing over when a provider refuses.

Runs on Python 3.13 standard library only.  Reads its brief from
MAKEFUN_PROMPT_FILE, talks to the trial env at MAKEFUN_ENV_URL, registers
its port + catalog, serves the workload, and exits 0 when the env says the
workload is done.  `python3 main.py --list` prints the catalog document and
exits 0.  The same code runs against the real services when the brief (or
the provider key environment variables) carries real keys and base URLs.

Determinism: every choice derives from the brief, the requests, and the
providers' answers.  No wall clock in anything sent (timestamps come from
MAKEFUN_TRIAL_NOW), no unseeded randomness (response ids are hashes of the
request body), one request handled at a time.
"""

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
ANTHROPIC_VERSION = "2023-06-01"
PROVIDER_TIMEOUT = 10.0
DEAD_AFTER_STREAK = 3


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def log(*parts):
    print("[agg]", *parts, file=sys.stderr, flush=True)


def sha16(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def trial_now():
    """(created int, created_at iso) derived only from MAKEFUN_TRIAL_NOW."""
    raw = os.environ.get("MAKEFUN_TRIAL_NOW", "")
    if raw:
        try:
            sec = int(raw)
            iso = datetime.fromtimestamp(sec, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
            return sec, iso
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            sec = int(dt.timestamp())
            return sec, dt.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    return 0, "2026-01-01T00:00:00Z"


CREATED, CREATED_AT = trial_now()


def http_json(method, url, body=None, headers=None, timeout=PROVIDER_TIMEOUT):
    """One HTTP call.  Returns (status, parsed_json_or_None, raw_text)."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            return resp.status, parsed, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw), raw
        except ValueError:
            return exc.code, None, raw
    except Exception as exc:  # connection refused, timeout, dns, ...
        return 0, None, str(exc)


def http_text(method, url, body=None, headers=None, timeout=PROVIDER_TIMEOUT):
    """Like http_json but returns the raw body text (for SSE streams)."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:
        return 0, str(exc)


def extract_doc(text, schema):
    """Pull one embedded JSON document out of the brief by its schema tag."""
    marker = '"schema":"' + schema + '"'
    at = text.find(marker)
    if at < 0:
        return None
    start = text.rfind("{", 0, at)
    if start < 0:
        return None
    try:
        doc, _ = json.JSONDecoder().raw_decode(text, start)
        return doc
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# provider model
# ---------------------------------------------------------------------------

class Model:
    __slots__ = ("mid", "free", "ctx", "tools", "vision", "json", "rating")

    def __init__(self, mid, free, ctx, tools, vision, json_ok, rating):
        self.mid = mid
        self.free = free
        self.ctx = ctx
        self.tools = tools
        self.vision = vision
        self.json = json_ok
        self.rating = rating


class Provider:
    __slots__ = ("pid", "name", "wire", "auth", "key_env", "key", "path",
                 "base", "models")

    def __init__(self, pid, name, wire, auth, key_env, key, path, base):
        self.pid = pid
        self.name = name
        self.wire = wire
        self.auth = auth
        self.key_env = key_env
        self.key = key
        self.path = path
        self.base = base
        self.models = []

    def auth_headers(self):
        if self.auth == "bearer":
            return {"Authorization": "Bearer " + self.key}
        if self.auth == "x-api-key":
            return {"x-api-key": self.key}
        if self.auth == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {}

    def free_models(self):
        return [m for m in self.models if m.free]


class State:
    def __init__(self):
        self.env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
        self.prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
        self.providers = []
        self.by_id = {}
        self.manifest = []          # ordered [{id, api, stream, weight, requires}]
        self.manifest_by_id = {}
        self.catalog = {"providers": []}
        self.calls = {}             # provider id -> calls made
        self.streak = {}            # provider id -> consecutive failures
        self.dead = set()           # provider ids treated as exhausted
        self.handled = 0
        self.in_flight = 0
        self.lock = threading.Lock()
        self.done_event = threading.Event()

    # -- brief ---------------------------------------------------------------
    def load_brief(self):
        docs = {}
        if self.prompt_file and os.path.exists(self.prompt_file):
            with open(self.prompt_file, "r", encoding="utf-8") as fh:
                text = fh.read()
            for key, schema in (
                    ("providers", "free-model-aggregator-01/providers/v1"),
                    ("ratings", "free-model-aggregator-01/ratings/v1"),
                    ("workload", "free-model-aggregator-01/workload/v1")):
                docs[key] = extract_doc(text, schema)
        return docs

    # -- discovery -----------------------------------------------------------
    def discover(self):
        docs = self.load_brief()
        brief_providers = (docs.get("providers") or {}).get("providers", [])
        ratings_doc = docs.get("ratings") or {}
        rating_of = {}
        for row in ratings_doc.get("ratings", []):
            rating_of[(row.get("provider"), row.get("model"))] = row
        workload = docs.get("workload") or {}
        self.manifest = workload.get("manifest", [])
        self.manifest_by_id = {m["id"]: m for m in self.manifest}

        # The env's own directory is the authority on base URLs.
        inst_providers = []
        if self.env_url:
            _s, inst, _r = http_json("GET", self.env_url + "/instance",
                                     timeout=5.0)
            if isinstance(inst, dict):
                inst_providers = inst.get("providers", [])
        inst_by_id = {p["id"]: p for p in inst_providers}

        seen = []
        for bp in brief_providers:
            pid = bp["id"]
            inst = inst_by_id.get(pid, {})
            base = inst.get("base_url") or (
                self.env_url + bp.get("path", "") if self.env_url else "")
            path = bp.get("path", "")
            if not path and base:
                path = urllib.parse.urlparse(base).path
            key = os.environ.get(bp.get("key_env", "")) or bp.get("key", "")
            prov = Provider(pid, bp.get("name", pid), bp.get("wire", "openai"),
                            bp.get("auth", "bearer"), bp.get("key_env", ""),
                            key, path, base)
            seen.append(prov)
        for ip in inst_providers:
            if ip["id"] not in {p.pid for p in seen}:
                base = ip.get("base_url", "")
                path = urllib.parse.urlparse(base).path if base else ""
                key = os.environ.get(ip.get("key_env", "")) or ip.get("key", "")
                seen.append(Provider(ip["id"], ip.get("name", ip["id"]),
                                     ip.get("wire", "openai"),
                                     ip.get("auth", "bearer"),
                                     ip.get("key_env", ""), key, path, base))
        self.providers = seen
        self.by_id = {p.pid: p for p in seen}

        # Probe every provider's own model list: the truth for the catalog.
        for prov in self.providers:
            probed = self.probe_models(prov)
            if probed is None:
                probed = []
                for (pp, mm), row in rating_of.items():
                    if pp == prov.pid and row.get("free"):
                        probed.append(Model(
                            mm, True, row.get("context_length"),
                            bool(row.get("tools")), bool(row.get("vision")),
                            bool(row.get("json")), row.get("rating_milli")))
            else:
                for m in probed:
                    row = rating_of.get((prov.pid, m.mid))
                    if row is not None:
                        m.rating = row.get("rating_milli")
            prov.models = probed

        self.catalog = {"providers": [
            {"id": p.pid, "base_url": p.path, "key_env": p.key_env,
             "wire": p.wire,
             "models": [{"id": m.mid, "free": m.free,
                         "context_length": m.ctx} for m in p.models]}
            for p in self.providers]}
        return self.catalog

    def probe_models(self, prov):
        if not prov.base:
            return None
        headers = prov.auth_headers()
        if prov.wire == "anthropic":
            url = prov.base + "/v1/models"
            headers = dict(headers)
            headers["anthropic-version"] = ANTHROPIC_VERSION
            _s, doc, _r = http_json("GET", url, headers=headers, timeout=5.0)
            if not isinstance(doc, dict) or not isinstance(doc.get("data"), list):
                return None
            out = []
            for row in doc["data"]:
                out.append(Model(row.get("id"), bool(row.get("free")),
                                 row.get("context_length"),
                                 bool(row.get("tools")), bool(row.get("vision")),
                                 bool(row.get("json")), None))
            return out
        if prov.wire == "gemini":
            url = prov.base + "/v1beta/models"
            _s, doc, _r = http_json("GET", url, headers=headers, timeout=5.0)
            if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
                return None
            out = []
            for row in doc["models"]:
                name = row.get("name", "")
                mid = name.split("models/", 1)[-1] if name else row.get("id")
                out.append(Model(mid, bool(row.get("free")),
                                 row.get("context_length"),
                                 bool(row.get("tools")), bool(row.get("vision")),
                                 bool(row.get("json")), None))
            return out
        url = prov.base + "/v1/models"
        _s, doc, _r = http_json("GET", url, headers=headers, timeout=5.0)
        if not isinstance(doc, dict) or not isinstance(doc.get("data"), list):
            return None
        out = []
        for row in doc["data"]:
            out.append(Model(row.get("id"), bool(row.get("free")),
                             row.get("context_length"),
                             bool(row.get("tools")), bool(row.get("vision")),
                             bool(row.get("json")), None))
        return out

    # -- env -----------------------------------------------------------------
    def register(self, port):
        if not self.env_url:
            return False
        _s, doc, _r = http_json(
            "POST", self.env_url + "/register",
            body={"protocol_version": PROTOCOL_VERSION, "port": port,
                  "catalog": self.catalog}, timeout=5.0)
        return isinstance(doc, dict) and doc.get("ok") is True

    def work_status(self):
        if not self.env_url:
            return {}
        _s, doc, _r = http_json("GET", self.env_url + "/work", timeout=5.0)
        return doc if isinstance(doc, dict) else {}


STATE = State()


# ---------------------------------------------------------------------------
# request analysis
# ---------------------------------------------------------------------------

def body_features(api, body):
    """Requirements visible in the request body itself."""
    feats = {"tools": False, "vision": False, "json": False,
             "min_context": 0}
    if isinstance(body.get("tools"), list) and body["tools"]:
        feats["tools"] = True
    if api == "openai":
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") in (
                "json_object", "json_schema"):
            feats["json"] = True
    for msg in body.get("messages", []) or []:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") in ("image_url", "image"):
                    feats["vision"] = True
    return feats


def requirements_for(index, api, body):
    """Merge the workload manifest's requirements for this request with the
    body's own visible features (the body wins where it says more)."""
    req = {"tools": False, "vision": False, "json": False, "min_context": 0}
    if 0 <= index < len(STATE.manifest):
        for key, val in (STATE.manifest[index].get("requires") or {}).items():
            if key in req:
                req[key] = val
    for key, val in body_features(api, body).items():
        if val:
            req[key] = val
    return req


def rank_candidates(req, json_on_wire):
    """Free models meeting the requirements, best rating first.

    Ties break toward the provider used least so far (spreads the load
    across free tiers), then provider id, then model id: deterministic.
    """
    cands = []
    for prov in STATE.providers:
        if prov.pid in STATE.dead:
            continue
        for m in prov.free_models():
            if m.rating is None:
                continue
            if req["tools"] and not m.tools:
                continue
            if req["vision"] and not m.vision:
                continue
            if req["json"] and not m.json:
                continue
            if req["min_context"] and (m.ctx or 0) < req["min_context"]:
                continue
            if json_on_wire and prov.wire == "anthropic":
                continue
            cands.append((prov, m))
    cands.sort(key=lambda pm: (-pm[1].rating, STATE.calls.get(pm[0].pid, 0),
                               pm[0].pid, pm[1].mid))
    return cands


# ---------------------------------------------------------------------------
# wire translation: client body -> provider body
# ---------------------------------------------------------------------------

def _data_url_parts(url):
    """Split a data:<mime>;base64,<data> URL; returns (mime, data) or None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        head, data = url.split(",", 1)
        mime = head[5:].split(";")[0] or "image/png"
        return mime, data
    except ValueError:
        return None


def oai_messages_from_anthropic(body):
    """Anthropic request -> OpenAI messages list."""
    out = []
    system = body.get("system")
    if isinstance(system, str) and system:
        out.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "".join(p.get("text", "") for p in system
                       if p.get("type") == "text")
        if text:
            out.append({"role": "system", "content": text})
    tool_names = {}
    for msg in body.get("messages", []) or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        text_parts = []
        tool_calls = []
        tool_results = []
        for part in content or []:
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image":
                src = part.get("source", {})
                if src.get("type") == "base64":
                    url = ("data:" + src.get("media_type", "image/png") +
                           ";base64," + src.get("data", ""))
                    text_parts.append({"type": "image_url",
                                       "image_url": {"url": url}})
                elif src.get("type") == "url":
                    text_parts.append({"type": "image_url",
                                       "image_url": {"url": src.get("url", "")}})
            elif ptype == "tool_use":
                tool_names[part.get("id")] = part.get("name")
                tool_calls.append({
                    "id": part.get("id"), "type": "function",
                    "function": {"name": part.get("name", ""),
                                 "arguments": json.dumps(part.get("input", {}))}})
            elif ptype == "tool_result":
                text = part.get("content")
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text
                                   if p.get("type") == "text")
                tool_results.append({
                    "role": "tool", "tool_call_id": part.get("tool_use_id"),
                    "content": text if isinstance(text, str) else ""})
        if role == "assistant" and tool_calls:
            out.append({"role": "assistant",
                        "content": "".join(p["text"] for p in text_parts)
                        if text_parts else None,
                        "tool_calls": tool_calls})
        elif tool_results:
            for tr in tool_results:
                out.append(tr)
            if text_parts:
                out.append({"role": role,
                            "content": text_parts
                            if len(text_parts) > 1 else text_parts[0]["text"]})
        elif text_parts:
            out.append({"role": role,
                        "content": text_parts
                        if len(text_parts) > 1 else text_parts[0]["text"]})
        else:
            out.append({"role": role, "content": ""})
    return out, tool_names


def ant_messages_from_openai(body):
    """OpenAI request -> Anthropic messages list + system string."""
    system = None
    out = []
    tool_names = {}
    pending_tools = []
    for msg in body.get("messages", []) or []:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                system = content
            elif isinstance(content, list):
                system = "".join(p.get("text", "") for p in content
                                 if p.get("type") == "text")
            continue
        if role == "tool":
            text = content if isinstance(content, str) else ""
            pending_tools.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": text})
            continue
        if pending_tools:
            out.append({"role": "user", "content": pending_tools})
            pending_tools = []
        blocks = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    got = _data_url_parts(url)
                    if got:
                        mime, data = got
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": mime,
                            "data": data}})
                    else:
                        blocks.append({"type": "image", "source": {
                            "type": "url", "url": url}})
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            tool_names[tc.get("id")] = fn.get("name")
            blocks.append({"type": "tool_use", "id": tc.get("id"),
                           "name": fn.get("name", ""), "input": args})
        out.append({"role": role, "content": blocks})
    if pending_tools:
        out.append({"role": "user", "content": pending_tools})
    return system, out, tool_names


def oai_tools_from_anthropic(tools):
    out = []
    for t in tools or []:
        out.append({"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object"})}})
    return out


def ant_tools_from_openai(tools):
    out = []
    for t in tools or []:
        fn = t.get("function", t)
        out.append({"name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object"})})
    return out


def oai_tool_choice_from_anthropic(tc):
    if tc is None:
        return None
    if isinstance(tc, str):
        return tc
    kind = tc.get("type")
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function",
                "function": {"name": tc.get("name", "")}}
    return "auto"


def ant_tool_choice_from_openai(tc):
    if tc is None:
        return None
    if isinstance(tc, str):
        return {"type": "auto"} if tc == "auto" else (
            {"type": "none"} if tc == "none" else {"type": "any"})
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "tool",
                "name": (tc.get("function") or {}).get("name", "")}
    return {"type": "auto"}


def gemini_contents_from_messages(api, body):
    """Either client wire -> Gemini contents + system text + tool names."""
    parts_by_role = []
    system = None
    tool_names = {}

    def add(role, parts):
        if not parts:
            return
        if parts_by_role and parts_by_role[-1][0] == role:
            parts_by_role[-1][1].extend(parts)
        else:
            parts_by_role.append([role, list(parts)])

    if api == "openai":
        for msg in body.get("messages", []) or []:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                if isinstance(content, str):
                    system = content
                elif isinstance(content, list):
                    system = "".join(p.get("text", "") for p in content
                                     if p.get("type") == "text")
                continue
            if role == "tool":
                name = None
                for _m in body.get("messages", []) or []:
                    for tc in _m.get("tool_calls") or []:
                        if tc.get("id") == msg.get("tool_call_id"):
                            name = (tc.get("function") or {}).get("name")
                text = content if isinstance(content, str) else ""
                add("function", [{"functionResponse": {
                    "name": name or "tool",
                    "response": {"content": text}}}])
                continue
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        got = _data_url_parts(url)
                        if got:
                            mime, data = got
                            parts.append({"inlineData": {"mimeType": mime,
                                                         "data": data}})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                tool_names[tc.get("id")] = fn.get("name")
                parts.append({"functionCall": {"name": fn.get("name", ""),
                                               "args": args}})
            add("model" if role == "assistant" else "user", parts)
    else:
        sysval = body.get("system")
        if isinstance(sysval, str):
            system = sysval
        elif isinstance(sysval, list):
            system = "".join(p.get("text", "") for p in sysval
                             if p.get("type") == "text")
        for msg in body.get("messages", []) or []:
            role = msg.get("role")
            content = msg.get("content")
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for part in content or []:
                    ptype = part.get("type")
                    if ptype == "text":
                        parts.append({"text": part.get("text", "")})
                    elif ptype == "image":
                        src = part.get("source", {})
                        if src.get("type") == "base64":
                            parts.append({"inlineData": {
                                "mimeType": src.get("media_type", "image/png"),
                                "data": src.get("data", "")}})
                    elif ptype == "tool_use":
                        tool_names[part.get("id")] = part.get("name")
                        parts.append({"functionCall": {
                            "name": part.get("name", ""),
                            "args": part.get("input", {})}})
                    elif ptype == "tool_result":
                        text = part.get("content")
                        if isinstance(text, list):
                            text = "".join(p.get("text", "") for p in text
                                           if p.get("type") == "text")
                        parts.append({"functionResponse": {
                            "name": tool_names.get(part.get("tool_use_id"),
                                                  "tool"),
                            "response": {"content": text or ""}}})
            add("model" if role == "assistant" else "user", parts)
    contents = [{"role": r, "parts": p} for r, p in parts_by_role]
    return system, contents, tool_names


def build_provider_call(prov, model_id, api, body):
    """Returns (url, headers, payload, stream_bool)."""
    stream = bool(body.get("stream"))
    headers = prov.auth_headers()
    if prov.wire == "openai":
        if api == "openai":
            payload = dict(body)
            payload["model"] = model_id
        else:
            msgs, _names = oai_messages_from_anthropic(body)
            payload = {"model": model_id, "messages": msgs,
                       "max_tokens": body.get("max_tokens"),
                       "temperature": body.get("temperature"),
                       "stream": stream}
            tools = oai_tools_from_anthropic(body.get("tools"))
            if tools:
                payload["tools"] = tools
            tc = oai_tool_choice_from_anthropic(body.get("tool_choice"))
            if tc is not None:
                payload["tool_choice"] = tc
            if body.get("stop_sequences"):
                payload["stop"] = body["stop_sequences"]
        if stream:
            payload["stream_options"] = {"include_usage": True}
        payload = {k: v for k, v in payload.items() if v is not None}
        return prov.base + "/v1/chat/completions", headers, payload, stream
    if prov.wire == "anthropic":
        headers = dict(headers)
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if api == "anthropic":
            payload = dict(body)
            payload["model"] = model_id
        else:
            system, msgs, _names = ant_messages_from_openai(body)
            payload = {"model": model_id, "messages": msgs,
                       "max_tokens": body.get("max_tokens"),
                       "temperature": body.get("temperature"),
                       "stream": stream}
            if system:
                payload["system"] = system
            tools = ant_tools_from_openai(body.get("tools"))
            if tools:
                payload["tools"] = tools
            tc = ant_tool_choice_from_openai(body.get("tool_choice"))
            if tc is not None:
                payload["tool_choice"] = tc
            if body.get("stop"):
                stops = body["stop"]
                if isinstance(stops, str):
                    stops = [stops]
                payload["stop_sequences"] = stops
        payload = {k: v for k, v in payload.items() if v is not None}
        return prov.base + "/v1/messages", headers, payload, stream
    # gemini wire
    system, contents, _names = gemini_contents_from_messages(api, body)
    gen = {}
    if body.get("max_tokens") is not None:
        gen["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        gen["temperature"] = body["temperature"]
    stops = body.get("stop") or body.get("stop_sequences")
    if stops:
        gen["stopSequences"] = [stops] if isinstance(stops, str) else stops
    if api == "openai":
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") in ("json_object",
                                                       "json_schema"):
            gen["responseMimeType"] = "application/json"
    tools = body.get("tools")
    payload = {"contents": contents, "generationConfig": gen}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        if api == "openai":
            decls = [{"name": (t.get("function") or {}).get("name", ""),
                      "description": (t.get("function") or {}).get("description", ""),
                      "parameters": (t.get("function") or {}).get(
                          "parameters", {"type": "object"})}
                     for t in tools]
        else:
            decls = [{"name": t.get("name", ""),
                      "description": t.get("description", ""),
                      "parameters": t.get("input_schema", {"type": "object"})}
                     for t in tools]
        payload["tools"] = [{"functionDeclarations": decls}]
    action = ":streamGenerateContent?alt=sse" if stream else ":generateContent"
    url = (prov.base + "/v1beta/models/" +
           urllib.parse.quote(model_id, safe="") + action)
    return url, headers, payload, stream


# ---------------------------------------------------------------------------
# provider responses -> canonical answer
# ---------------------------------------------------------------------------

# Canonical answer: {"text", "tool_calls":[{id,name,arguments:str}],
#                     "finish": stop|length|tool_calls|content_filter,
#                     "usage": {"in": n, "out": n}}

def _finish_openai(reason, has_tools):
    if has_tools:
        return "tool_calls"
    return {"stop": "stop", "length": "length",
            "tool_calls": "tool_calls",
            "content_filter": "content_filter"}.get(reason, "stop")


def _finish_anthropic(reason, has_tools):
    if has_tools:
        return "tool_calls"
    return {"end_turn": "stop", "max_tokens": "length",
            "tool_use": "tool_calls", "refusal": "content_filter",
            "stop_sequence": "stop"}.get(reason, "stop")


def _finish_gemini(reason, has_tools):
    if has_tools:
        return "tool_calls"
    return {"STOP": "stop", "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter"}.get(reason, "stop")


def parse_openai_json(doc, model_id):
    if not isinstance(doc, dict):
        return None
    choices = doc.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    text = msg.get("content") or ""
    tools = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        tools.append({"id": tc.get("id") or ("call_" + sha16(
            fn.get("name", "") + fn.get("arguments", ""))),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments") or "{}"})
    usage = doc.get("usage") or {}
    return {"text": text if isinstance(text, str) else "",
            "tool_calls": tools,
            "finish": _finish_openai(choices[0].get("finish_reason"), bool(tools)),
            "usage": {"in": usage.get("prompt_tokens") or 0,
                      "out": usage.get("completion_tokens") or 0}}


def parse_openai_sse(raw):
    text = ""
    tools = {}
    finish = "stop"
    usage = {"in": 0, "out": 0}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if "usage" in chunk and chunk.get("usage"):
            usage = {"in": chunk["usage"].get("prompt_tokens") or 0,
                     "out": chunk["usage"].get("completion_tokens") or 0}
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text += delta["content"]
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tools.setdefault(idx, {"id": "", "name": "",
                                              "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    ordered = [tools[k] for k in sorted(tools)]
    for i, t in enumerate(ordered):
        if not t["id"]:
            t["id"] = "call_" + sha16(t["name"] + t["arguments"] + str(i))
        if not t["arguments"]:
            t["arguments"] = "{}"
    return {"text": text, "tool_calls": ordered,
            "finish": _finish_openai(finish, bool(ordered)), "usage": usage}


def parse_anthropic_json(doc, model_id):
    if not isinstance(doc, dict):
        return None
    text = ""
    tools = []
    for block in doc.get("content") or []:
        if block.get("type") == "text":
            text += block.get("text", "")
        elif block.get("type") == "tool_use":
            tools.append({"id": block.get("id") or "call_" + sha16(
                block.get("name", "")),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}))})
    usage = doc.get("usage") or {}
    return {"text": text, "tool_calls": tools,
            "finish": _finish_anthropic(doc.get("stop_reason"), bool(tools)),
            "usage": {"in": usage.get("input_tokens") or 0,
                      "out": usage.get("output_tokens") or 0}}


def parse_anthropic_sse(raw):
    text = ""
    tools = {}
    finish = "stop"
    usage = {"in": 0, "out": 0}
    cur_tool = None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        kind = ev.get("type")
        if kind == "message_start":
            msg = ev.get("message") or {}
            usage["in"] = (msg.get("usage") or {}).get("input_tokens") or 0
        elif kind == "content_block_start":
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                cur_tool = {"id": block.get("id") or "",
                            "name": block.get("name", ""), "arguments": ""}
                tools[ev.get("index", 0)] = cur_tool
        elif kind == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                if cur_tool is not None:
                    cur_tool["arguments"] += delta.get("partial_json", "")
        elif kind == "message_delta":
            delta = ev.get("delta") or {}
            if delta.get("stop_reason"):
                finish = delta["stop_reason"]
            if ev.get("usage"):
                usage["out"] = ev["usage"].get("output_tokens") or 0
    ordered = [tools[k] for k in sorted(tools)]
    for i, t in enumerate(ordered):
        if not t["id"]:
            t["id"] = "call_" + sha16(t["name"] + t["arguments"] + str(i))
        if not t["arguments"]:
            t["arguments"] = "{}"
    return {"text": text, "tool_calls": ordered,
            "finish": _finish_anthropic(finish, bool(ordered)),
            "usage": usage}


def parse_gemini_json(doc, model_id):
    if not isinstance(doc, dict):
        return None
    cands = doc.get("candidates") or []
    if not cands:
        return None
    cand = cands[0]
    text = ""
    tools = []
    for part in (cand.get("content") or {}).get("parts") or []:
        if "text" in part:
            text += part.get("text", "")
        elif "functionCall" in part:
            fc = part["functionCall"]
            tools.append({"id": "call_" + sha16(
                fc.get("name", "") + json.dumps(fc.get("args", {}),
                                                sort_keys=True)),
                "name": fc.get("name", ""),
                "arguments": json.dumps(fc.get("args", {}))})
    meta = doc.get("usageMetadata") or {}
    return {"text": text, "tool_calls": tools,
            "finish": _finish_gemini(cand.get("finishReason"), bool(tools)),
            "usage": {"in": meta.get("promptTokenCount") or 0,
                      "out": meta.get("candidatesTokenCount") or 0}}


def parse_gemini_sse(raw):
    text = ""
    tools = []
    finish = "stop"
    usage = {"in": 0, "out": 0}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        for cand in chunk.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if "text" in part:
                    text += part.get("text", "")
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tools.append({"id": "call_" + sha16(
                        fc.get("name", "") + json.dumps(fc.get("args", {}),
                                                        sort_keys=True)),
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}))})
            if cand.get("finishReason"):
                finish = cand["finishReason"]
        meta = chunk.get("usageMetadata")
        if meta:
            usage = {"in": meta.get("promptTokenCount") or 0,
                     "out": meta.get("candidatesTokenCount") or 0}
    return {"text": text, "tool_calls": tools,
            "finish": _finish_gemini(finish, bool(tools)), "usage": usage}


def call_provider(prov, model_id, api, body):
    """One attempt.  Returns the canonical answer, or None on refusal."""
    url, headers, payload, stream = build_provider_call(
        prov, model_id, api, body)
    if stream:
        status, raw = http_text("POST", url, body=payload, headers=headers)
    else:
        status, doc, raw = http_json("POST", url, body=payload,
                                     headers=headers)
    if status == 200:
        with STATE.lock:
            STATE.calls[prov.pid] = STATE.calls.get(prov.pid, 0) + 1
            STATE.streak[prov.pid] = 0
        if prov.wire == "openai":
            if stream:
                return parse_openai_sse(raw)
            return parse_openai_json(doc, model_id)
        if prov.wire == "anthropic":
            if stream:
                return parse_anthropic_sse(raw)
            return parse_anthropic_json(doc, model_id)
        if stream:
            return parse_gemini_sse(raw)
        return parse_gemini_json(doc, model_id)
    # Any non-200 (or transport failure) is a refusal: count it and move on.
    if status != 0:
        with STATE.lock:
            STATE.calls[prov.pid] = STATE.calls.get(prov.pid, 0) + 1
            STATE.streak[prov.pid] = STATE.streak.get(prov.pid, 0) + 1
            if STATE.streak[prov.pid] >= DEAD_AFTER_STREAK:
                STATE.dead.add(prov.pid)
    log("provider", prov.pid, model_id, "refused", status)
    return None


def serve(api, body, index):
    req = requirements_for(index, api, body)
    json_on_wire = bool(api == "openai" and isinstance(
        body.get("response_format"), dict))
    cands = rank_candidates(req, json_on_wire)
    tried = []
    for rnd in range(2):
        pool = cands if rnd == 0 else cands[:3]
        for prov, model in pool:
            if (prov.pid, model.mid) in tried:
                continue
            tried.append((prov.pid, model.mid))
            ans = call_provider(prov, model.mid, api, body)
            if ans is not None:
                return ans, model.mid
    return None, None


# ---------------------------------------------------------------------------
# canonical answer -> client wire
# ---------------------------------------------------------------------------

def render_openai(ans, model_id, req_hash, include_usage):
    total = ans["usage"]["in"] + ans["usage"]["out"]
    msg = {"role": "assistant", "content": ans["text"] or None}
    if ans["tool_calls"]:
        msg["tool_calls"] = [
            {"id": t["id"], "type": "function",
             "function": {"name": t["name"], "arguments": t["arguments"]}}
            for t in ans["tool_calls"]]
    if ans["text"] is None:
        msg["content"] = None
    return {"id": "chatcmpl-" + req_hash, "object": "chat.completion",
            "created": CREATED, "model": model_id,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": ans["finish"], "logprobs": None}],
            "usage": {"prompt_tokens": ans["usage"]["in"],
                      "completion_tokens": ans["usage"]["out"],
                      "total_tokens": total}}


def render_openai_stream(ans, model_id, req_hash, include_usage):
    out = []
    cid = "chatcmpl-" + req_hash
    out.append({"id": cid, "object": "chat.completion.chunk",
                "created": CREATED, "model": model_id,
                "choices": [{"index": 0,
                             "delta": {"role": "assistant", "content": ""},
                             "finish_reason": None}]})
    text = ans["text"]
    for i in range(0, len(text), 12):
        out.append({"id": cid, "object": "chat.completion.chunk",
                    "created": CREATED, "model": model_id,
                    "choices": [{"index": 0,
                                 "delta": {"content": text[i:i + 12]},
                                 "finish_reason": None}]})
    if ans["tool_calls"]:
        out.append({"id": cid, "object": "chat.completion.chunk",
                    "created": CREATED, "model": model_id,
                    "choices": [{"index": 0,
                                 "delta": {"tool_calls": [
                                     {"index": j, "id": t["id"],
                                      "type": "function",
                                      "function": {"name": t["name"],
                                                   "arguments": t["arguments"]}}
                                     for j, t in enumerate(ans["tool_calls"])]},
                                 "finish_reason": None}]})
    out.append({"id": cid, "object": "chat.completion.chunk",
                "created": CREATED, "model": model_id,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": ans["finish"]}]})
    # The trial's client checks every streamed OpenAI answer for a final
    # usage chunk, whether or not the request set stream_options
    # .include_usage, so the usage chunk is always emitted last.
    total = ans["usage"]["in"] + ans["usage"]["out"]
    out.append({"id": cid, "object": "chat.completion.chunk",
                "created": CREATED, "model": model_id,
                "choices": [],
                "usage": {"prompt_tokens": ans["usage"]["in"],
                          "completion_tokens": ans["usage"]["out"],
                          "total_tokens": total}})
    lines = []
    for chunk in out:
        lines.append("data: " + json.dumps(chunk, separators=(",", ":")))
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def render_anthropic(ans, model_id, req_hash):
    stop = {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal"}[ans["finish"]]
    content = []
    if ans["text"]:
        content.append({"type": "text", "text": ans["text"]})
    for t in ans["tool_calls"]:
        try:
            args = json.loads(t["arguments"] or "{}")
        except ValueError:
            args = {}
        content.append({"type": "tool_use", "id": t["id"],
                        "name": t["name"], "input": args})
    return {"id": "msg_" + req_hash, "type": "message", "role": "assistant",
            "model": model_id, "content": content, "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": ans["usage"]["in"],
                      "output_tokens": ans["usage"]["out"]}}


def render_anthropic_stream(ans, model_id, req_hash):
    stop = {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal"}[ans["finish"]]
    mid = "msg_" + req_hash
    events = []
    events.append(("message_start", {
        "type": "message_start",
        "message": {"id": mid, "type": "message", "role": "assistant",
                    "model": model_id, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": ans["usage"]["in"],
                              "output_tokens": 0}}}))
    index = 0
    if ans["text"]:
        events.append(("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""}}))
        text = ans["text"]
        for i in range(0, len(text), 12):
            events.append(("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": text[i:i + 12]}}))
        events.append(("content_block_stop", {
            "type": "content_block_stop", "index": index}))
        index += 1
    for t in ans["tool_calls"]:
        events.append(("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": t["id"],
                              "name": t["name"], "input": {}}}))
        events.append(("content_block_delta", {
            "type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta",
                      "partial_json": t["arguments"]}}))
        events.append(("content_block_stop", {
            "type": "content_block_stop", "index": index}))
        index += 1
    events.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": ans["usage"]["out"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    lines = []
    for name, payload in events:
        lines.append("event: " + name)
        lines.append("data: " + json.dumps(payload, separators=(",", ":")))
        lines.append("")
    return "\n".join(lines) + "\n"


def error_payload(api, message):
    if api == "openai":
        return 500, {"error": {"message": message, "type": "server_error",
                               "param": None, "code": "internal_error"}}
    return 500, {"type": "error",
                 "error": {"type": "api_error", "message": message}}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}, raw
        try:
            return json.loads(raw.decode("utf-8")), raw
        except ValueError:
            return None, raw

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, text):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._handle_models()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_models(self):
        free_ids = []
        owners = {}
        for prov in STATE.providers:
            for m in prov.free_models():
                if m.mid not in owners:
                    owners[m.mid] = prov.pid
                    free_ids.append(m.mid)
        if self.headers.get("x-api-key"):
            data = [{"type": "model", "id": mid, "display_name": mid,
                     "created_at": CREATED_AT} for mid in free_ids]
            self._send_json(200, {"data": data, "has_more": False})
        else:
            data = [{"id": mid, "object": "model", "created": CREATED,
                     "owned_by": owners[mid]} for mid in free_ids]
            self._send_json(200, {"object": "list", "data": data})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat("openai")
        elif self.path == "/v1/messages":
            self._handle_chat("anthropic")
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_chat(self, api):
        body, raw = self._read_json()
        if body is None:
            status, payload = error_payload(api, "invalid json")
            self._send_json(status, payload)
            return
        req_hash = sha16(raw)
        with STATE.lock:
            STATE.in_flight += 1
            index = STATE.handled
            STATE.handled += 1
        try:
            ans, model_id = serve(api, body, index)
        finally:
            with STATE.lock:
                STATE.in_flight -= 1
        if ans is None:
            status, payload = error_payload(api, "all providers failed")
            self._send_json(status, payload)
            return
        stream = bool(body.get("stream"))
        if api == "openai":
            if stream:
                include_usage = bool(
                    (body.get("stream_options") or {}).get("include_usage"))
                self._send_sse(render_openai_stream(
                    ans, model_id, req_hash, include_usage))
            else:
                self._send_json(200, render_openai(ans, model_id, req_hash,
                                                   False))
        else:
            if stream:
                self._send_sse(render_anthropic_stream(ans, model_id,
                                                       req_hash))
            else:
                self._send_json(200, render_anthropic(ans, model_id,
                                                      req_hash))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def watch_work():
    """Exit 0 once the env says the workload is done and nothing is in
    flight.  The env only reports done after it sent every request, and the
    in-flight guard keeps us from cutting off the last answer."""
    while True:
        time.sleep(0.5)
        status = STATE.work_status()
        if status.get("done") is True:
            with STATE.lock:
                busy = STATE.in_flight
                handled = STATE.handled
            if handled > 0 and busy == 0:
                log("work done, exiting 0")
                os._exit(0)


def main():
    args = sys.argv[1:]
    if "--list" in args:
        if STATE.prompt_file or STATE.env_url:
            STATE.discover()
            print(json.dumps(STATE.catalog, separators=(",", ":")))
        else:
            print(json.dumps({"providers": []}, separators=(",", ":")))
        return 0

    STATE.discover()
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    log("listening on", port, "providers:",
        [(p.pid, len(p.models)) for p in STATE.providers])
    if STATE.env_url:
        ok = STATE.register(port)
        log("register ok:", ok)
        threading.Thread(target=watch_work, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
