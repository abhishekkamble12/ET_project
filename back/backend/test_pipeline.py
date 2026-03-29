"""
5 end-to-end pipeline test queries.
Tests: knowledge → strategy → content → compliance → engagement → localization → formatter
Human-review node is bypassed via /stream (pauses at WAITING_HUMAN) then auto-approved.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/json",
    "X-Company-ID": "demo-corp",
}

QUERIES = [
    {"query": "Launch a new AI-powered analytics dashboard for enterprise teams", "platform": "linkedin", "locale": "en"},
    {"query": "Summer fitness challenge campaign to boost gym memberships", "platform": "instagram", "locale": "en"},
    {"query": "Announce a sustainability initiative reducing carbon footprint by 40%", "platform": "linkedin", "locale": "en"},
    {"query": "New mobile app feature: real-time collaboration for remote teams", "platform": "instagram", "locale": "en"},
    {"query": "B2B SaaS product launch targeting fintech startups", "platform": "linkedin", "locale": "en"},
]


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()}


def stream_until_waiting(path, body):
    """Stream SSE and return (session_id, preview) when WAITING_HUMAN or complete."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=HEADERS, method="POST")
    session_id = None
    preview = {}
    agents_done = []

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            buffer = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n\n" in buffer:
                    event_bytes, buffer = buffer.split(b"\n\n", 1)
                    event_str = event_bytes.decode("utf-8", errors="replace")
                    lines = event_str.strip().split("\n")
                    etype, edata = "message", ""
                    for line in lines:
                        if line.startswith("event:"):
                            etype = line[6:].strip()
                        if line.startswith("data:"):
                            edata += line[5:].strip()
                    if not edata:
                        continue
                    try:
                        payload = json.loads(edata)
                    except Exception:
                        continue

                    if not session_id:
                        session_id = payload.get("session_id")

                    if etype == "agent_update":
                        agents_done.append(payload.get("node", "?"))
                        print(f"    ✓ {payload.get('node')}")

                    elif etype == "WAITING_HUMAN":
                        preview = payload.get("preview", {})
                        print(f"    ⏸  WAITING_HUMAN – session {session_id}")
                        return session_id, preview, agents_done, "waiting"

                    elif etype == "complete":
                        print(f"    ✅ complete (no human review needed)")
                        return session_id, preview, agents_done, "complete"

                    elif etype == "error":
                        print(f"    ❌ pipeline error: {payload.get('detail')}")
                        return session_id, preview, agents_done, "error"

    except Exception as exc:
        print(f"    ❌ stream exception: {exc}")
        return session_id, preview, agents_done, "exception"

    return session_id, preview, agents_done, "done"


def approve(session_id, decision="publish"):
    body = {"session_id": session_id, "decision": decision}
    return post("/api/v1/approve", body)


def run():
    print("=" * 65)
    print("  EngageTech AI Pipeline – 5 Query End-to-End Test")
    print("=" * 65)

    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/5] {q['query'][:60]}...")
        print(f"      platform={q['platform']}  locale={q['locale']}")
        t0 = time.monotonic()

        session_id, preview, agents, status = stream_until_waiting("/api/v1/stream", q)

        if status == "waiting" and session_id:
            print(f"    → auto-approving (publish)...")
            result = approve(session_id, "publish")
            elapsed = round(time.monotonic() - t0, 1)
            if "error" in result:
                print(f"    ❌ approve failed: {result}")
                results.append({"query": i, "status": "FAIL", "error": result})
            else:
                caption = None
                gp = result.get("generated_post") or {}
                if isinstance(gp, dict):
                    caption = gp.get("caption", "")
                print(f"    ✅ PASS  ({elapsed}s)")
                if caption:
                    print(f"    📝 Caption preview: {str(caption)[:120]}...")
                results.append({"query": i, "status": "PASS", "elapsed": elapsed, "agents": agents})
        elif status == "complete":
            elapsed = round(time.monotonic() - t0, 1)
            print(f"    ✅ PASS  ({elapsed}s)")
            results.append({"query": i, "status": "PASS", "elapsed": elapsed, "agents": agents})
        else:
            elapsed = round(time.monotonic() - t0, 1)
            print(f"    ❌ FAIL  status={status}  ({elapsed}s)")
            results.append({"query": i, "status": "FAIL", "elapsed": elapsed})

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    passed = sum(1 for r in results if r["status"] == "PASS")
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        agents_str = f"  agents={len(r.get('agents', []))}" if r.get("agents") else ""
        print(f"  {icon} Query {r['query']}  {r['status']}  {r.get('elapsed', '?')}s{agents_str}")
    print(f"\n  {passed}/5 passed")
    print("=" * 65)
    return passed


if __name__ == "__main__":
    passed = run()
    sys.exit(0 if passed == 5 else 1)
