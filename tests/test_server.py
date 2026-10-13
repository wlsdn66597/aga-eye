import base64
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import server.app as srv  # noqa: E402


@pytest.fixture
def client():
    srv.state = srv.State()
    with TestClient(srv.app) as c:
        yield c


def start(client, event_id="e1", device="crib-01"):
    return client.post(f"/devices/{device}/events", json={
        "event_id": event_id, "kind": "risk_started", "risk_type": "prone",
        "started_at": time.time() - 5, "confirmed_at": time.time(),
        "snapshot_jpeg_b64": base64.b64encode(b"\xff\xd8fakejpeg").decode()})


def test_heartbeat_returns_settings_and_status(client):
    r = client.post("/devices/crib-01/heartbeat", json={"status": "normal", "settings_version": 1})
    assert r.json()["settings"]["risk_threshold_s"] == 5.0
    s = client.get("/devices/crib-01/status").json()
    assert s["online"] and s["status"] == "normal" and s["settings_applied"]


def test_settings_change_bumps_version(client):
    client.post("/devices/crib-01/heartbeat", json={"settings_version": 1})
    s = client.put("/devices/crib-01/risk-settings", json={"risk_threshold_s": 8, "escalation_s": 15}).json()
    assert s["version"] == 2
    assert client.post("/devices/crib-01/heartbeat", json={}).json()["settings"]["risk_threshold_s"] == 8
    assert not client.get("/devices/crib-01/status").json()["settings_applied"]


def test_risk_event_creates_first_alert_and_is_idempotent(client):
    assert start(client).status_code == 200
    assert start(client).json().get("duplicate")
    alerts = client.get("/alerts").json()
    assert [a["level"] for a in alerts] == [1]
    assert client.get("/events/e1/snapshot").content.startswith(b"\xff\xd8")


def test_second_alert_when_not_acknowledged(client):
    client.put("/devices/crib-01/risk-settings", json={"risk_threshold_s": 5, "escalation_s": 0.2})
    start(client)
    time.sleep(0.5)
    levels = [a["level"] for a in client.get("/alerts").json()]
    assert 2 in levels
    assert client.get("/events").json()[0]["escalated"]


def test_no_second_alert_after_ack(client):
    client.put("/devices/crib-01/risk-settings", json={"risk_threshold_s": 5, "escalation_s": 0.2})
    start(client)
    ev = client.post("/events/e1/ack", json={"guardian": "엄마"}).json()
    assert ev["acked"] and ev["status"] == "active"  # 확인해도 위험은 계속
    time.sleep(0.5)
    assert 2 not in [a["level"] for a in client.get("/alerts").json()]


def test_no_second_alert_when_risk_ended_first(client):
    client.put("/devices/crib-01/risk-settings", json={"risk_threshold_s": 5, "escalation_s": 0.3})
    start(client)
    client.post("/devices/crib-01/events", json={"event_id": "e1", "kind": "risk_ended",
                                                  "risk_type": "prone", "ended_at": time.time()})
    time.sleep(0.6)
    alerts = client.get("/alerts").json()
    assert 2 not in [a["level"] for a in alerts]
    assert "미확인" in alerts[0]["message"]


def test_connection_lost_alert(client, monkeypatch):
    client.post("/devices/crib-01/heartbeat", json={"status": "normal"})
    srv.state.devices["crib-01"]["last_seen"] -= srv.OFFLINE_AFTER_S + 1
    time.sleep(1.5)  # watchdog은 1초마다 확인
    assert client.get("/alerts").json()[0]["kind"] == "connection_lost"
    client.post("/devices/crib-01/heartbeat", json={"status": "normal"})
    assert client.get("/alerts").json()[0]["kind"] == "connection_restored"
