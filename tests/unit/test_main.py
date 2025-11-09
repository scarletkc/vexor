import vexor.__main__ as vexor_main


def test_main_invokes_run(monkeypatch):
    called = {}

    def fake_run():
        called["ok"] = True

    monkeypatch.setattr("vexor.__main__.run", fake_run)

    vexor_main.main()

    assert called["ok"] is True
