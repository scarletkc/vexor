from vexor.text import Messages


def test_app_help_uses_ascii_dash():
    assert Messages.APP_HELP == "Vexor - A vector-powered CLI for semantic search over files."
