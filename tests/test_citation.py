"""The reminder of the works to cite."""

from encircle import Encircle, citation


def test_the_reminder_is_printed_once(capsys, monkeypatch):
    monkeypatch.delenv("DRONE_ENCIRCLE_CITE", raising=False)
    monkeypatch.setattr(citation, "_reminded", False)
    Encircle()
    note = capsys.readouterr().err
    assert "please cite" in note and "particlegym" in note and "Lawrence" in note
    Encircle()
    assert capsys.readouterr().err == ""


def test_the_reminder_can_be_turned_off(capsys, monkeypatch):
    monkeypatch.setenv("DRONE_ENCIRCLE_CITE", "0")
    monkeypatch.setattr(citation, "_reminded", False)
    Encircle()
    assert capsys.readouterr().err == ""
