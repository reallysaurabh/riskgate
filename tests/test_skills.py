"""The shipped agent skills (skills/*): spec-valid frontmatter, consistent manifests, and scripts that run."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import riskgate as rg

ROOT = Path(__file__).resolve().parents[1]
SKILLS = sorted(p for p in (ROOT / "skills").iterdir() if (p / "SKILL.md").exists())
NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def frontmatter(path: Path) -> dict:
    text = path.read_text()
    assert text.startswith("---\n"), path
    block = text.split("---\n", 2)[1]
    out, key = {}, None
    for line in block.splitlines():
        if line.startswith("  ") and key:
            out.setdefault(key + ".", []).append(line.strip())
        elif ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_skill_frontmatter_follows_the_agent_skills_spec(skill):
    fm = frontmatter(skill / "SKILL.md")
    assert fm["name"] == skill.name and NAME.match(fm["name"]) and len(fm["name"]) <= 64
    assert "claude" not in fm["name"] and "anthropic" not in fm["name"]
    assert 0 < len(fm["description"]) <= 200                     # claude.ai's upload limit (the spec allows 1024)
    assert len(fm.get("compatibility", "")) <= 500
    assert len((skill / "SKILL.md").read_text().splitlines()) < 500
    assert f'version: "{rg.__version__}"' in (skill / "SKILL.md").read_text()
    for ref in re.findall(r"`((?:scripts|references)/[^`\s]+)`", (skill / "SKILL.md").read_text()):
        assert (skill / ref).exists(), f"{skill.name}: SKILL.md names a missing file {ref}"


def test_manifests_agree_with_the_package():
    version = rg.__version__.replace("rc", "-rc")                 # semver spelling for plugin manifests
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    gemini = json.loads((ROOT / "gemini-extension.json").read_text())
    entry = next(p for p in market["plugins"] if p["name"] == plugin["name"])
    assert plugin["version"] == entry["version"] == gemini["version"] == version
    assert entry["source"] == "./" and gemini["name"] == plugin["name"] == "riskgate"


def test_quickstart_skill_script_shows_every_verdict():
    out = subprocess.run([sys.executable, "quickstart_gate.py"], cwd=ROOT / "skills" / "riskgate" / "scripts",
                         capture_output=True, text=True, timeout=300, check=True).stdout
    for verdict in ("PROBE", "CERTIFY", "ASK"):
        assert verdict in out
    assert "[valid]" in out


def test_inspect_traces_script(tmp_path):
    sys.path.insert(0, str(ROOT / "examples" / "use_cases"))
    import from_traces
    from_traces.simulate(300, tmp_path)
    script = ROOT / "skills" / "riskgate-from-traces" / "scripts" / "inspect_traces.py"
    ok = subprocess.run([sys.executable, str(script), str(tmp_path / "traces.jsonl"), str(tmp_path / "outcomes.csv"),
                         "--json"], capture_output=True, text=True, timeout=120)
    assert ok.returncode == 0, ok.stderr
    rep = json.loads(ok.stdout)
    assert rep["tools"]["issue_refund"]["kind"] == "write" and rep["labels"]["labelled"] == 300
    assert rep["supports_alpha"]["0.05"]["possible"]
    no_outcomes = subprocess.run([sys.executable, str(script), str(tmp_path / "traces.jsonl")],
                                 capture_output=True, text=True, timeout=120)
    assert no_outcomes.returncode == 1 and "cannot say which actions were unsafe" in no_outcomes.stdout
