# Vexor Claude Code Plugin

This plugin ships the `vexor-cli` Agent Skill so Claude Code can autonomously use Vexor for
semantic file discovery.

## What's included

- Agent Skill: `skills/vexor-cli/`

## Install

Install this plugin via any Claude Code marketplace you use by pointing its plugin `source` to this
folder (`./plugins/vexor` in this repo).

If you only want the `vexor-cli` Agent Skill (without installing a plugin), you can install it with:
```bash
vexor install --skills claude/codex
```

## Use

Ask Claude to find files by intent, and it can invoke the `vexor-cli` skill automatically.
You can also explicitly request it:

> Use the vexor-cli skill to find where config is loaded.
