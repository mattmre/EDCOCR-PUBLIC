<!--
Thanks for sending a pull request. Filling in the sections below makes
review faster and helps the change land cleanly.

If this is a small docs/typo fix, you can delete most of this template
and just describe what you changed.
-->

## What changed and why

<!-- One or two paragraphs. Focus on the "why", not just the "what". -->

## How it was tested

<!--
- Unit/integration tests added or updated (which files?)
- Manual verification steps (commands run, fixtures used)
- CI evidence
-->

## Risks and breaking changes

<!--
- What could break? Who is affected?
- Did you change a default feature flag, env var contract, output schema,
  or API surface? Flag it explicitly.
- If yes, does this need a CHANGELOG.md entry under a new release?
-->

## Documentation

<!--
Check all that apply:

- [ ] No docs needed (code-internal change)
- [ ] Updated `docs/06-CONFIGURATION-REFERENCE.md` (env vars / feature flags)
- [ ] Updated `docs/API-REFERENCE.md` (API endpoints)
- [ ] Updated `docs/08-SDK-REFERENCE.md` (SDK surface)
- [ ] Updated `ARCHITECTURE.md` (deployment topology)
- [ ] Updated `CHANGELOG.md` (user-visible runtime change)
- [ ] Other: ___
-->

## Checklist

- [ ] Tests added/updated and passing locally (`make test` or targeted `pytest`)
- [ ] `ruff check .` is clean for the files I touched
- [ ] Branch is rebased on current `main`
- [ ] PR title is short and descriptive (will become the squash commit subject)
- [ ] I have not included AI/LLM co-author footers (`Co-Authored-By: Claude` etc.)

## Linked issues

<!-- "Closes #123", "Refs #456" — link related issues here. -->
