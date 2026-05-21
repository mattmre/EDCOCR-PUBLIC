# Getting Help with EDCOCR

Thanks for using EDCOCR. Here is the fastest way to get a useful answer for the kind of question you have.

## Bugs, regressions, or unexpected behavior

[Open an issue](https://github.com/mattmre/EDCOCR-PUBLIC/issues/new?template=bug_report.yml) using the **Bug report** template. Include the version, deployment method, a minimal reproduction, and the relevant log lines (redact any sensitive content first).

## Feature ideas or enhancement requests

[Open an issue](https://github.com/mattmre/EDCOCR-PUBLIC/issues/new?template=feature_request.yml) using the **Feature request** template. Lead with the problem you are trying to solve, not the implementation.

## Questions, "how do I…", or general discussion

Start in [Discussions](https://github.com/mattmre/EDCOCR-PUBLIC/discussions). It is the right place for:

- Open-ended Q&A and design conversations
- Sharing your deployment setup
- Asking for advice on configuration, scaling, or integration
- General community chat

If you have a sharp, version-tied question you want surfaced in the issue tracker for future searchers, [open a Question issue](https://github.com/mattmre/EDCOCR-PUBLIC/issues/new?template=question.yml) instead.

## Security vulnerabilities

**Do not open a public issue.** Report security issues privately via [GitHub Security Advisories](https://github.com/mattmre/EDCOCR-PUBLIC/security/advisories/new). See [`SECURITY.md`](SECURITY.md) for the full disclosure policy and response timeline.

## Joining the team as a contributor

Send a direct message to [**@mattmre**](https://github.com/mattmre) on GitHub. See [§10 "Joining the Team"](CONTRIBUTING.md#10-joining-the-team) in `CONTRIBUTING.md` for what to include in your introduction.

## Commercial support or consulting

Reach out via DM to [**@mattmre**](https://github.com/mattmre) on GitHub.

---

## Before you ask

A few things that almost always help:

- **Read the version you are running** — `python -c "import version; print(version.__version__)"` or check the Helm chart `appVersion`. Old versions often have fixed bugs.
- **Check [`CHANGELOG.md`](CHANGELOG.md)** for recent changes that may be relevant.
- **Check [`docs/09-TROUBLESHOOTING.md`](docs/09-TROUBLESHOOTING.md)** for the common deployment and runtime issues.
- **Search existing issues and Discussions** — your question may already be answered.
