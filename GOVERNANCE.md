# Governance

## Status

`civic-slm` is currently a **single-maintainer** project (BDFL model). Final
decisions on scope, releases, and merges rest with `@itsmeduncan`. This will
change as the maintainer team grows; this document will be updated when it
does.

## How decisions get made

1. **Bug fixes, documentation improvements, new recipes:** open a PR. A
   single maintainer review is sufficient. New recipes additionally require
   a `docs/SOURCES.md` audit entry.
2. **Schema changes, eval-harness changes, training-pipeline changes:**
   open an issue first to align on direction, then a PR. The PR description
   must explain the migration story for any consumer.
3. **Scope changes, license changes, security-policy changes:** require an
   RFC issue tagged `rfc` open for at least 7 days before a PR lands.

## Roadmap

See `ROADMAP.md`.

## Maintainer responsibilities

- Triage issues within 7 days.
- Respond to security reports within 3 days (`SECURITY.md`).
- Cut a release at least once per quarter while pre-1.0.
- Keep `CHANGELOG.md` honest.

## Issue triage and labels

Every new issue gets at least one of the labels below within 7 days:

| Label              | Meaning                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `bug`              | Something is broken vs. documented behavior.                            |
| `enhancement`      | A feature request that fits the roadmap.                                |
| `recipe`           | New jurisdiction request or recipe-related bug.                         |
| `eval`             | Eval-harness, scorers, benchmarks.                                      |
| `train`            | Training pipeline (CPT/SFT/DPO/merge/quantize).                         |
| `docs`             | Documentation, glossary, examples.                                      |
| `security`         | Security-relevant; cross-references the private advisory if any.        |
| `good first issue` | Scoped tightly enough that a first-time contributor can land it.        |
| `help wanted`      | Maintainer is unlikely to do it soon; PRs welcome.                      |
| `wontfix`          | Out of scope — see `ARCHITECTURE.md` "Out of scope" + `ROADMAP.md`.     |
| `needs-repro`      | Bug report missing repro steps; will close after 30 days if not filled. |

PRs get the same labels plus `breaking` for anything that requires a
SemVer major bump per `RELEASING.md`.

## Communication channels

While pre-1.0, the project deliberately keeps surface area small:

- **GitHub Issues** — bug reports, recipe requests, feature requests.
  Templates under `.github/ISSUE_TEMPLATE/`.
- **GitHub Discussions** (planned, not yet enabled) — long-form
  architecture questions, recipe-pattern feedback, model-card
  retrospectives. Until enabled, file an issue with the `discussion`
  label and the maintainer will move it.
- **Security** — private advisories per `SECURITY.md`. Do not file
  security bugs in public Issues or Discussions.
- **Real-time chat** — none. The project is small enough that async
  github threads are sufficient; introducing a Slack/Discord/Matrix
  before there's sustained volume is more cost than benefit.

## Trademark and naming

The MIT license covers the code in this repository. It does **not**
grant any right to use the name "civic-slm" or any associated
identifier in a way that implies affiliation, endorsement, or
maintainer status. Forks are welcome — please:

- Make it visible in your README that the project is a fork.
- Choose a fork name that is not confusable with `civic-slm` (e.g.,
  `civic-slm-yourorg` is fine; `civic-slm-official` is not).
- Do not publish to PyPI, HF Hub, or other registries under a name
  that suggests this is the upstream project.

The project name was chosen on the basis of a 2026-04-25 search of
PyPI, HF Hub, and the GitHub `civic` topic and was not in conflict
with any prior project at that time. If you believe it conflicts with
yours, contact the maintainer at `itsmeduncan@gmail.com` and we will
discuss in good faith.

City and jurisdiction names that appear in the corpus, recipes, or
documentation (San Clemente, etc.) are referenced for the purpose of
public-records analysis only. The project is **not affiliated with**
and does not represent any city, county, township, school district, or
other government body.

## Becoming a maintainer

Maintainership is by invitation. Sustained, high-quality contributions over
multiple releases — code, recipes, documentation, or community work — are
the path. There is no fixed metric.

## Funding and conflicts of interest

The project takes no money today. If that changes, this section will be
updated to disclose funders, the nature of the support, and any restrictions
they place on roadmap decisions.

## Forking

The MIT license expressly permits forking. We ask (but do not legally require)
that forks:

- Make it clear in their README that they are forks.
- Honor `ACCEPTABLE_USE_POLICY.md` even though the MIT license does not bind
  them to it.
- Do not use the `civic-slm` name for the fork in a way that implies
  affiliation.
