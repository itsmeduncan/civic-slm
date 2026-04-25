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
