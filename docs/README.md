# SupoClip Documentation

This directory is the canonical documentation hub for SupoClip.

**Want to clip videos with Claude Code and no API keys?** Go straight to the [Claude-native guide](./claude-native/README.md): install, `/clip`, `/clip-edit`, options and troubleshooting.

If you are new to the project, start here:

1. Read [Setup](./setup.md) to get the app running.
2. Review [Configuration](./configuration.md) to understand environment variables and operating modes.
3. Use [App Guide](./app-guide.md) to learn the user-facing parts of the product.
4. Use [Architecture](./architecture.md) to understand how the system works end to end.
5. Use [Troubleshooting](./troubleshooting.md) when something goes wrong.

## Documentation Map

- [Claude Code mode](./claude-native/README.md)
  - Install (macOS, Linux) and Claude Code setup
  - `/clip` and `/clip-edit` usage and options
  - Run folders, B-roll, Whisper model choice
  - The `clipper` CLI, settings, troubleshooting, limitations
- [Setup](./setup.md)
  - Docker-first installation
  - Local development commands
  - First-run checklist
  - Production-minded setup notes
- [Configuration](./configuration.md)
  - Required API keys
  - DataFast analytics settings
  - Processing modes
  - Auth and monetization settings
  - YouTube auth rotation settings
  - Twitch and Kick import providers
  - Feedback and email configuration
- [App Guide](./app-guide.md)
  - Main screens and routes
  - Core user workflows
  - Admin features
  - Hosted versus self-host differences
- [Architecture](./architecture.md)
  - Frontend, backend, worker, Redis, PostgreSQL
  - Queue and SSE progress flow
  - Video processing pipeline
  - Database model overview
- [API Reference](./api-reference.md)
  - Frontend proxy routes
  - Backend endpoints
  - Admin and billing endpoints
  - Notes on auth and streaming
- [Development](./development.md)
  - Repository layout
  - Commands for each app
  - Common workflows
  - Where to modify major features
- [Troubleshooting](./troubleshooting.md)
  - Startup failures
  - Stuck tasks
  - Auth, fonts, billing, and YouTube issues
  - Performance and recovery guidance

## What SupoClip Is

SupoClip is an open-source AI video clipping application. It takes long-form videos, transcribes them, uses an LLM to select the most promising short segments, and renders vertical or source-aspect clips with subtitles and optional effects.

It can also run without the web app: in **Claude-native mode**, Claude Code does the AI steps itself and the `clipper` CLI runs transcription and rendering locally.

The current repository snapshot includes:

- `frontend/`: the main Next.js application
- `backend/`: the FastAPI API and ARQ worker code, plus the `clipper` CLI (`backend/src/native/`)
- `.claude/skills/`: the `/clip` and `/clip-edit` Claude Code skills
- Root-level infrastructure files such as `docker-compose.yml`, `init.sql`, `.env.example`, and `start.sh`

## Recommended Reading Paths

For clipping with Claude Code (no Docker, no API keys):

1. [Claude-native guide](./claude-native/README.md)

For operators:

1. [Setup](./setup.md)
2. [Configuration](./configuration.md)
3. [Troubleshooting](./troubleshooting.md)

For developers:

1. [Development](./development.md)
2. [Architecture](./architecture.md)
3. [API Reference](./api-reference.md)

For product and support:

1. [App Guide](./app-guide.md)
2. [Troubleshooting](./troubleshooting.md)

## Existing Documentation Outside `docs/`

This new docs tree replaces the need to hunt across several markdown files, but these older documents still contain useful context:

- [`README.md`](../README.md)
- [`QUICKSTART.md`](../QUICKSTART.md)
- [`CLAUDE.md`](../CLAUDE.md)
- [`REFACTORING_COMPLETE.md`](../REFACTORING_COMPLETE.md)
- [`backend/REFACTORING_GUIDE.md`](../backend/REFACTORING_GUIDE.md)
