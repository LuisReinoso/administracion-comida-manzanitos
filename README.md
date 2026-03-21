# Manzanitos 🍎

A minimal, elegant **food expiration tracker** built as a single-file PWA that uses [Todoist](https://todoist.com) as its backend.

Track what's in your pantry, get alerts before things expire, and keep everything synced across devices — all without a server.

## Features

- **Zero backend** — Todoist API handles all storage and sync
- **Real-time sync** — Multiple users can share the same pantry with one Todoist token
- **Smart alerts** — Tasks are created in Todoist 3 days before expiration
- **Status grouping** — Products sorted by: fresh, expiring soon, urgent, expired
- **PWA** — Installable on mobile, works offline for cached data
- **i18n** — English and Spanish with persistent language preference
- **Single file** — One `index.html`, no build step, no dependencies

## Getting Started

### 1. Get your Todoist API token

1. Go to [todoist.com/app/settings/integrations/developer](https://app.todoist.com/app/settings/integrations/developer)
2. Copy your **API token**

### 2. Open the app

Visit the hosted version or open `index.html` locally.

### 3. Paste your token

The app will create a **Manzanitos** project in your Todoist account and you're ready to go.

> Your token is stored in `localStorage` and never leaves your device.

## Install as PWA

### Android (Chrome)
1. Open the app URL in Chrome
2. Tap **⋮** → **Add to Home screen**
3. The app will behave like a native app

### iOS (Safari)
1. Open the app URL in Safari
2. Tap **Share** → **Add to Home Screen**

## How It Works

Each product is stored as a Todoist task with the format:

```
🥦 Broccoli — vence: 2025-07-15
```

The task's due date is set to **3 days before** the expiration date, so Todoist sends you a reminder before it's too late.

## Tech Stack

- HTML + CSS + JS (vanilla)
- Todoist REST API v2
- Inline PWA manifest and service worker
- Google Fonts (Playfair Display + DM Mono)

## License

MIT
