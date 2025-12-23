# Vexor Desktop (Electron + Vue 3)

## Development

```bash
cd gui
npm install
npm run dev
```

This starts Vite on port 5173 and launches Electron.

## Production build

```bash
npm run build
npm run start
```

`npm run start` loads the built renderer from `gui/dist`.

## CLI path

The UI calls the `vexor` CLI. If it is not on `PATH`, set a custom path
in the Config screen (stored locally in the renderer).

## CLI downloads

The GUI can download the standalone CLI into the Electron user data folder:
`app.getPath('userData')/cli/` (auto-used when present).

## Forge packaging

```bash
npm run package
npm run make
```

Notes:
- This build currently produces ZIP packages only (Windows/Linux).
- Installer makers are commented out in `gui/forge.config.js`.
