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
