const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const initSessions = new Map();
let nextInitId = 1;

function createWindow() {
  const win = new BrowserWindow({
    width: 1240,
    height: 820,
    minWidth: 980,
    minHeight: 640,
    backgroundColor: "#f5eee4",
    title: "Vexor Desktop",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.js")
    }
  });

  const devServerUrl = process.env.VITE_DEV_SERVER_URL;
  const indexPath = path.join(__dirname, "..", "dist", "index.html");
  if (devServerUrl) {
    win.loadURL(devServerUrl);
  } else if (fs.existsSync(indexPath)) {
    win.loadFile(indexPath);
  } else {
    win.loadURL("http://localhost:5173");
  }
}

function resolveCliPath(cliPath) {
  if (!cliPath) {
    return "vexor";
  }
  const trimmed = cliPath.trim();
  if (!trimmed) {
    return "vexor";
  }
  if (trimmed.startsWith("~")) {
    return path.join(os.homedir(), trimmed.slice(1));
  }
  return trimmed;
}

function runVexorCommand({ cliPath, args, input }) {
  return new Promise((resolve) => {
    const resolvedPath = resolveCliPath(cliPath);
    const child = spawn(resolvedPath, args, {
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";

    if (input) {
      child.stdin.write(input);
    }
    child.stdin.end();

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    child.on("error", (error) => {
      resolve({
        code: null,
        stdout,
        stderr,
        error: error.message
      });
    });
    child.on("close", (code) => {
      resolve({
        code,
        stdout,
        stderr,
        error: null
      });
    });
  });
}

function getConfigInfo() {
  const configPath = path.join(os.homedir(), ".vexor", "config.json");
  const exists = fs.existsSync(configPath);
  let config = {};
  let parseError = null;
  if (exists) {
    try {
      config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    } catch (error) {
      parseError = error.message;
    }
  }
  return { path: configPath, exists, config, parseError };
}

app.whenReady().then(() => {
  createWindow();

  ipcMain.handle("vexor:select-directory", async () => {
    const result = await dialog.showOpenDialog({
      properties: ["openDirectory", "createDirectory"]
    });
    if (result.canceled || result.filePaths.length === 0) {
      return null;
    }
    return result.filePaths[0];
  });

  ipcMain.handle("vexor:run", async (_event, payload) => {
    return runVexorCommand(payload);
  });

  ipcMain.handle("vexor:config-info", async () => {
    return getConfigInfo();
  });

  ipcMain.handle("vexor:init:start", async (event, { cliPath }) => {
    const resolvedPath = resolveCliPath(cliPath);
    const initId = nextInitId++;
    const child = spawn(resolvedPath, ["init"], {
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
      windowsHide: true
    });
    initSessions.set(initId, child);

    child.stdout.on("data", (data) => {
      event.sender.send("vexor:init:output", {
        id: initId,
        stream: "stdout",
        chunk: data.toString()
      });
    });

    child.stderr.on("data", (data) => {
      event.sender.send("vexor:init:output", {
        id: initId,
        stream: "stderr",
        chunk: data.toString()
      });
    });

    child.on("close", (code) => {
      event.sender.send("vexor:init:exit", { id: initId, code });
      initSessions.delete(initId);
    });

    child.on("error", (error) => {
      event.sender.send("vexor:init:output", {
        id: initId,
        stream: "stderr",
        chunk: `Init error: ${error.message}\n`
      });
      event.sender.send("vexor:init:exit", { id: initId, code: null });
      initSessions.delete(initId);
    });

    return { id: initId };
  });

  ipcMain.handle("vexor:init:input", async (_event, { id, input }) => {
    const child = initSessions.get(id);
    if (!child) {
      return { ok: false };
    }
    child.stdin.write(input);
    return { ok: true };
  });

  ipcMain.handle("vexor:init:stop", async (_event, { id }) => {
    const child = initSessions.get(id);
    if (!child) {
      return { ok: false };
    }
    child.kill();
    initSessions.delete(id);
    return { ok: true };
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
