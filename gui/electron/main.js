const { app, BrowserWindow, dialog, ipcMain, shell, Menu } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const initSessions = new Map();
let nextInitId = 1;

function createWindow() {
  const iconCandidates =
    process.platform === "win32"
      ? ["vexor.ico", "vexor.png"]
      : ["vexor.png", "vexor.ico"];
  let iconPath = null;
  for (const name of iconCandidates) {
    const candidate = path.join(__dirname, "..", "assets", name);
    if (fs.existsSync(candidate)) {
      iconPath = candidate;
      break;
    }
  }
  const windowOptions = {
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
  };
  if (iconPath) {
    windowOptions.icon = iconPath;
  }
  const win = new BrowserWindow(windowOptions);
  win.setMenuBarVisibility(false);
  win.setAutoHideMenuBar(true);

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

function getCliRootDir() {
  return path.join(app.getPath("userData"), "cli");
}

function getDownloadedCliPath() {
  const filename = process.platform === "win32" ? "vexor.exe" : "vexor";
  return path.join(getCliRootDir(), filename);
}

function _expandHome(value) {
  if (value.startsWith("~")) {
    return path.join(os.homedir(), value.slice(1));
  }
  return value;
}

function resolveCustomCliPath(cliPath) {
  if (!cliPath) {
    return null;
  }
  const trimmed = cliPath.trim();
  if (!trimmed) {
    return null;
  }
  const expanded = _expandHome(trimmed);
  const isPathLike =
    expanded.includes("/") ||
    expanded.includes("\\") ||
    expanded.startsWith(".") ||
    path.isAbsolute(expanded);
  if (isPathLike) {
    return fs.existsSync(expanded) ? expanded : null;
  }
  return expanded;
}

function resolveCliPathInfo(cliPath) {
  const downloaded = getDownloadedCliPath();
  if (fs.existsSync(downloaded)) {
    return { path: downloaded, source: "downloaded" };
  }
  const custom = resolveCustomCliPath(cliPath);
  if (custom) {
    return { path: custom, source: "custom" };
  }
  return { path: "vexor", source: "path" };
}

function resolveCliPath(cliPath) {
  return resolveCliPathInfo(cliPath).path;
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

function parseCliVersion(output) {
  if (!output) {
    return null;
  }
  const match = output.match(/v(\d+\.\d+\.\d+(?:[a-z0-9.+-]+)?)/i);
  if (!match) {
    return null;
  }
  return match[1];
}

async function getCliInfo(cliPath) {
  const info = resolveCliPathInfo(cliPath);
  const result = await runVexorCommand({ cliPath, args: ["--version"] });
  const text = `${result.stdout || ""}\n${result.stderr || ""}`.trim();
  const version = parseCliVersion(text);
  const available = Boolean(version) && result.code === 0;
  return {
    appVersion: app.getVersion(),
    cliPath: info.path,
    cliSource: info.source,
    cliAvailable: available,
    cliVersion: version,
    cliOutput: text
  };
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

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    https
      .get(
        url,
        {
          headers: {
            "User-Agent": "vexor-desktop",
            Accept: "application/vnd.github+json"
          }
        },
        (res) => {
          let body = "";
          res.on("data", (chunk) => {
            body += chunk.toString();
          });
          res.on("end", () => {
            if (res.statusCode !== 200) {
              reject(new Error(`HTTP ${res.statusCode}`));
              return;
            }
            try {
              resolve(JSON.parse(body));
            } catch (error) {
              reject(error);
            }
          });
        }
      )
      .on("error", reject);
  });
}

function getPlatformAssetSuffix() {
  if (process.platform === "win32") {
    return "windows.exe";
  }
  if (process.platform === "linux") {
    return "linux";
  }
  return null;
}

async function fetchLatestRelease() {
  const suffix = getPlatformAssetSuffix();
  if (!suffix) {
    return {
      ok: false,
      error: "Unsupported platform",
      releaseUrl: "https://github.com/scarletkc/vexor/releases"
    };
  }
  const data = await fetchJson(
    "https://api.github.com/repos/scarletkc/vexor/releases/latest"
  );
  const tag = (data.tag_name || "").trim();
  const version = tag.startsWith("v") ? tag.slice(1) : tag;
  const assets = Array.isArray(data.assets) ? data.assets : [];
  const expectedName = `vexor-${version}-${suffix}`;
  const asset =
    assets.find((item) => item.name === expectedName) ||
    assets.find((item) => item.name.endsWith(suffix));
  if (!asset) {
    return {
      ok: false,
      error: `No release asset for ${process.platform}`,
      releaseUrl: data.html_url || "https://github.com/scarletkc/vexor/releases"
    };
  }
  return {
    ok: true,
    version,
    tag,
    releaseUrl: data.html_url || "https://github.com/scarletkc/vexor/releases",
    assetName: asset.name,
    assetUrl: asset.browser_download_url
  };
}

function downloadWithRedirect(url, destination, onProgress) {
  return new Promise((resolve, reject) => {
    const request = https.get(
      url,
      { headers: { "User-Agent": "vexor-desktop" } },
      (res) => {
        if (res.statusCode && [301, 302, 303, 307, 308].includes(res.statusCode)) {
          const redirect = res.headers.location;
          if (!redirect) {
            reject(new Error(`Redirect without location (HTTP ${res.statusCode})`));
            return;
          }
          res.resume();
          downloadWithRedirect(redirect, destination, onProgress)
            .then(resolve)
            .catch(reject);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`Download failed (HTTP ${res.statusCode})`));
          return;
        }
        const total = Number(res.headers["content-length"]) || 0;
        const tempPath = `${destination}.download`;
        const file = fs.createWriteStream(tempPath);
        let received = 0;
        res.on("data", (chunk) => {
          received += chunk.length;
          if (onProgress) {
            onProgress({ received, total });
          }
        });
        res.pipe(file);
        file.on("finish", () => {
          file.close(() => {
            fs.renameSync(tempPath, destination);
            if (process.platform !== "win32") {
              fs.chmodSync(destination, 0o755);
            }
            resolve();
          });
        });
        file.on("error", (error) => {
          reject(error);
        });
      }
    );
    request.on("error", reject);
  });
}

async function downloadLatestCli(sender) {
  const info = await fetchLatestRelease();
  if (!info.ok) {
    return info;
  }
  const targetDir = getCliRootDir();
  const targetPath = getDownloadedCliPath();
  fs.mkdirSync(targetDir, { recursive: true });
  sender.send("vexor:cli-download-progress", {
    status: "starting",
    received: 0,
    total: 0
  });
  await downloadWithRedirect(info.assetUrl, targetPath, ({ received, total }) => {
    sender.send("vexor:cli-download-progress", {
      status: "downloading",
      received,
      total
    });
  });
  sender.send("vexor:cli-download-progress", {
    status: "done",
    received: 0,
    total: 0
  });
  return { ok: true, path: targetPath, info };
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
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

  ipcMain.handle("vexor:cli-info", async (_event, payload) => {
    return getCliInfo(payload?.cliPath);
  });

  ipcMain.handle("vexor:cli-check-update", async () => {
    try {
      return await fetchLatestRelease();
    } catch (error) {
      return { ok: false, error: error.message || String(error) };
    }
  });

  ipcMain.handle("vexor:cli-download", async (event) => {
    try {
      return await downloadLatestCli(event.sender);
    } catch (error) {
      return { ok: false, error: error.message || String(error) };
    }
  });

  ipcMain.handle("vexor:open-external", async (_event, payload) => {
    if (!payload?.url) {
      return { ok: false };
    }
    await shell.openExternal(payload.url);
    return { ok: true };
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
