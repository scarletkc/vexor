const { app, BrowserWindow, dialog, ipcMain, shell, Menu } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");

const initSessions = new Map();
let nextInitId = 1;
let activeRunChild = null;

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

function isDirOnPath(dirPath) {
  const envPath = process.env.PATH || "";
  if (!envPath) {
    return false;
  }
  const target = path.resolve(dirPath);
  const normalizedTarget =
    process.platform === "win32" ? target.toLowerCase() : target;
  const entries = envPath.split(path.delimiter);
  return entries.some((entry) => {
    if (!entry) {
      return false;
    }
    const resolved = path.resolve(entry.trim());
    const normalized = process.platform === "win32" ? resolved.toLowerCase() : resolved;
    return normalized === normalizedTarget;
  });
}

function runVexorCommand({ cliPath, args, input }, options = {}) {
  return new Promise((resolve) => {
    const trackActive = Boolean(options.trackActive);
    const resolvedPath = resolveCliPath(cliPath);
    const child = spawn(resolvedPath, args, {
      env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1" },
      windowsHide: true
    });
    if (trackActive) {
      activeRunChild = child;
    }
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

    const finalize = (payload) => {
      if (trackActive && activeRunChild === child) {
        activeRunChild = null;
      }
      resolve(payload);
    };

    child.on("error", (error) => {
      finalize({
        code: null,
        signal: null,
        stdout,
        stderr,
        error: error.message
      });
    });
    child.on("close", (code, signal) => {
      finalize({
        code,
        signal,
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
  const downloadedDir = getCliRootDir();
  const downloadedPath = getDownloadedCliPath();
  const downloadedExists = fs.existsSync(downloadedPath);
  const downloadedInPath = isDirOnPath(downloadedDir);
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
    cliOutput: text,
    downloadedDir,
    downloadedPath,
    downloadedExists,
    downloadedInPath
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

function resolveProfilePath() {
  const shell = process.env.SHELL || "";
  const name = path.basename(shell);
  if (name === "zsh") {
    return path.join(os.homedir(), ".zprofile");
  }
  if (name === "bash") {
    return path.join(os.homedir(), ".bashrc");
  }
  if (name === "fish") {
    return path.join(os.homedir(), ".config", "fish", "config.fish");
  }
  return path.join(os.homedir(), ".profile");
}

function buildPathExportLine(dirPath, profilePath) {
  const escaped = dirPath.replace(/"/g, '\\"');
  if (profilePath.endsWith("config.fish")) {
    return `set -gx PATH $PATH "${escaped}"`;
  }
  return `export PATH="$PATH:${escaped}"`;
}

function removePathFromProfile(dirPath) {
  const profilePath = resolveProfilePath();
  if (!fs.existsSync(profilePath)) {
    return { ok: true, pathRemoved: false, profilePath };
  }
  const marker = "# Added by Vexor Desktop";
  const lines = fs.readFileSync(profilePath, "utf-8").split(/\r?\n/);
  let changed = false;
  const filtered = [];
  for (let idx = 0; idx < lines.length; idx += 1) {
    const line = lines[idx];
    if (line.trim() === marker) {
      changed = true;
      const next = lines[idx + 1];
      if (next && next.includes(dirPath)) {
        idx += 1;
      }
      continue;
    }
    filtered.push(line);
  }
  if (changed) {
    fs.writeFileSync(profilePath, filtered.join("\n"), "utf-8");
  }
  return { ok: true, pathRemoved: changed, profilePath };
}

function removePathFromWindows(dirPath) {
  const currentPath = process.env.PATH || "";
  const entries = currentPath.split(path.delimiter).filter(Boolean);
  const normalizedTarget = path.resolve(dirPath).toLowerCase();
  const filtered = entries.filter((entry) => {
    const resolved = path.resolve(entry.trim()).toLowerCase();
    return resolved !== normalizedTarget;
  });
  const updated = filtered.join(path.delimiter);
  return new Promise((resolve) => {
    const child = spawn("setx", ["PATH", updated], {
      shell: true,
      windowsHide: true
    });
    let stderr = "";
    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ ok: true, pathRemoved: true, profilePath: "User PATH" });
        return;
      }
      resolve({
        ok: false,
        pathRemoved: false,
        error: stderr.trim() || `setx failed (${code})`
      });
    });
    child.on("error", (error) => {
      resolve({ ok: false, pathRemoved: false, error: error.message });
    });
  });
}

function addDownloadedCliToPath() {
  const downloadedPath = getDownloadedCliPath();
  if (!fs.existsSync(downloadedPath)) {
    return Promise.resolve({
      ok: false,
      error: "Downloaded CLI not found."
    });
  }
  const dirPath = getCliRootDir();
  if (isDirOnPath(dirPath)) {
    return Promise.resolve({ ok: true, already: true });
  }
  if (process.platform === "win32") {
    const currentPath = process.env.PATH || "";
    const parts = currentPath.split(path.delimiter).filter(Boolean);
    parts.push(dirPath);
    const updated = parts.join(path.delimiter);
    return new Promise((resolve) => {
      const child = spawn("setx", ["PATH", updated], {
        shell: true,
        windowsHide: true
      });
      let stderr = "";
      child.stderr.on("data", (data) => {
        stderr += data.toString();
      });
      child.on("close", (code) => {
        if (code === 0) {
          resolve({ ok: true, profilePath: "User PATH", restartRequired: true });
          return;
        }
        resolve({
          ok: false,
          error: stderr.trim() || `setx failed (${code})`
        });
      });
      child.on("error", (error) => {
        resolve({ ok: false, error: error.message });
      });
    });
  }

  if (process.platform === "linux") {
    const profilePath = resolveProfilePath();
    const exportLine = buildPathExportLine(dirPath, profilePath);
    const marker = "# Added by Vexor Desktop";
    const parentDir = path.dirname(profilePath);
    if (!fs.existsSync(parentDir)) {
      fs.mkdirSync(parentDir, { recursive: true });
    }
    let content = "";
    if (fs.existsSync(profilePath)) {
      content = fs.readFileSync(profilePath, "utf-8");
      if (content.includes(dirPath)) {
        return Promise.resolve({ ok: true, already: true, profilePath });
      }
    }
    const prefix = content && !content.endsWith("\n") ? "\n" : "";
    const block = `${marker}\n${exportLine}\n`;
    fs.appendFileSync(profilePath, `${prefix}${block}`, "utf-8");
    return Promise.resolve({ ok: true, profilePath, restartRequired: true });
  }

  return Promise.resolve({ ok: false, error: "Unsupported platform." });
}

async function removeDownloadedCli() {
  const downloadedPath = getDownloadedCliPath();
  const dirPath = getCliRootDir();
  let deleted = false;
  let deleteError = null;
  if (fs.existsSync(downloadedPath)) {
    try {
      fs.unlinkSync(downloadedPath);
      deleted = true;
    } catch (error) {
      deleteError = error.message;
    }
  }
  try {
    if (fs.existsSync(dirPath)) {
      const entries = fs.readdirSync(dirPath);
      if (entries.length === 0) {
        fs.rmdirSync(dirPath);
      }
    }
  } catch (_) {
    // Ignore cleanup errors.
  }

  let pathResult = { ok: true, pathRemoved: false, profilePath: "" };
  if (isDirOnPath(dirPath)) {
    if (process.platform === "win32") {
      pathResult = await removePathFromWindows(dirPath);
    } else if (process.platform === "linux") {
      pathResult = removePathFromProfile(dirPath);
    } else {
      pathResult = { ok: false, pathRemoved: false, error: "Unsupported platform." };
    }
  }

  const errors = [];
  if (deleteError) {
    errors.push(`Delete failed: ${deleteError}`);
  }
  if (!pathResult.ok && pathResult.error) {
    errors.push(`PATH update failed: ${pathResult.error}`);
  }

  return {
    ok: errors.length === 0,
    deleted,
    pathRemoved: Boolean(pathResult.pathRemoved),
    profilePath: pathResult.profilePath || "",
    error: errors.join("; ")
  };
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
    return runVexorCommand(payload, { trackActive: true });
  });

  ipcMain.handle("vexor:run-cancel", async () => {
    if (!activeRunChild) {
      return { ok: false, error: "No command is currently running." };
    }
    const child = activeRunChild;
    try {
      const ok = child.kill("SIGINT");
      if (!ok) {
        child.kill();
      }
      return { ok: true };
    } catch (error) {
      return { ok: false, error: error.message || String(error) };
    }
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

  ipcMain.handle("vexor:cli-add-to-path", async () => {
    try {
      return await addDownloadedCliToPath();
    } catch (error) {
      return { ok: false, error: error.message || String(error) };
    }
  });

  ipcMain.handle("vexor:cli-remove-download", async () => {
    try {
      return await removeDownloadedCli();
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

  ipcMain.handle("vexor:open-devtools", async (event) => {
    const win = BrowserWindow.fromWebContents(event.sender);
    if (!win) {
      return { ok: false };
    }
    win.webContents.openDevTools({ mode: "detach" });
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
