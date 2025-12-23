const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("vexor", {
  selectDirectory: () => ipcRenderer.invoke("vexor:select-directory"),
  run: (payload) => ipcRenderer.invoke("vexor:run", payload),
  getCliInfo: (payload) => ipcRenderer.invoke("vexor:cli-info", payload),
  checkCliUpdate: () => ipcRenderer.invoke("vexor:cli-check-update"),
  downloadCli: () => ipcRenderer.invoke("vexor:cli-download"),
  addCliToPath: () => ipcRenderer.invoke("vexor:cli-add-to-path"),
  openExternal: (url) => ipcRenderer.invoke("vexor:open-external", { url }),
  openDevTools: () => ipcRenderer.invoke("vexor:open-devtools"),
  getConfigInfo: () => ipcRenderer.invoke("vexor:config-info"),
  initStart: (payload) => ipcRenderer.invoke("vexor:init:start", payload),
  initSend: (payload) => ipcRenderer.invoke("vexor:init:input", payload),
  initStop: (payload) => ipcRenderer.invoke("vexor:init:stop", payload),
  onInitOutput: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on("vexor:init:output", listener);
    return () => ipcRenderer.removeListener("vexor:init:output", listener);
  },
  onInitExit: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on("vexor:init:exit", listener);
    return () => ipcRenderer.removeListener("vexor:init:exit", listener);
  },
  onCliDownloadProgress: (callback) => {
    const listener = (_event, data) => callback(data);
    ipcRenderer.on("vexor:cli-download-progress", listener);
    return () => ipcRenderer.removeListener("vexor:cli-download-progress", listener);
  }
});
