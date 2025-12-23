<template>
  <div class="app">
    <header class="hero">
      <div class="brand">
        <img src="/vexor.svg" alt="Vexor" />
        <div>
          <h1>Vexor Desktop</h1>
          <p>A vector-powered desktop app for semantic file search.</p>
        </div>
      </div>
      <div class="status">
        <span class="pill" :class="configInfo.exists ? 'good' : 'warn'">
          Config: {{ configInfo.exists ? 'ready' : 'missing' }}
        </span>
        <span class="pill">
          CLI: {{ cliSourceLabel }}
        </span>
      </div>
    </header>

    <div class="shell">
      <nav class="tabs">
        <button :class="{ active: view === 'run' }" @click="view = 'run'">
          Search / Index
        </button>
        <button :class="{ active: view === 'config' }" @click="view = 'config'">
          Config & Tools
        </button>
        <button :class="{ active: view === 'updates' }" @click="view = 'updates'">
          Updates
        </button>
      </nav>

      <section v-if="view === 'run'">
        <div v-if="!configInfo.exists" class="notice">
          First run: set provider/API key below or run the init wizard.
        </div>
        <div class="grid">
          <form class="card" @submit.prevent="runAction">
            <h2>Run Mode</h2>
            <div class="segmented">
              <button
                type="button"
                :class="{ active: runMode === 'search' }"
                @click="runMode = 'search'"
              >
                Search
              </button>
              <button
                type="button"
                :class="{ active: runMode === 'index' }"
                @click="runMode = 'index'"
              >
                Index
              </button>
            </div>

            <div v-if="runMode === 'search'" class="field">
              <label>Query</label>
              <input v-model="runForm.query" placeholder="e.g. config loader" required />
            </div>

            <div class="field">
              <label>Path</label>
              <div class="inline">
                <input v-model="runForm.path" placeholder="default: current directory" />
                <button type="button" class="secondary" @click="pickDirectory">
                  Browse
                </button>
              </div>
            </div>

            <div class="inline">
              <div class="field">
                <label>Index Mode</label>
                <select v-model="runForm.mode">
                  <option value="auto">auto (smart)</option>
                  <option value="name">name</option>
                  <option value="head">head</option>
                  <option value="brief">brief</option>
                  <option value="full">full</option>
                  <option value="code">code</option>
                  <option value="outline">outline</option>
                </select>
              </div>
              <div v-if="runMode === 'search'" class="field">
                <label>Top K</label>
                <input v-model.number="runForm.top" type="number" min="1" />
              </div>
            </div>

            <div class="inline">
              <div class="field">
                <label>Extensions</label>
                <input v-model="runForm.ext" placeholder=".py,.md" />
              </div>
              <div class="field">
                <label>Exclude Patterns</label>
                <input v-model="runForm.exclude" placeholder="tests/**, .js" />
              </div>
            </div>

            <div class="toggle-group">
              <label class="toggle">
                <input v-model="runForm.includeHidden" type="checkbox" />
                Include hidden files
              </label>
              <label class="toggle">
                <input v-model="runForm.noRespectGitignore" type="checkbox" />
                Ignore gitignore
              </label>
              <label class="toggle">
                <input v-model="runForm.noRecursive" type="checkbox" />
                No recursion
              </label>
            </div>

            <div v-if="runMode === 'index'" class="toggle-group">
              <label class="toggle">
                <input v-model="runForm.clearIndex" type="checkbox" />
                Clear index cache
              </label>
              <label class="toggle">
                <input v-model="runForm.showIndex" type="checkbox" />
                Show index details
              </label>
            </div>

            <div class="actions">
              <button class="primary" type="submit" :disabled="busy">
                {{ runMode === 'search' ? 'Run Search' : 'Build Index' }}
              </button>
              <button type="button" class="secondary" @click="resetRunForm">
                Reset
              </button>
            </div>
          </form>

          <div class="card">
            <h2>Results</h2>
            <div v-if="results.length === 0" class="field">
              <p>No results yet. Run a search.</p>
            </div>
            <div v-else class="results-table">
              <div v-for="result in results" :key="result.rank" class="result-row">
                <div><strong>#{{ result.rank }}</strong></div>
                <div>{{ result.score }}</div>
                <div :title="result.path">{{ result.path }}</div>
                <div>{{ result.lines }}</div>
                <div class="preview">{{ result.preview }}</div>
              </div>
            </div>
          </div>
        </div>

        <div class="card command-log">
          <h2>Command Log</h2>
          <div class="log">{{ logOutput || 'Waiting for command output...' }}</div>
        </div>
      </section>

      <section v-else-if="view === 'updates'">
        <div class="card">
          <h2>Updates</h2>
          <div class="inline">
            <div class="field">
              <label>GUI Version</label>
              <input :value="guiVersionLabel" readonly />
            </div>
            <div class="field">
              <label>CLI Version</label>
              <input :value="cliVersionLabel" readonly />
            </div>
          </div>
          <div class="field">
            <label>CLI Source</label>
            <div class="inline">
              <span class="pill">{{ cliSourceLabel }}</span>
              <span class="pill" v-if="cliInfo.cliPath">{{ cliInfo.cliPath }}</span>
            </div>
          </div>
          <div class="actions">
            <button
              class="primary"
              type="button"
              @click="downloadCli"
              :disabled="busy || downloadInfo.inProgress"
            >
              Download CLI
            </button>
            <button class="secondary" type="button" @click="checkCliUpdate" :disabled="busy">
              Check CLI Update
            </button>
            <button class="secondary" type="button" @click="openReleases">
              Open GitHub Releases
            </button>
          </div>
          <div v-if="updateMessage" class="notice">{{ updateMessage }}</div>
          <div v-if="downloadInfo.inProgress" class="progress">
            <div class="progress-bar" :style="{ width: downloadProgressPercent + '%' }"></div>
            <div class="progress-meta">{{ downloadProgressLabel }}</div>
          </div>
        </div>

        <div class="card command-log">
          <h2>Command Log</h2>
          <div class="log">{{ logOutput || 'Waiting for command output...' }}</div>
        </div>
      </section>

      <section v-else>
        <div class="grid">
          <form class="card" @submit.prevent="saveConfig">
            <h2>Core Config</h2>
            <div class="field">
              <label>CLI Path (optional)</label>
              <input v-model="cliPath" placeholder="e.g. /usr/local/bin/vexor" />
              <small>Downloaded CLI takes priority; this is used when missing.</small>
            </div>

            <div class="inline">
              <div class="field">
                <label>Provider</label>
                <select v-model="configForm.provider">
                  <option value="">(keep default)</option>
                  <option value="openai">openai</option>
                  <option value="gemini">gemini</option>
                  <option value="custom">custom</option>
                  <option value="local">local</option>
                </select>
              </div>
              <div class="field">
                <label>Model</label>
                <input v-model="configForm.model" placeholder="text-embedding-3-small" />
              </div>
            </div>

            <div class="field">
              <label>API Key (leave blank to keep current)</label>
              <input v-model="configForm.apiKey" type="password" placeholder="sk-..." autocomplete="off" />
            </div>

            <div class="field">
              <label>Base URL</label>
              <input v-model="configForm.baseUrl" placeholder="https://api.openai.com/v1" />
            </div>

            <div class="inline">
              <div class="field">
                <label>Batch Size</label>
                <input v-model.number="configForm.batchSize" type="number" min="0" />
              </div>
              <div class="field">
                <label>Embed Concurrency</label>
                <input v-model.number="configForm.embedConcurrency" type="number" min="1" />
              </div>
            </div>

            <div class="toggle-group">
              <label class="toggle">
                <input v-model="configForm.autoIndex" type="checkbox" />
                Auto index
              </label>
              <label class="toggle">
                <input v-model="configForm.localCuda" type="checkbox" />
                Local CUDA
              </label>
            </div>

            <div class="actions">
              <button class="primary" type="submit" :disabled="busy">Save Config</button>
              <button type="button" class="secondary" @click="loadConfig(true)">
                Load Current
              </button>
            </div>
          </form>

          <div class="card">
            <h2>Rerank & Local</h2>
            <div class="field">
              <label>Rerank Mode</label>
              <select v-model="configForm.rerank">
                <option value="off">off</option>
                <option value="bm25">bm25</option>
                <option value="flashrank">flashrank</option>
                <option value="remote">remote</option>
              </select>
            </div>

            <div v-if="configForm.rerank === 'flashrank'" class="field">
              <label>FlashRank Model</label>
              <input v-model="configForm.flashrankModel" placeholder="ms-marco-TinyBERT-L-2-v2" />
            </div>

            <div v-if="configForm.rerank === 'remote'">
              <div class="field">
                <label>Remote Rerank URL</label>
                <input v-model="configForm.remoteRerankUrl" placeholder="https://proxy.example.com/v1/rerank" />
              </div>
              <div class="field">
                <label>Remote Rerank Model</label>
                <input v-model="configForm.remoteRerankModel" placeholder="bge-reranker-v2-m3" />
              </div>
              <div class="field">
                <label>Remote Rerank API Key</label>
                <input v-model="configForm.remoteRerankApiKey" type="password" placeholder="rk-..." autocomplete="off" />
              </div>
            </div>

            <h3>Local Model Actions</h3>
            <div class="field">
              <label>Local Model</label>
              <input v-model="localModel" placeholder="intfloat/multilingual-e5-small" />
            </div>
            <div class="actions">
              <button class="primary" type="button" @click="runLocalSetup" :disabled="busy">
                Setup Local Model
              </button>
              <button class="secondary" type="button" @click="switchLocalCuda(true)">
                Switch to CUDA
              </button>
              <button class="secondary" type="button" @click="switchLocalCuda(false)">
                Switch to CPU
              </button>
              <button class="secondary" type="button" @click="cleanLocalCache">
                Clean Local Cache
              </button>
            </div>

            <h3>Quick Tools</h3>
            <div class="actions">
              <button class="secondary" type="button" @click="runDoctor">
                Run doctor
              </button>
              <button class="secondary" type="button" @click="showConfig">
                Show config
              </button>
              <button class="secondary" type="button" @click="showIndexAll">
                Show index list
              </button>
              <button class="secondary" type="button" @click="clearIndexAll">
                Clear all indexes
              </button>
              <button class="secondary" type="button" @click="clearFlashrank">
                Clear FlashRank cache
              </button>
              <button class="secondary" type="button" @click="clearBaseUrl">
                Clear Base URL
              </button>
              <button class="secondary" type="button" @click="clearRemoteRerank">
                Clear remote rerank
              </button>
              <button class="secondary" type="button" @click="clearApiKey">
                Clear API key
              </button>
              <button class="secondary" type="button" @click="openInitWizard">
                Init wizard
              </button>
            </div>
          </div>
        </div>

        <div class="card command-log">
          <h2>Command Log</h2>
          <div class="log">{{ logOutput || 'Waiting for command output...' }}</div>
        </div>
      </section>
    </div>

    <div v-if="initModalOpen" class="modal">
      <div class="modal-card">
        <h2>Init Wizard (CLI)</h2>
        <p>This runs <code>vexor init</code> in an interactive console. Follow the prompts.</p>
        <textarea readonly :value="initLog"></textarea>
        <div class="inline">
          <input
            v-model="initInput"
            placeholder="Type and press Enter"
            @keyup.enter="sendInitInput"
          />
        </div>
        <div class="modal-actions">
          <button class="secondary" type="button" @click="stopInitWizard">Stop</button>
          <button class="primary" type="button" @click="closeInitWizard">Close</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";

const view = ref("run");
const runMode = ref("search");
const busy = ref(false);
const results = ref([]);
const logOutput = ref("");
const cliPath = ref(localStorage.getItem("vexorCliPath") || "");
const configInfo = reactive({ path: "", exists: false, parseError: null, config: {} });
const appVersion = ref("");
const cliInfo = reactive({
  cliPath: "",
  cliSource: "",
  cliAvailable: false,
  cliVersion: "",
  cliOutput: ""
});
const updateInfo = reactive({
  checked: false,
  latestVersion: "",
  releaseUrl: "https://github.com/scarletkc/vexor/releases",
  assetName: "",
  assetUrl: "",
  error: ""
});
const downloadInfo = reactive({
  inProgress: false,
  received: 0,
  total: 0,
  status: ""
});

const runForm = reactive({
  query: "",
  path: "",
  mode: "auto",
  top: 5,
  includeHidden: false,
  noRespectGitignore: false,
  noRecursive: false,
  ext: "",
  exclude: "",
  clearIndex: false,
  showIndex: false
});

const configForm = reactive({
  provider: "",
  apiKey: "",
  model: "",
  baseUrl: "",
  batchSize: "",
  embedConcurrency: "",
  autoIndex: true,
  rerank: "off",
  flashrankModel: "",
  remoteRerankUrl: "",
  remoteRerankModel: "",
  remoteRerankApiKey: "",
  localCuda: false
});

const localModel = ref("");
const initModalOpen = ref(false);
const initSessionId = ref(null);
const initLog = ref("");
const initInput = ref("");
let removeInitOutput = null;
let removeInitExit = null;
let removeCliDownloadProgress = null;

watch(cliPath, (value) => {
  localStorage.setItem("vexorCliPath", value);
});

const guiVersionLabel = computed(() => {
  return appVersion.value ? `v${appVersion.value}` : "unknown";
});

const cliVersionLabel = computed(() => {
  return cliInfo.cliVersion ? `v${cliInfo.cliVersion}` : "not found";
});

const cliSourceLabel = computed(() => {
  if (!cliInfo.cliSource) {
    return "unknown";
  }
  if (cliInfo.cliSource === "downloaded") {
    return "downloaded";
  }
  if (cliInfo.cliSource === "custom") {
    return "custom";
  }
  if (cliInfo.cliSource === "path") {
    return "path";
  }
  return cliInfo.cliSource;
});

const downloadProgressPercent = computed(() => {
  if (!downloadInfo.total) {
    return 0;
  }
  return Math.min(100, Math.round((downloadInfo.received / downloadInfo.total) * 100));
});

const downloadProgressLabel = computed(() => {
  if (!downloadInfo.total) {
    return "Downloading...";
  }
  return `${formatBytes(downloadInfo.received)} / ${formatBytes(downloadInfo.total)}`;
});

const updateMessage = computed(() => {
  if (!updateInfo.checked) {
    return "";
  }
  if (updateInfo.error) {
    return `Update check failed: ${updateInfo.error}`;
  }
  if (!updateInfo.latestVersion) {
    return "Unable to determine latest release.";
  }
  if (!cliInfo.cliVersion) {
    return `Latest release: v${updateInfo.latestVersion}. CLI not installed. Use Download CLI.`;
  }
  const comparison = compareVersions(cliInfo.cliVersion, updateInfo.latestVersion);
  if (comparison === null) {
    return `Latest release: v${updateInfo.latestVersion}.`;
  }
  if (comparison < 0) {
    return `Update available: v${updateInfo.latestVersion}. Use Download CLI or open GitHub Releases.`;
  }
  return "CLI is up to date.";
});

function stripAnsi(text) {
  return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function formatBytes(value) {
  if (!Number.isFinite(value)) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  const units = ["KB", "MB", "GB"];
  let size = value;
  let idx = -1;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(1)} ${units[idx]}`;
}

function normalizeVersion(value) {
  if (!value) {
    return null;
  }
  const match = value.match(/(\d+)\.(\d+)\.(\d+)/);
  if (!match) {
    return null;
  }
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

function compareVersions(left, right) {
  const a = normalizeVersion(left);
  const b = normalizeVersion(right);
  if (!a || !b) {
    return null;
  }
  for (let idx = 0; idx < 3; idx += 1) {
    if (a[idx] > b[idx]) {
      return 1;
    }
    if (a[idx] < b[idx]) {
      return -1;
    }
  }
  return 0;
}

async function loadConfig(applyToForm) {
  const info = await window.vexor.getConfigInfo();
  configInfo.path = info.path;
  configInfo.exists = info.exists;
  configInfo.parseError = info.parseError;
  configInfo.config = info.config || {};
  if (applyToForm) {
    applyConfigSnapshot(info.config || {});
  }
}

function applyConfigSnapshot(snapshot) {
  configForm.provider = snapshot.provider || "";
  configForm.model = snapshot.model || "";
  configForm.baseUrl = snapshot.base_url || "";
  configForm.batchSize = snapshot.batch_size ?? "";
  configForm.embedConcurrency = snapshot.embed_concurrency ?? "";
  configForm.autoIndex = snapshot.auto_index ?? true;
  configForm.rerank = snapshot.rerank || "off";
  configForm.flashrankModel = snapshot.flashrank_model || "";
  configForm.localCuda = Boolean(snapshot.local_cuda);
  if (snapshot.remote_rerank) {
    configForm.remoteRerankUrl = snapshot.remote_rerank.base_url || "";
    configForm.remoteRerankModel = snapshot.remote_rerank.model || "";
    configForm.remoteRerankApiKey = snapshot.remote_rerank.api_key || "";
  } else {
    configForm.remoteRerankUrl = "";
    configForm.remoteRerankModel = "";
    configForm.remoteRerankApiKey = "";
  }
}

async function refreshCliInfo() {
  const info = await window.vexor.getCliInfo({ cliPath: cliPath.value });
  appVersion.value = info.appVersion || "";
  cliInfo.cliPath = info.cliPath || "";
  cliInfo.cliSource = info.cliSource || "";
  cliInfo.cliAvailable = Boolean(info.cliAvailable);
  cliInfo.cliVersion = info.cliVersion || "";
  cliInfo.cliOutput = info.cliOutput || "";
}

async function pickDirectory() {
  const selected = await window.vexor.selectDirectory();
  if (selected) {
    runForm.path = selected;
  }
}

function resetRunForm() {
  runForm.query = "";
  runForm.path = "";
  runForm.mode = "auto";
  runForm.top = 5;
  runForm.includeHidden = false;
  runForm.noRespectGitignore = false;
  runForm.noRecursive = false;
  runForm.ext = "";
  runForm.exclude = "";
  runForm.clearIndex = false;
  runForm.showIndex = false;
  results.value = [];
  logOutput.value = "";
}

function buildCommonArgs() {
  const args = [];
  if (runForm.path.trim()) {
    args.push("--path", runForm.path.trim());
  }
  if (runForm.mode) {
    args.push("--mode", runForm.mode);
  }
  if (runForm.ext.trim()) {
    args.push("--ext", runForm.ext.trim());
  }
  if (runForm.exclude.trim()) {
    args.push("--exclude-pattern", runForm.exclude.trim());
  }
  if (runForm.includeHidden) {
    args.push("--include-hidden");
  }
  if (runForm.noRespectGitignore) {
    args.push("--no-respect-gitignore");
  }
  if (runForm.noRecursive) {
    args.push("--no-recursive");
  }
  return args;
}

function parsePorcelain(stdout) {
  const lines = stdout.trim().split("\n").filter(Boolean);
  return lines.map((line) => {
    const parts = line.split("\t");
    const preview = parts.slice(6).join("\t");
    const startLine = parts[4] === "-" ? "" : parts[4];
    const endLine = parts[5] === "-" ? "" : parts[5];
    const linesLabel = startLine
      ? endLine && endLine !== startLine
        ? `L${startLine}-${endLine}`
        : `L${startLine}`
      : "-";
    return {
      rank: parts[0],
      score: parts[1],
      path: parts[2],
      lines: linesLabel,
      preview: preview ? preview.replace(/\\\\/g, "\\") : "-"
    };
  });
}

function buildLog(result, includeStdout) {
  let log = "";
  if (result.error) {
    log += `Error: ${result.error}\n`;
  }
  if (result.stderr) {
    log += result.stderr;
  }
  if (includeStdout && result.stdout) {
    log += result.stdout;
  }
  return stripAnsi(log).trim();
}

async function runAction() {
  if (busy.value) {
    return;
  }
  if (runMode.value === "search" && !runForm.query.trim()) {
    logOutput.value = "Please enter a query.";
    return;
  }
  if (runMode.value === "index" && runForm.clearIndex && runForm.showIndex) {
    logOutput.value = "Clear index and show details cannot be selected together.";
    return;
  }
  busy.value = true;
  results.value = [];

  const args = [];
  if (runMode.value === "search") {
    args.push("search", runForm.query.trim(), "--format", "porcelain");
    args.push("--top", String(runForm.top || 5));
  } else {
    args.push("index");
    if (runForm.clearIndex) {
      args.push("--clear");
    }
    if (runForm.showIndex) {
      args.push("--show");
    }
  }
  args.push(...buildCommonArgs());

  const result = await window.vexor.run({ cliPath: cliPath.value, args });
  logOutput.value = buildLog(result, runMode.value !== "search");
  if (runMode.value === "search" && result.stdout) {
    results.value = parsePorcelain(result.stdout);
  }
  busy.value = false;
}

async function saveConfig() {
  if (busy.value) {
    return;
  }
  busy.value = true;
  const args = ["config"];
  if (configForm.apiKey.trim()) {
    args.push("--set-api-key", configForm.apiKey.trim());
  }
  if (configForm.provider) {
    args.push("--set-provider", configForm.provider);
  }
  if (configForm.model.trim()) {
    args.push("--set-model", configForm.model.trim());
  }
  if (configForm.baseUrl.trim()) {
    args.push("--set-base-url", configForm.baseUrl.trim());
  }
  if (Number.isFinite(configForm.batchSize)) {
    args.push("--set-batch-size", String(configForm.batchSize));
  }
  if (Number.isFinite(configForm.embedConcurrency)) {
    args.push("--set-embed-concurrency", String(configForm.embedConcurrency));
  }
  args.push("--set-auto-index", configForm.autoIndex ? "true" : "false");
  if (configForm.rerank) {
    args.push("--rerank", configForm.rerank);
  }
  if (configForm.flashrankModel.trim()) {
    args.push("--set-flashrank-model", configForm.flashrankModel.trim());
  }
  if (configForm.remoteRerankUrl.trim()) {
    args.push("--set-remote-rerank-url", configForm.remoteRerankUrl.trim());
  }
  if (configForm.remoteRerankModel.trim()) {
    args.push("--set-remote-rerank-model", configForm.remoteRerankModel.trim());
  }
  if (configForm.remoteRerankApiKey.trim()) {
    args.push("--set-remote-rerank-api-key", configForm.remoteRerankApiKey.trim());
  }

  const result = await window.vexor.run({ cliPath: cliPath.value, args });
  logOutput.value = buildLog(result, true);
  await loadConfig(true);
  await refreshCliInfo();
  busy.value = false;
}

async function runConfigAction(args) {
  if (busy.value) {
    return;
  }
  busy.value = true;
  const result = await window.vexor.run({ cliPath: cliPath.value, args });
  logOutput.value = buildLog(result, true);
  await loadConfig(true);
  await refreshCliInfo();
  busy.value = false;
}

async function runLocalSetup() {
  const modelName = (localModel.value || configForm.model || "intfloat/multilingual-e5-small").trim();
  const args = ["local", "--setup", "--model", modelName];
  if (configForm.localCuda) {
    args.push("--cuda");
  } else {
    args.push("--cpu");
  }
  await runConfigAction(args);
}

async function switchLocalCuda(enable) {
  const args = ["local", enable ? "--cuda" : "--cpu"];
  await runConfigAction(args);
}

async function cleanLocalCache() {
  await runConfigAction(["local", "--clean-up"]);
}

async function runDoctor() {
  await runConfigAction(["doctor"]);
}

async function showConfig() {
  await runConfigAction(["config", "--show"]);
}

async function clearIndexAll() {
  await runConfigAction(["config", "--clear-index-all"]);
}

async function showIndexAll() {
  await runConfigAction(["config", "--show-index-all"]);
}

async function clearFlashrank() {
  await runConfigAction(["config", "--clear-flashrank"]);
}

async function clearBaseUrl() {
  await runConfigAction(["config", "--clear-base-url"]);
}

async function clearRemoteRerank() {
  await runConfigAction(["config", "--clear-remote-rerank"]);
}

async function clearApiKey() {
  await runConfigAction(["config", "--clear-api-key"]);
}

async function checkCliUpdate() {
  if (busy.value) {
    return;
  }
  busy.value = true;
  updateInfo.checked = true;
  updateInfo.error = "";
  const result = await window.vexor.checkCliUpdate();
  if (!result.ok) {
    updateInfo.error = result.error || "Update check failed.";
    updateInfo.latestVersion = "";
    updateInfo.assetName = "";
    updateInfo.assetUrl = "";
    updateInfo.releaseUrl =
      result.releaseUrl || "https://github.com/scarletkc/vexor/releases";
  } else {
    updateInfo.latestVersion = result.version || "";
    updateInfo.assetName = result.assetName || "";
    updateInfo.assetUrl = result.assetUrl || "";
    updateInfo.releaseUrl =
      result.releaseUrl || "https://github.com/scarletkc/vexor/releases";
  }
  await refreshCliInfo();
  busy.value = false;
}

async function downloadCli() {
  if (downloadInfo.inProgress || busy.value) {
    return;
  }
  busy.value = true;
  downloadInfo.inProgress = true;
  downloadInfo.received = 0;
  downloadInfo.total = 0;
  const result = await window.vexor.downloadCli();
  if (!result.ok) {
    logOutput.value = `CLI download failed: ${result.error || "unknown error"}`;
  }
  downloadInfo.inProgress = false;
  await refreshCliInfo();
  busy.value = false;
  await checkCliUpdate();
}

async function openReleases() {
  const url = updateInfo.releaseUrl || "https://github.com/scarletkc/vexor/releases";
  await window.vexor.openExternal(url);
}

async function openInitWizard() {
  initModalOpen.value = true;
  if (initSessionId.value) {
    return;
  }
  initLog.value = "";
  const response = await window.vexor.initStart({ cliPath: cliPath.value });
  initSessionId.value = response.id;
}

async function sendInitInput() {
  if (!initSessionId.value) {
    return;
  }
  const text = initInput.value;
  await window.vexor.initSend({ id: initSessionId.value, input: `${text}\n` });
  initInput.value = "";
}

async function stopInitWizard() {
  if (initSessionId.value) {
    await window.vexor.initStop({ id: initSessionId.value });
  }
}

function closeInitWizard() {
  initModalOpen.value = false;
}

onMounted(async () => {
  await loadConfig(true);
  await refreshCliInfo();
  removeInitOutput = window.vexor.onInitOutput((payload) => {
    if (!initSessionId.value || payload.id !== initSessionId.value) {
      return;
    }
    initLog.value += stripAnsi(payload.chunk);
  });
  removeInitExit = window.vexor.onInitExit((payload) => {
    if (!initSessionId.value || payload.id !== initSessionId.value) {
      return;
    }
    initLog.value += `\n[init] exited with code ${payload.code ?? "?"}\n`;
    initSessionId.value = null;
    loadConfig(true);
  });
  removeCliDownloadProgress = window.vexor.onCliDownloadProgress((payload) => {
    downloadInfo.status = payload.status || "";
    downloadInfo.received = payload.received || 0;
    downloadInfo.total = payload.total || 0;
    downloadInfo.inProgress = payload.status === "downloading" || payload.status === "starting";
  });
});

onBeforeUnmount(() => {
  if (removeInitOutput) {
    removeInitOutput();
  }
  if (removeInitExit) {
    removeInitExit();
  }
  if (removeCliDownloadProgress) {
    removeCliDownloadProgress();
  }
});
</script>
