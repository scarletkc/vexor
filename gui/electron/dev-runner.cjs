const { spawn } = require("child_process");
const path = require("path");
const electronPath = require("electron");

const cwd = path.resolve(__dirname, "..");
const env = {
  ...process.env,
  VITE_DEV_SERVER_URL: process.env.VITE_DEV_SERVER_URL || "http://localhost:5173"
};

const child = spawn(electronPath, ["."], { cwd, env, stdio: "inherit" });
child.on("exit", (code) => process.exit(code ?? 0));
