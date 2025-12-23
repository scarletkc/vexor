const path = require("path");

const iconName = process.platform === "win32" ? "vexor.ico" : "vexor.png";

module.exports = {
  packagerConfig: {
    asar: true,
    icon: path.join(__dirname, "assets", iconName)
  },
  makers: [
    {
      name: "@electron-forge/maker-zip",
      platforms: ["linux", "win32"]
    }
  ]
};
