const path = require("path");

const iconName = process.platform === "win32" ? "vexor.ico" : "vexor.png";
const iconPath = path.join(__dirname, "assets", iconName);
const iconPngPath = path.join(__dirname, "assets", "vexor.png");
const iconIcoPath = path.join(__dirname, "assets", "vexor.ico");

module.exports = {
  packagerConfig: {
    asar: true,
    icon: iconPath
  },
  makers: [
    {
      name: "@electron-forge/maker-zip",
      platforms: ["linux", "win32"]
    },
    {
      name: "@electron-forge/maker-squirrel",
      config: {
        name: "vexor-desktop",
        setupIcon: iconIcoPath
      },
      platforms: ["win32"]
    },
    {
      name: "@electron-forge/maker-deb",
      config: {
        options: {
          maintainer: "Vexor",
          homepage: "https://github.com/scarletkc/vexor",
          icon: iconPngPath
        }
      },
      platforms: ["linux"]
    },
    {
      name: "@electron-forge/maker-rpm",
      config: {
        options: {
          homepage: "https://github.com/scarletkc/vexor",
          icon: iconPngPath
        }
      },
      platforms: ["linux"]
    }
  ]
};
