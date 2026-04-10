/**
 * Build a standalone HTML file for the animated workflow diagram.
 *
 * Usage:  node scripts/build-export.js
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const BUILD_DIR = path.join(ROOT, "build");
const EXPORT_DIR = path.join(ROOT, "exports");

// 1. Build
console.log("Building production bundle...");
execSync("npx react-scripts build", {
  cwd: ROOT,
  stdio: "inherit",
  env: {
    ...process.env,
    GENERATE_SOURCEMAP: "false",
    PUBLIC_URL: ".",
  },
});

// 2. Read the built assets
const cssFile = fs.readdirSync(path.join(BUILD_DIR, "static/css")).find(
  (f) => f.endsWith(".css")
);
const jsFile = fs.readdirSync(path.join(BUILD_DIR, "static/js")).find(
  (f) => f.endsWith(".js") && !f.endsWith(".LICENSE.txt")
);

const css = fs.readFileSync(
  path.join(BUILD_DIR, "static/css", cssFile),
  "utf-8"
);
const js = fs.readFileSync(
  path.join(BUILD_DIR, "static/js", jsFile),
  "utf-8"
);

// 3. Base64-encode the JS to completely avoid HTML parser issues
//    (React DOM source contains literal </script> and <script> strings)
const jsBase64 = Buffer.from(js, "utf-8").toString("base64");

// 4. Construct a clean standalone HTML using string concatenation
//    to avoid template literal issues with the large base64 payload
const parts = [
  '<!doctype html>',
  '<html lang="en">',
  '<head>',
  '<meta charset="utf-8"/>',
  '<meta name="viewport" content="width=1920,height=1080,initial-scale=1"/>',
  '<title>Copilot Agent Workflow</title>',
  '<style>' + css + '</style>',
  '</head>',
  '<body>',
  '<noscript>You need to enable JavaScript to run this app.</noscript>',
  '<div id="root"></div>',
  '<script>',
  'if(!window.location.hash)window.location.hash="#/export";',
  'window.addEventListener("error",function(e){if(e.message&&e.message.includes("ResizeObserver loop"))e.stopImmediatePropagation()});',
  'var _s=document.createElement("script");',
  '_s.textContent=atob("' + jsBase64 + '");',
  'document.body.appendChild(_s);',
  '<\/script>',
  '</body>',
  '</html>',
];

const html = parts.join("\n");

// 5. Write output
if (!fs.existsSync(EXPORT_DIR)) {
  fs.mkdirSync(EXPORT_DIR, { recursive: true });
}
const outPath = path.join(EXPORT_DIR, "workflow.html");
fs.writeFileSync(outPath, html, "utf-8");

const sizeKB = Math.round(fs.statSync(outPath).size / 1024);
console.log("\nExported: " + outPath + " (" + sizeKB + " KB)");
console.log("Open in browser for the animated workflow diagram.");
