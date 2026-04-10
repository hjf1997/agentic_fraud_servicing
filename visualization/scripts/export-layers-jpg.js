/**
 * Export the Layered Architecture View (#/layers) as a high-res JPG.
 *
 * Usage: node scripts/export-layers-jpg.js
 * Requires dev server running: npm start
 */

const puppeteer = require("puppeteer");
const path = require("path");
const fs = require("fs");

const EXPORT_DIR = path.join(__dirname, "..", "exports");
const OUT_PATH = path.join(EXPORT_DIR, "layers.jpg");
const URL = "http://localhost:3000/#/layers";
const WIDTH = 1920;
const HEIGHT = 1080;

async function main() {
  if (!fs.existsSync(EXPORT_DIR)) {
    fs.mkdirSync(EXPORT_DIR, { recursive: true });
  }

  console.log("Launching browser...");
  const browser = await puppeteer.launch({
    headless: true,
    args: [`--window-size=${WIDTH},${HEIGHT}`],
  });

  const page = await browser.newPage();
  // Use deviceScaleFactor 2 for high-res (retina) output
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 2 });

  console.log(`Navigating to ${URL}...`);
  await page.goto(URL, { waitUntil: "networkidle0" });
  await new Promise((r) => setTimeout(r, 1500));

  console.log("Capturing screenshot...");
  await page.screenshot({
    path: OUT_PATH,
    type: "jpeg",
    quality: 95,
    fullPage: false,
  });

  await browser.close();

  const sizeKB = Math.round(fs.statSync(OUT_PATH).size / 1024);
  console.log(`\nExported: ${OUT_PATH} (${sizeKB} KB, ${WIDTH * 2}x${HEIGHT * 2})`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
