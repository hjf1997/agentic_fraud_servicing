/**
 * Export the Copilot Agent Workflow (#/export) as a high-res JPG
 * at a specific animation step.
 *
 * Usage: node scripts/export-workflow-jpg.js [step]
 *   step: 1-indexed animation step (default: 30, last step)
 *
 * Requires dev server running: npm start
 */

const puppeteer = require("puppeteer");
const path = require("path");
const fs = require("fs");

const EXPORT_DIR = path.join(__dirname, "..", "exports");
const OUT_PATH = path.join(EXPORT_DIR, "workflow.jpg");
const URL = "http://localhost:3000/#/export";
const WIDTH = 1920;
const HEIGHT = 1080;

// Target step (1-indexed), default 30
const TARGET_STEP = parseInt(process.argv[2] || "30", 10);

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
  await page.setViewport({ width: WIDTH, height: HEIGHT, deviceScaleFactor: 2 });

  console.log(`Navigating to ${URL}...`);
  await page.goto(URL, { waitUntil: "networkidle0" });
  // Wait for initial render
  await new Promise((r) => setTimeout(r, 2000));

  // Pause the animation first
  console.log("Pausing animation...");
  await page.evaluate(() => {
    // Click the pause button (the play/pause button)
    const buttons = document.querySelectorAll("button");
    for (const btn of buttons) {
      if (btn.title === "Pause" || btn.title === "Play") {
        btn.click();
        break;
      }
    }
  });
  await new Promise((r) => setTimeout(r, 500));

  // Click restart to go to step 1
  console.log("Restarting from step 1...");
  await page.evaluate(() => {
    const buttons = document.querySelectorAll("button");
    for (const btn of buttons) {
      if (btn.title === "Restart") {
        btn.click();
        break;
      }
    }
  });
  await new Promise((r) => setTimeout(r, 500));

  // Pause again after restart
  await page.evaluate(() => {
    const buttons = document.querySelectorAll("button");
    for (const btn of buttons) {
      if (btn.title === "Pause") {
        btn.click();
        break;
      }
    }
  });
  await new Promise((r) => setTimeout(r, 500));

  // Advance step by step using the progress dots to build cumulative state
  console.log(`Advancing to step ${TARGET_STEP}...`);
  for (let i = 0; i < TARGET_STEP; i++) {
    await page.evaluate((stepIdx) => {
      // Click the progress dot for this step
      const dots = document.querySelectorAll("div[style*='border-radius: 4px'][style*='cursor: pointer']");
      if (dots[stepIdx]) {
        (dots[stepIdx]).click();
      }
    }, i);
    // Brief pause to let cumulative doneNodes accumulate
    await new Promise((r) => setTimeout(r, 150));
  }

  // Wait for transitions to settle
  console.log("Waiting for transitions to settle...");
  await new Promise((r) => setTimeout(r, 2000));

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
