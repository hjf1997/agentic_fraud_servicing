/**
 * Export each of the 6 agent diagrams as separate high-res JPGs.
 *
 * Usage: node scripts/export-agents-jpg.js
 * Requires dev server running: npm start
 */

const puppeteer = require("puppeteer");
const path = require("path");
const fs = require("fs");

const EXPORT_DIR = path.join(__dirname, "..", "exports");
const BASE_URL = "http://localhost:3000/#/agent";
const WIDTH = 1920;
const HEIGHT = 1080;

const agents = [
  { id: "triage", name: "allegation_extractor" },
  { id: "auth", name: "auth_agent" },
  { id: "retrieval", name: "retrieval_agent" },
  { id: "specialists", name: "specialist_panel" },
  { id: "arbitrator", name: "typing_arbitrator" },
  { id: "advisor", name: "case_advisor" },
];

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

  for (const agent of agents) {
    const url = `${BASE_URL}/${agent.id}`;
    const outPath = path.join(EXPORT_DIR, `agent_${agent.name}.jpg`);

    console.log(`Capturing ${agent.name}...`);
    await page.goto(url, { waitUntil: "networkidle0" });
    await new Promise((r) => setTimeout(r, 1000));

    // Click the correct tab in case hash navigation didn't trigger re-render
    const tabIndex = agents.findIndex((a) => a.id === agent.id);
    const buttons = await page.$$("button");
    if (buttons[tabIndex]) {
      await buttons[tabIndex].click();
      await new Promise((r) => setTimeout(r, 500));
    }

    await page.screenshot({
      path: outPath,
      type: "jpeg",
      quality: 95,
      fullPage: false,
    });

    const sizeKB = Math.round(fs.statSync(outPath).size / 1024);
    console.log(`  -> ${outPath} (${sizeKB} KB)`);
  }

  await browser.close();
  console.log("\nAll 6 agent diagrams exported.");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
