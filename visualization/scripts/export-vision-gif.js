/**
 * Export the Vision diagram (#/vision) as an animated GIF.
 *
 * Usage: node scripts/export-vision-gif.js
 *
 * Requires the dev server running: npm start
 * Then run this script in another terminal.
 */

const puppeteer = require("puppeteer");
const GIFEncoder = require("gif-encoder");
const { PNG } = require("pngjs");
const fs = require("fs");
const path = require("path");

const EXPORT_DIR = path.join(__dirname, "..", "exports");
const OUT_PATH = path.join(EXPORT_DIR, "vision.gif");
const URL = "http://localhost:3000/#/vision";
const WIDTH = 1920;
const HEIGHT = 1080;
const FRAMES = 60; // total frames
const FRAME_DELAY = 80; // ms between frames in GIF (80ms = ~12fps)
const CAPTURE_INTERVAL = 100; // ms between captures

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
  await page.setViewport({ width: WIDTH, height: HEIGHT });

  console.log(`Navigating to ${URL}...`);
  await page.goto(URL, { waitUntil: "networkidle0" });

  // Wait for React Flow to render
  await page.waitForSelector(".react-flow__renderer", { timeout: 10000 });
  await new Promise((r) => setTimeout(r, 2000)); // extra settle time

  console.log(`Capturing ${FRAMES} frames...`);

  // Set up GIF encoder
  const encoder = new GIFEncoder(WIDTH, HEIGHT);
  const gifStream = fs.createWriteStream(OUT_PATH);
  encoder.pipe(gifStream);
  encoder.setRepeat(0); // loop forever
  encoder.setDelay(FRAME_DELAY);
  encoder.setQuality(10);
  encoder.writeHeader();

  for (let i = 0; i < FRAMES; i++) {
    const pngBuffer = await page.screenshot({ type: "png" });
    const png = PNG.sync.read(pngBuffer);

    // GIF encoder expects raw RGBA pixel data
    encoder.addFrame(png.data);

    if ((i + 1) % 10 === 0) {
      console.log(`  Frame ${i + 1}/${FRAMES}`);
    }

    await new Promise((r) => setTimeout(r, CAPTURE_INTERVAL));
  }

  encoder.finish();
  await browser.close();

  // Wait for file write to complete
  await new Promise((resolve) => gifStream.on("finish", resolve));

  const sizeKB = Math.round(fs.statSync(OUT_PATH).size / 1024);
  console.log(`\nExported: ${OUT_PATH} (${sizeKB} KB)`);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
