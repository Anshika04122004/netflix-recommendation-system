import { chromium } from 'playwright';
import path from 'path';

async function run() {
  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true,
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 950 },
  });

  const page = await context.newPage();
  console.log('Navigating to http://127.0.0.1:8000 ...');
  await page.goto('http://127.0.0.1:8000', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // 1. Dashboard
  console.log('Capturing Dashboard...');
  await page.screenshot({ path: path.join('..', 'screenshots', '01_dashboard.png') });

  // 2. Rating Prediction
  console.log('Capturing Rating Prediction...');
  await page.click('text="Rating Prediction"');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join('..', 'screenshots', '02_prediction.png') });

  // 3. Preset Trigger and Prediction Result
  console.log('Triggering preset prediction...');
  await page.click('text="Sci-Fi & Horror Series"');
  await page.waitForTimeout(500);
  await page.click('button:has-text("Predict Audience Rating")');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join('..', 'screenshots', '03_prediction_result.png') });

  // 4. Model Comparison
  console.log('Capturing Model Comparison...');
  await page.click('text="Model Comparison"');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join('..', 'screenshots', '04_model_comparison.png') });

  // 5. Analytics & Insights
  console.log('Capturing Analytics & Insights...');
  await page.click('text="Analytics & Insights"');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join('..', 'screenshots', '05_analytics_distributions.png') });

  // 6. Confusion Matrix Subtab
  console.log('Capturing Confusion Matrix & Features Subtab...');
  await page.click('text="Confusion Matrix & Features"');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join('..', 'screenshots', '06_analytics_confusion_matrix.png') });

  // 7. About Task 3
  console.log('Capturing About Page...');
  await page.click('text="About Task 3"');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join('..', 'screenshots', '07_about.png') });

  await browser.close();
  console.log('All screenshots captured successfully!');
}

run().catch((err) => {
  console.error('Error during screenshot capture:', err);
  process.exit(1);
});
