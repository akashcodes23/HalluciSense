const { chromium } = require('/Users/akashgpatil/major_project/frontend/node_modules/playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE RUNTIME ERROR:', err.message));

  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  const cssContent = await page.evaluate(async () => {
    const sec = document.querySelector('section');
    if (!sec) return { error: 'No section found' };

    return {
      className: sec.className,
      computedStyle: {
        paddingTop: window.getComputedStyle(sec).paddingTop,
        paddingBottom: window.getComputedStyle(sec).paddingBottom,
      }
    };
  });

  console.log('RESULT:', cssContent);
  await browser.close();
})();
