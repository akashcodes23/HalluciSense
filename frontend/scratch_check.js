const { chromium } = require('/Users/akashgpatil/major_project/frontend/node_modules/playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  const cssContent = await page.evaluate(async () => {
    const link = document.querySelector('link[rel="stylesheet"]');
    if (!link) return { error: 'No stylesheet link found' };

    try {
      const resp = await fetch(link.href);
      const text = await resp.text();
      return {
        href: link.href,
        length: text.length,
        hasPt32: text.includes('.pt-32'),
        hasRelative: text.includes('.relative'),
        hasBgBase: text.includes('bg-base'),
      };
    } catch (e) {
      return { error: e.message, href: link.href };
    }
  });

  console.log('SERVED CSS DETAILS:', JSON.stringify(cssContent, null, 2));
  await browser.close();
})();
