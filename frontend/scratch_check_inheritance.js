const { chromium } = require('/Users/akashgpatil/major_project/frontend/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({
    args: ['--disable-web-security']
  });
  const page = await browser.newPage();
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  const matchedRules = await page.evaluate(() => {
    const sec = document.querySelector('section');
    if (!sec) return null;

    const matched = [];

    function checkRulesRecursive(rules, element) {
      for (let i = 0; i < rules.length; i++) {
        const rule = rules[i];
        if (rule.selectorText && element.matches(rule.selectorText)) {
          matched.push({
            selector: rule.selectorText,
            cssText: rule.cssText,
            parentLayer: rule.parentRule ? rule.parentRule.cssText.substring(0, 100) : 'none'
          });
        }
        if (rule.cssRules) {
          checkRulesRecursive(rule.cssRules, element);
        }
      }
    }

    for (let i = 0; i < document.styleSheets.length; i++) {
      const sheet = document.styleSheets[i];
      try {
        checkRulesRecursive(sheet.cssRules || sheet.rules, sec);
      } catch (e) {
        matched.push({ error: e.message, sheetHref: sheet.href || 'inline' });
      }
    }

    return {
      classList: Array.from(sec.classList),
      computedStyle: {
        paddingTop: window.getComputedStyle(sec).paddingTop,
        paddingBottom: window.getComputedStyle(sec).paddingBottom,
      },
      matched
    };
  });

  console.log('RECURSIVE STYLE INHERITANCE FOR SECTION:', JSON.stringify(matchedRules, null, 2));
  await browser.close();
})();
