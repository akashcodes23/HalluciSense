const { chromium } = require('/Users/akashgpatil/major_project/frontend/node_modules/playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(1000);

  const matchedRule = await page.evaluate(() => {
    function findRuleRecursive(rules, selector) {
      for (let i = 0; i < rules.length; i++) {
        const rule = rules[i];
        if (rule.selectorText && rule.selectorText.includes(selector)) {
          return {
            selector: rule.selectorText,
            cssText: rule.cssText,
            parentLayer: rule.parentRule ? rule.parentRule.cssText.substring(0, 100) : 'none'
          };
        }
        if (rule.cssRules) {
          const found = findRuleRecursive(rule.cssRules, selector);
          if (found) return found;
        }
      }
      return null;
    }

    const results = [];
    for (let i = 0; i < document.styleSheets.length; i++) {
      const sheet = document.styleSheets[i];
      try {
        const found = findRuleRecursive(sheet.cssRules || sheet.rules, '.pt-32');
        if (found) {
          results.push({
            sheetHref: sheet.href || 'inline',
            rule: found
          });
        }
      } catch (e) {
        results.push({
          error: e.message,
          sheetHref: sheet.href || 'inline'
        });
      }
    }
    return results;
  });

  console.log('FOUND RULE IN CSSOM:', JSON.stringify(matchedRule, null, 2));
  await browser.close();
})();
