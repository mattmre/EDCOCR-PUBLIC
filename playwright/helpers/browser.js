const { expect } = require("@playwright/test");

function _matchesAllowedPattern(text, allowPatterns) {
  return allowPatterns.some((pattern) => {
    if (pattern instanceof RegExp) {
      return pattern.test(text);
    }

    return text.includes(String(pattern));
  });
}

function trackConsoleErrors(page, options = {}) {
  const allowPatterns = options.allowPatterns || [];
  const errors = [];
  const listener = (message) => {
    if (message.type() !== "error") {
      return;
    }

    const text = message.text();
    if (_matchesAllowedPattern(text, allowPatterns)) {
      return;
    }

    errors.push(text);
  };

  page.on("console", listener);
  return {
    errors,
    stop() {
      page.off("console", listener);
    },
  };
}

async function expectNoUnexpectedConsoleErrors(tracker) {
  expect(
    tracker.errors,
    `Unexpected browser console errors:\n${tracker.errors.join("\n")}`).toEqual([]);
}

module.exports = {
  trackConsoleErrors,
  expectNoUnexpectedConsoleErrors,
};
