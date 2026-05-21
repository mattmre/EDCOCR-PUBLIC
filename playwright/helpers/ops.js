const { hasEnv } = require("./runtime");
const { buildFeatureFlagHeaders, getExpectedFlag } = require("./feature-flags");

function hasPrimaryApiKey() {
  return Object.keys(buildPrimaryApiHeaders()).length > 0;
}

function buildPrimaryApiHeaders() {
  return buildFeatureFlagHeaders();
}

function hasMetricsKey() {
  return hasEnv("PLAYWRIGHT_METRICS_API_KEY");
}

function buildMetricsHeaders(mode = "x-api-key") {
  const key = process.env.PLAYWRIGHT_METRICS_API_KEY || "";
  if (!key.trim()) {
    return {};
  }

  if (mode === "bearer") {
    return {
      Authorization: `Bearer ${key.trim()}`,
    };
  }

  return {
    "X-Api-Key": key.trim(),
  };
}

module.exports = {
  buildMetricsHeaders,
  buildPrimaryApiHeaders,
  getExpectedFlag,
  hasMetricsKey,
  hasPrimaryApiKey,
};
