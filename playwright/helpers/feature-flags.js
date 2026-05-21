const { buildApiHeaders, hasEnv } = require("./runtime");

function getExpectedFlag(envName) {
  if (!hasEnv(envName)) {
    return null;
  }

  const raw = process.env[envName].trim().toLowerCase();
  return !["0", "false", "no", "off"].includes(raw);
}

function hasFeatureExecutionPaths() {
  return hasEnv("PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH")
    && hasEnv("PLAYWRIGHT_FEATURE_OUTPUT_DIR");
}

function getFeatureInputPath() {
  return process.env.PLAYWRIGHT_FEATURE_SAMPLE_INPUT_PATH || "";
}

function buildServerOutputPath(prefix) {
  const outputDir = process.env.PLAYWRIGHT_FEATURE_OUTPUT_DIR || "";
  if (!outputDir.trim()) {
    return "";
  }

  const trimmed = outputDir.replace(/[\\/]+$/, "");
  const separator = trimmed.includes("\\") ? "\\" : "/";
  const uniqueToken = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  return `${trimmed}${separator}${prefix}-${uniqueToken}.pdf`;
}

function buildFeatureFlagHeaders() {
  if (hasEnv("PLAYWRIGHT_FEATURE_FLAGS_API_KEY")) {
    return buildApiHeaders("PLAYWRIGHT_FEATURE_FLAGS_API_KEY");
  }

  return buildApiHeaders("PLAYWRIGHT_OCR_API_KEY");
}

function hasPlatformAdminKey() {
  return hasEnv("PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY");
}

function buildPlatformAdminHeaders() {
  return buildApiHeaders("PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY");
}

module.exports = {
  buildFeatureFlagHeaders,
  buildPlatformAdminHeaders,
  buildServerOutputPath,
  getExpectedFlag,
  getFeatureInputPath,
  hasFeatureExecutionPaths,
  hasPlatformAdminKey,
};
