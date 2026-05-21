function hasEnv(name) {
  return Boolean(process.env[name] && process.env[name].trim());
}

function hasAllEnv(names) {
  return names.every(hasEnv);
}

function isTruthyEnv(name) {
  if (!hasEnv(name)) {
    return false;
  }

  return !["0", "false", "no", "off"].includes(process.env[name].trim().toLowerCase());
}

function buildApiHeaders(keyEnvName) {
  const envName = keyEnvName || "PLAYWRIGHT_OCR_API_KEY";
  const headers = {};
  const key = process.env[envName];

  if (key && key.trim()) {
    headers["X-API-Key"] = key.trim();
  }

  return headers;
}

module.exports = {
  hasEnv,
  hasAllEnv,
  isTruthyEnv,
  buildApiHeaders,
};
