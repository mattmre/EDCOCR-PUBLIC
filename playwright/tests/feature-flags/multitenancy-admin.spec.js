const { test, expect } = require("@playwright/test");
const { hasEnv } = require("../../helpers/runtime");
const {
  buildPlatformAdminHeaders,
  getExpectedFlag,
  hasPlatformAdminKey,
} = require("../../helpers/feature-flags");

const MULTITENANCY_ENABLED = getExpectedFlag("PLAYWRIGHT_EXPECT_MULTITENANCY_ENABLED");

function uniqueTenantName(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

async function createTenant(request, baseURL, headers, name) {
  const response = await request.post(`${baseURL}/api/v1/admin/tenants`, {
    headers,
    data: {
      name,
      display_name: `${name} Display`,
      tier: "enterprise",
      max_concurrent_jobs: 6,
      max_pages_per_month: 20000,
      allowed_features: ["docintel", "transforms"],
      admin_email: `${name.toLowerCase()}@example.com`,
    },
  });

  expect(response.status()).toBe(201);
  return response.json();
}

async function createTenantAdminKey(request, baseURL, headers, tenantId, permissions = ["submit", "read", "admin"]) {
  const response = await request.post(`${baseURL}/api/v1/admin/tenants/${tenantId}/keys`, {
    headers,
    data: {
      name: "playwright-tenant-admin",
      permissions,
    },
  });

  expect(response.status()).toBe(201);
  return response.json();
}

test.describe("Phase 7 multi-tenancy admin matrix", => {
  test.skip(
    !hasEnv("PLAYWRIGHT_API_BASE_URL"),
    "Set PLAYWRIGHT_API_BASE_URL to run feature-flag request tests.");
  test.skip(
    MULTITENANCY_ENABLED === null,
    "Set PLAYWRIGHT_EXPECT_MULTITENANCY_ENABLED to declare the local multi-tenancy matrix state.");

  test("openapi reflects multi-tenancy admin route registration", async ({ request, baseURL }) => {
    const response = await request.get(`${baseURL}/openapi.json`);
    expect(response.status()).toBe(200);

    const payload = await response.json();
    const adminPaths = Object.keys(payload.paths).filter((path) => path.startsWith("/api/v1/admin/"));

    if (!MULTITENANCY_ENABLED) {
      expect(adminPaths).toEqual([]);
      return;
    }

    expect(adminPaths).not.toEqual([]);
    expect(adminPaths).toContain("/api/v1/admin/tenants");
  });

  test("platform admin can manage tenant lifecycle and usage", async ({ request, baseURL }) => {
    test.skip(!MULTITENANCY_ENABLED, "Tenant lifecycle coverage only applies when multi-tenancy is enabled.");
    test.skip(
      !hasPlatformAdminKey(),
      "Set PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY for live multi-tenancy admin coverage.");

    const headers = buildPlatformAdminHeaders();
    const tenant = await createTenant(request, baseURL, headers, uniqueTenantName("playwright-tenant"));

    const listResponse = await request.get(`${baseURL}/api/v1/admin/tenants`, { headers });
    expect(listResponse.status()).toBe(200);
    const tenants = await listResponse.json();
    expect(tenants.some((item) => item.tenant_id === tenant.tenant_id)).toBeTruthy();

    const detailResponse = await request.get(`${baseURL}/api/v1/admin/tenants/${tenant.tenant_id}`, { headers });
    expect(detailResponse.status()).toBe(200);
    const detail = await detailResponse.json();
    expect(detail.tenant_id).toBe(tenant.tenant_id);
    expect(detail.usage).toBeTruthy();

    const updateResponse = await request.put(`${baseURL}/api/v1/admin/tenants/${tenant.tenant_id}`, {
      headers,
      data: {
        display_name: "Playwright Updated",
        max_concurrent_jobs: 8,
        allowed_features: ["docintel", "transforms", "stamps"],
      },
    });
    expect(updateResponse.status()).toBe(200);
    const updated = await updateResponse.json();
    expect(updated.display_name).toBe("Playwright Updated");
    expect(updated.max_concurrent_jobs).toBe(8);

    const usageResponse = await request.get(`${baseURL}/api/v1/admin/tenants/${tenant.tenant_id}/usage`, { headers });
    expect(usageResponse.status()).toBe(200);
    const usage = await usageResponse.json();
    expect(usage.tenant_id).toBe(tenant.tenant_id);
    expect(usage.period).toMatch(/^\d{4}-\d{2}$/);

    const suspendResponse = await request.post(`${baseURL}/api/v1/admin/tenants/${tenant.tenant_id}/suspend`, { headers });
    expect(suspendResponse.status()).toBe(200);
    expect((await suspendResponse.json()).status).toBe("suspended");

    const activateResponse = await request.post(`${baseURL}/api/v1/admin/tenants/${tenant.tenant_id}/activate`, { headers });
    expect(activateResponse.status()).toBe(200);
    expect((await activateResponse.json()).status).toBe("active");
  });

  test("platform admin can create and revoke tenant API keys", async ({ request, baseURL }) => {
    test.skip(!MULTITENANCY_ENABLED, "Tenant key coverage only applies when multi-tenancy is enabled.");
    test.skip(
      !hasPlatformAdminKey(),
      "Set PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY for live multi-tenancy admin coverage.");

    const headers = buildPlatformAdminHeaders();
    const tenant = await createTenant(request, baseURL, headers, uniqueTenantName("playwright-key-tenant"));

    const keyPayload = await createTenantAdminKey(request, baseURL, headers, tenant.tenant_id);
    expect(keyPayload.key_id).toMatch(/^key_[0-9a-f]{12}$/);
    expect(keyPayload.api_key).toMatch(/^ocr_/);

    const revokeResponse = await request.delete(
      `${baseURL}/api/v1/admin/tenants/${tenant.tenant_id}/keys/${keyPayload.key_id}`,
      { headers });
    expect(revokeResponse.status()).toBe(204);
  });

  test("tenant admin remains self-scoped and cannot mint platform-admin keys", async ({ request, baseURL }) => {
    test.skip(!MULTITENANCY_ENABLED, "Tenant scope coverage only applies when multi-tenancy is enabled.");
    test.skip(
      !hasPlatformAdminKey(),
      "Set PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY for live multi-tenancy admin coverage.");

    const platformHeaders = buildPlatformAdminHeaders();
    const ownTenant = await createTenant(request, baseURL, platformHeaders, uniqueTenantName("playwright-own"));
    const otherTenant = await createTenant(request, baseURL, platformHeaders, uniqueTenantName("playwright-other"));
    const tenantAdminKey = await createTenantAdminKey(request, baseURL, platformHeaders, ownTenant.tenant_id);
    const tenantHeaders = { "X-API-Key": tenantAdminKey.api_key };

    const listResponse = await request.get(`${baseURL}/api/v1/admin/tenants`, { headers: tenantHeaders });
    expect(listResponse.status()).toBe(200);
    const visibleTenants = await listResponse.json();
    expect(visibleTenants).toHaveLength(1);
    expect(visibleTenants[0].tenant_id).toBe(ownTenant.tenant_id);

    const otherTenantResponse = await request.get(
      `${baseURL}/api/v1/admin/tenants/${otherTenant.tenant_id}`,
      { headers: tenantHeaders });
    expect(otherTenantResponse.status()).toBe(403);
    expect((await otherTenantResponse.json()).detail.error).toBe("forbidden");

    const createTenantResponse = await request.post(`${baseURL}/api/v1/admin/tenants`, {
      headers: tenantHeaders,
      data: {
        name: uniqueTenantName("playwright-forbidden"),
      },
    });
    expect(createTenantResponse.status()).toBe(403);
    expect((await createTenantResponse.json()).detail.error).toBe("forbidden");

    const mintPlatformKeyResponse = await request.post(
      `${baseURL}/api/v1/admin/tenants/${ownTenant.tenant_id}/keys`,
      {
        headers: tenantHeaders,
        data: {
          name: "playwright-escalation-attempt",
          permissions: ["submit", "read", "admin", "platform_admin"],
        },
      });
    expect(mintPlatformKeyResponse.status()).toBe(403);
    expect((await mintPlatformKeyResponse.json()).detail.error).toBe("forbidden");
  });
});
