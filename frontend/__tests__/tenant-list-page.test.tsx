import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

import TenantsListPage from "@/app/admin/tenants/page";
import { setApiKey } from "@/lib/auth";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

interface JsonResponseInit {
  status?: number;
}

function jsonResponse(body: unknown, init?: JsonResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: init?.status ?? 200,
    headers: { "content-type": "application/json" },
  });
}

function notFoundResponse(): Response {
  return new Response(JSON.stringify({ detail: "tenant config not found" }), {
    status: 404,
    headers: { "content-type": "application/json" },
  });
}

const tenantSummary = (id: string) => ({
  tenant_id: id,
  name: id,
  display_name: null,
  status: "active",
  tier: "standard",
  created_at: "2026-04-01T00:00:00Z",
  updated_at: "2026-04-25T12:00:00Z",
  max_concurrent_jobs: 4,
  max_pages_per_month: 10000,
  max_storage_bytes: 0,
  allowed_features: [],
  admin_email: null,
});

const tenantConfig = (id: string) => ({
  tenant_id: id,
  target_languages: ["en", "fr"],
  preferred_engines: ["opus_mt"],
  allow_nc_licensed: false,
  require_certified: false,
  default_quality_tier: "standard",
  created_at: null,
  updated_at: null,
});

const glossaryList = (total: number) => ({
  entries: [],
  total,
  page: 1,
  page_size: 1,
});

describe("<TenantsListPage />", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders rows when /admin/tenants succeeds", async => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/api/v1/admin/tenants")) {
        return Promise.resolve(jsonResponse([tenantSummary("acme")]));
      }
      if (url.includes("/translation/tenants/acme/config")) {
        return Promise.resolve(jsonResponse(tenantConfig("acme")));
      }
      if (url.includes("/translation/tenants/acme/glossary")) {
        return Promise.resolve(jsonResponse(glossaryList(7)));
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<TenantsListPage />);

    await waitFor(() => {
      expect(screen.getByTestId("tenant-row-acme")).toBeInTheDocument();
    });

    expect(screen.getByTestId("tenant-link-acme")).toHaveAttribute(
      "href",
      "/admin/tenants/acme"
    );
    // Engine chip rendered.
    expect(screen.getByText("opus_mt")).toBeInTheDocument();
    // Glossary count rendered.
    expect(screen.getByText("7")).toBeInTheDocument();

    fetchMock.mockRestore();
  });

  it("falls back to manual mode on 404 from /admin/tenants", async => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/api/v1/admin/tenants")) {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: "Multi-tenancy disabled" }), {
            status: 404,
            headers: { "content-type": "application/json" },
          })
        );
      }
      if (url.includes("/translation/tenants/manual-co/config")) {
        return Promise.resolve(jsonResponse(tenantConfig("manual-co")));
      }
      if (url.includes("/translation/tenants/manual-co/glossary")) {
        return Promise.resolve(jsonResponse(glossaryList(0)));
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<TenantsListPage />);

    await waitFor(() => {
      expect(screen.getByTestId("tenants-manual-mode")).toBeInTheDocument();
    });

    fireEvent.change(screen.getByTestId("tenants-manual-input"), {
      target: { value: "manual-co" },
    });
    fireEvent.click(screen.getByTestId("tenants-manual-load"));

    await waitFor(() => {
      expect(screen.getByTestId("tenant-row-manual-co")).toBeInTheDocument();
    });
  });

  it("creates a tenant via the dialog and shows the new row", async => {
    let createdConfig: object | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.endsWith("/api/v1/admin/tenants") && method === "GET") {
        return Promise.resolve(jsonResponse([]));
      }
      if (url.includes("/translation/tenants/new-tenant/config") && method === "PUT") {
        const body = init?.body ? JSON.parse(String(init.body)) : null;
        createdConfig = body;
        return Promise.resolve(
          jsonResponse(
            tenantConfig("new-tenant"),
            { status: 200 }
          )
        );
      }
      if (url.includes("/translation/tenants/new-tenant/config") && method === "GET") {
        return Promise.resolve(jsonResponse(tenantConfig("new-tenant")));
      }
      if (url.includes("/translation/tenants/new-tenant/glossary")) {
        return Promise.resolve(jsonResponse(glossaryList(0)));
      }
      // Other paths -- 404.
      return Promise.resolve(notFoundResponse());
    });

    render(<TenantsListPage />);
    await waitFor(() => {
      expect(screen.getByTestId("tenants-empty")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("tenants-create-button"));
    fireEvent.change(screen.getByTestId("tenants-create-id"), {
      target: { value: "new-tenant" },
    });
    fireEvent.change(screen.getByTestId("tenants-create-langs"), {
      target: { value: "en, de" },
    });
    fireEvent.click(screen.getByTestId("tenants-create-submit"));

    await waitFor(() => {
      expect(screen.getByTestId("tenant-row-new-tenant")).toBeInTheDocument();
    });

    expect(createdConfig).toMatchObject({
      target_languages: ["en", "de"],
      allow_nc_licensed: false,
      require_certified: false,
      default_quality_tier: "standard",
    });
  });
});
