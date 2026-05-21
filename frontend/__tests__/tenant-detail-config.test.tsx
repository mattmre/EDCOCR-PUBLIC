import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

import TenantDetailPage from "@/app/admin/tenants/[tenantId]/page";
import { TenantConfigForm } from "@/components/TenantConfigForm";
import { setApiKey } from "@/lib/auth";
import type { TenantConfig } from "@/lib/types";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

const useParamsMock = vi.fn();
vi.mock("next/navigation", => ({
  useParams: => useParamsMock(),
  useRouter: => ({ push: vi.fn() }),
  usePathname: => "/admin/tenants/acme",
}));

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function makeConfig(): TenantConfig {
  return {
    tenant_id: "acme",
    target_languages: ["en"],
    preferred_engines: ["opus_mt"],
    allow_nc_licensed: false,
    require_certified: false,
    default_quality_tier: "standard",
    created_at: null,
    updated_at: null,
  };
}

describe("<TenantConfigForm />", => {
  it("requires NC opt-in before NLLB-200 can be selected", => {
    const onSubmit = vi.fn();
    render(
      <TenantConfigForm
        tenantId="acme"
        initial={makeConfig()}
        onSubmit={onSubmit}
      />
    );
    const nllbCheckbox = screen.getByTestId("engine-checkbox-nllb_200") as HTMLInputElement;
    expect(nllbCheckbox.disabled).toBe(true);
    // Help text is displayed.
    expect(screen.getByTestId("allow-nc-help")).toHaveTextContent(
      /non-commercial-use/i
    );
    // Toggle NC on; NLLB becomes enabled.
    fireEvent.click(screen.getByTestId("allow-nc-toggle"));
    expect(
      (screen.getByTestId("engine-checkbox-nllb_200") as HTMLInputElement).disabled
    ).toBe(false);
  });

  it("rejects an invalid BCP-47 tag and accepts a valid one", => {
    const onSubmit = vi.fn();
    render(
      <TenantConfigForm
        tenantId="acme"
        initial={makeConfig()}
        onSubmit={onSubmit}
      />
    );
    fireEvent.change(screen.getByTestId("target-language-input"), {
      target: { value: "12-bad" },
    });
    fireEvent.click(screen.getByTestId("target-language-add"));
    expect(screen.getByText(/not a valid BCP-47 tag/i)).toBeInTheDocument();

    fireEvent.change(screen.getByTestId("target-language-input"), {
      target: { value: "zh-Hans" },
    });
    fireEvent.click(screen.getByTestId("target-language-add"));
    // Chip rendered inside the chips container.
    const chips = screen.getByTestId("target-languages-chips");
    expect(chips).toHaveTextContent("zh-Hans");
  });

  it("calls onSubmit with the dirty payload on save", async => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(
      <TenantConfigForm
        tenantId="acme"
        initial={makeConfig()}
        onSubmit={onSubmit}
      />
    );
    fireEvent.click(screen.getByTestId("require-certified-toggle"));
    fireEvent.click(screen.getByTestId("tenant-config-save"));
    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledOnce();
    });
    const payload = onSubmit.mock.calls[0][0];
    expect(payload.require_certified).toBe(true);
    expect(payload.target_languages).toEqual(["en"]);
    expect(payload.preferred_engines).toEqual(["opus_mt"]);
  });

  it("disables Save when the form is pristine and re-enables it on edit", => {
    render(
      <TenantConfigForm
        tenantId="acme"
        initial={makeConfig()}
        onSubmit={vi.fn()}
      />
    );
    expect(
      (screen.getByTestId("tenant-config-save") as HTMLButtonElement).disabled
    ).toBe(true);

    fireEvent.click(screen.getByTestId("require-certified-toggle"));
    expect(
      (screen.getByTestId("tenant-config-save") as HTMLButtonElement).disabled
    ).toBe(false);
    expect(screen.getByTestId("tenant-config-dirty")).toBeInTheDocument();
  });
});

describe("<TenantDetailPage />", => {
  beforeEach(() => {
    setApiKey("test-key");
    useParamsMock.mockReturnValue({ tenantId: "acme" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads the tenant config and renders the Config tab by default", async => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.includes("/translation/tenants/acme/config")) {
        return Promise.resolve(jsonResponse(makeConfig()));
      }
      if (url.includes("/translation/tenants/acme/glossary")) {
        return Promise.resolve(
          jsonResponse({ entries: [], total: 0, page: 1, page_size: 100 })
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<TenantDetailPage />);

    await waitFor(() => {
      expect(screen.getByTestId("tenant-config-form")).toBeInTheDocument();
    });
    expect(screen.getByTestId("tenant-detail-id")).toHaveTextContent("acme");
  });

  it("PUTs the config when Save is clicked", async => {
    let putBody: unknown = null;
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      if (url.includes("/translation/tenants/acme/config") && method === "PUT") {
        putBody = init?.body ? JSON.parse(String(init.body)) : null;
        return Promise.resolve(
          jsonResponse({
            ...makeConfig(),
            require_certified: true,
          })
        );
      }
      if (url.includes("/translation/tenants/acme/config")) {
        return Promise.resolve(jsonResponse(makeConfig()));
      }
      if (url.includes("/translation/tenants/acme/glossary")) {
        return Promise.resolve(
          jsonResponse({ entries: [], total: 0, page: 1, page_size: 100 })
        );
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<TenantDetailPage />);
    await waitFor(() => {
      expect(screen.getByTestId("tenant-config-form")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("require-certified-toggle"));
    fireEvent.click(screen.getByTestId("tenant-config-save"));

    await waitFor(() => {
      expect(putBody).toMatchObject({ require_certified: true });
    });
  });
});
