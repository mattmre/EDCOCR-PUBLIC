import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import FeatureFlagsListPage from "@/app/admin/features/page";
import { FeatureFlagsList } from "@/components/FeatureFlagsList";
import { setApiKey } from "@/lib/auth";
import type { FeatureFlag } from "@/lib/types";

vi.mock("next/link", => ({
  __esModule: true,
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

function flag(overrides: Partial<FeatureFlag>): FeatureFlag {
  return {
    key: "FLAG_X",
    category: "pipeline",
    value_type: "boolean",
    current_value: false,
    default_value: false,
    source: "default",
    description: "Test flag.",
    requires_strong_auth: false,
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const SAMPLE_FLAGS: FeatureFlag[] = [
  flag({
    key: "ENABLE_TRANSLATION",
    category: "translation",
    current_value: false,
    requires_strong_auth: true,
    requires_bake_hours: 48,
    description: "Master translation toggle.",
  }),
  flag({
    key: "ENABLE_CUSTODY",
    category: "custody",
    current_value: true,
    description: "Hash-chain custody logging.",
    source: "config",
  }),
  flag({
    key: "OCR_LANGUAGE_TIERS",
    category: "pipeline",
    value_type: "enum",
    current_value: "core",
    default_value: "core",
    allowed_values: ["core", "extended"],
    source: "env",
    description: "Language tier selection.",
  }),
  flag({
    key: "WEBHOOK_SECRET_KEY",
    category: "operations",
    value_type: "string",
    current_value: "abcdef1234567890SECRETSTRING",
    default_value: null,
    source: "database",
    description: "Webhook HMAC secret.",
  }),
];

describe("<FeatureFlagsList /> grouping & rendering", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    if (typeof window !== "undefined") {
      window.localStorage.clear();
    }
  });

  it("renders rows grouped by category", => {
    render(<FeatureFlagsList flags={SAMPLE_FLAGS} />);
    expect(screen.getByTestId("flag-category-translation")).toBeInTheDocument();
    expect(screen.getByTestId("flag-category-custody")).toBeInTheDocument();
    expect(screen.getByTestId("flag-category-pipeline")).toBeInTheDocument();
    expect(screen.getByTestId("flag-category-operations")).toBeInTheDocument();
    expect(screen.getByTestId("flag-row-ENABLE_TRANSLATION")).toBeInTheDocument();
  });

  it("uses the correct source-badge color class for each source", => {
    render(<FeatureFlagsList flags={SAMPLE_FLAGS} />);
    expect(screen.getByTestId("flag-source-ENABLE_TRANSLATION")).toHaveClass(
      "bg-gray-100"
    );
    expect(screen.getByTestId("flag-source-ENABLE_CUSTODY")).toHaveClass(
      "bg-green-100"
    );
    expect(screen.getByTestId("flag-source-OCR_LANGUAGE_TIERS")).toHaveClass(
      "bg-blue-100"
    );
    expect(screen.getByTestId("flag-source-WEBHOOK_SECRET_KEY")).toHaveClass(
      "bg-purple-100"
    );
  });

  it("renders boolean ON/OFF pills", => {
    render(<FeatureFlagsList flags={SAMPLE_FLAGS} />);
    expect(screen.getByTestId("flag-value-ENABLE_TRANSLATION")).toHaveTextContent(
      "OFF"
    );
    expect(screen.getByTestId("flag-value-ENABLE_CUSTODY")).toHaveTextContent("ON");
  });

  it("masks values for keys that look secret-y", => {
    render(<FeatureFlagsList flags={SAMPLE_FLAGS} />);
    const cell = screen.getByTestId("flag-value-WEBHOOK_SECRET_KEY");
    // Original value is `abcdef1234567890SECRETSTRING`; should show first 2/last 2.
    expect(cell.textContent).not.toContain("1234567890");
    expect(cell.textContent).toMatch(/^ab.*NG$/);
  });

  it("shows a strong-auth badge when requires_strong_auth=true", => {
    render(<FeatureFlagsList flags={SAMPLE_FLAGS} />);
    expect(
      screen.getByTestId("flag-strongauth-ENABLE_TRANSLATION")
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("flag-strongauth-ENABLE_CUSTODY")
    ).not.toBeInTheDocument();
  });

  it("renders the View link with the flag key", => {
    render(<FeatureFlagsList flags={SAMPLE_FLAGS} />);
    expect(
      screen.getByTestId("flag-link-OCR_LANGUAGE_TIERS")
    ).toHaveAttribute("href", "/admin/features/OCR_LANGUAGE_TIERS");
  });

  it("collapses and expands a category and persists state to localStorage", => {
    render(<FeatureFlagsList flags={SAMPLE_FLAGS} />);
    const tableId = "flag-table-translation";
    expect(screen.getByTestId(tableId)).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("flag-category-toggle-translation"));
    expect(screen.queryByTestId(tableId)).not.toBeInTheDocument();
    const stored = window.localStorage.getItem("ocr-local:flags-ui");
    expect(stored).toBeTruthy();
    expect(JSON.parse(String(stored))).toMatchObject({
      collapsed: { translation: true },
    });
  });

  it("renders an empty state when no flags are passed", => {
    render(<FeatureFlagsList flags={[]} />);
    expect(screen.getByTestId("flags-empty")).toBeInTheDocument();
  });

  it("renders integer values as plain numbers", => {
    render(
      <FeatureFlagsList
        flags={[
          flag({
            key: "NUM_WORKERS",
            value_type: "integer",
            current_value: 12,
            default_value: 12,
          }),
        ]}
      />
    );
    expect(screen.getByTestId("flag-value-NUM_WORKERS")).toHaveTextContent("12");
  });
});

describe("<FeatureFlagsListPage /> backend fallbacks", => {
  beforeEach(() => {
    setApiKey("test-key");
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the 'API not provisioned' empty state on 404", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "not found" }, 404)
    );
    render(<FeatureFlagsListPage />);
    await waitFor(() => {
      expect(screen.getByTestId("flags-empty-state")).toBeInTheDocument();
    });
    expect(screen.getByText(/not yet provisioned/i)).toBeInTheDocument();
  });

  it("renders 'admin role required' on 403", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "forbidden" }, 403)
    );
    render(<FeatureFlagsListPage />);
    await waitFor(() => {
      expect(screen.getByTestId("flags-empty-state")).toBeInTheDocument();
    });
    expect(screen.getByText(/Platform admin role required/i)).toBeInTheDocument();
  });

  it("renders the flag table on 200 with envelope", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ flags: SAMPLE_FLAGS }, 200)
    );
    render(<FeatureFlagsListPage />);
    await waitFor(() => {
      expect(screen.getByTestId("flag-row-ENABLE_TRANSLATION")).toBeInTheDocument();
    });
  });
});
