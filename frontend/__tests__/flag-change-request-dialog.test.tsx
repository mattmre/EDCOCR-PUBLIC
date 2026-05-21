import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { FlagChangeRequestDialog } from "@/components/FlagChangeRequestDialog";
import { setApiKey } from "@/lib/auth";
import type { FeatureFlag } from "@/lib/types";

function flag(overrides: Partial<FeatureFlag>): FeatureFlag {
  return {
    key: "ENABLE_TRANSLATION",
    category: "translation",
    value_type: "boolean",
    current_value: false,
    default_value: false,
    source: "default",
    description: "Master translation toggle.",
    requires_strong_auth: true,
    requires_bake_hours: 48,
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const VALID_REASON = "Routine canary rollout to validate flow.";

describe("<FlagChangeRequestDialog />", => {
  beforeEach(() => {
    setApiKey("test-key");
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does not render when open=false", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({})}
        open={false}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    expect(screen.queryByTestId("flag-change-dialog")).toBeNull();
  });

  it("seeds the bool value to the inverse of current_value when opened", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({ current_value: false })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    // current_value=false -> default proposed=true -> ON radio should be checked
    expect(
      (screen.getByTestId("flag-change-bool-on") as HTMLInputElement).checked
    ).toBe(true);
  });

  it("disables submit until the reason is at least 20 chars", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({ requires_strong_auth: false })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    expect(screen.getByTestId("flag-change-submit")).toBeDisabled();
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: "too short" },
    });
    expect(screen.getByTestId("flag-change-submit")).toBeDisabled();
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: VALID_REASON },
    });
    expect(screen.getByTestId("flag-change-submit")).not.toBeDisabled();
  });

  it("disables submit when the proposed value matches the current value", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({ requires_strong_auth: false, current_value: true })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: VALID_REASON },
    });
    // current=true, default proposed=false -> submit allowed
    expect(screen.getByTestId("flag-change-submit")).not.toBeDisabled();
    // Pick the matching value -> submit disabled, novalue notice shown
    fireEvent.click(screen.getByTestId("flag-change-bool-on"));
    expect(screen.getByTestId("flag-change-submit")).toBeDisabled();
    expect(screen.getByTestId("flag-change-novalue")).toBeInTheDocument();
  });

  it("renders strong-auth fields only when requires_strong_auth=true", => {
    const { rerender } = render(
      <FlagChangeRequestDialog
        flag={flag({ requires_strong_auth: false })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    expect(
      screen.queryByTestId("flag-change-strongauth-section")
    ).not.toBeInTheDocument();

    rerender(
      <FlagChangeRequestDialog
        flag={flag({ requires_strong_auth: true })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    expect(
      screen.getByTestId("flag-change-strongauth-section")
    ).toBeInTheDocument();
  });

  it("requires both auth method and token before submit when strong-auth is on", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({})}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: VALID_REASON },
    });
    expect(screen.getByTestId("flag-change-submit")).toBeDisabled();

    fireEvent.click(screen.getByTestId("flag-change-auth-piv_cac"));
    expect(screen.getByTestId("flag-change-submit")).toBeDisabled();

    fireEvent.change(screen.getByTestId("flag-change-auth-token"), {
      target: { value: "tok-abc" },
    });
    expect(screen.getByTestId("flag-change-submit")).not.toBeDisabled();
  });

  it("posts a complete payload on the happy path and calls onSubmitted", async => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        {
          request_id: "req_x",
          flag_key: "ENABLE_TRANSLATION",
          previous_value: false,
          new_value: true,
          reason: VALID_REASON,
          requested_by: "ops@example.com",
          requested_at: "2026-04-25T10:00:00Z",
          status: "pending",
        },
        202
      )
    );
    const onSubmitted = vi.fn();
    const onClose = vi.fn();

    render(
      <FlagChangeRequestDialog
        flag={flag({})}
        open={true}
        onClose={onClose}
        onSubmitted={onSubmitted}
      />
    );
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: VALID_REASON },
    });
    fireEvent.click(screen.getByTestId("flag-change-auth-oidc_mfa"));
    fireEvent.change(screen.getByTestId("flag-change-auth-token"), {
      target: { value: "tok-zzz" },
    });
    fireEvent.click(screen.getByTestId("flag-change-submit"));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toMatch(
      /\/admin\/feature-flags\/ENABLE_TRANSLATION\/change-request$/
    );
    const body = JSON.parse(String((init as RequestInit).body));
    expect(body).toMatchObject({
      flag_key: "ENABLE_TRANSLATION",
      new_value: true,
      reason: VALID_REASON,
      auth_method: "oidc_mfa",
      auth_token: "tok-zzz",
    });
    await waitFor(() => {
      expect(onSubmitted).toHaveBeenCalledTimes(1);
      expect(onClose).toHaveBeenCalled();
    });
  });

  it("surfaces a method-specific message when the backend returns 403 strong_auth_required", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        { detail: "strong auth", error_code: "strong_auth_required" },
        403
      )
    );
    render(
      <FlagChangeRequestDialog
        flag={flag({})}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: VALID_REASON },
    });
    fireEvent.click(screen.getByTestId("flag-change-auth-hardware_token"));
    fireEvent.change(screen.getByTestId("flag-change-auth-token"), {
      target: { value: "tok" },
    });
    fireEvent.click(screen.getByTestId("flag-change-submit"));

    await waitFor(() => {
      expect(screen.getByTestId("flag-change-error")).toHaveTextContent(
        /Strong authentication required.*hardware token/i
      );
    });
  });

  it("surfaces a validation error on 422", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "value out of range" }, 422)
    );
    render(
      <FlagChangeRequestDialog
        flag={flag({})}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: VALID_REASON },
    });
    fireEvent.click(screen.getByTestId("flag-change-auth-piv_cac"));
    fireEvent.change(screen.getByTestId("flag-change-auth-token"), {
      target: { value: "tok" },
    });
    fireEvent.click(screen.getByTestId("flag-change-submit"));

    await waitFor(() => {
      expect(screen.getByTestId("flag-change-error")).toHaveTextContent(
        /Validation rejected.*value out of range/i
      );
    });
  });

  it("renders an enum select for enum-typed flags", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({
          value_type: "enum",
          current_value: "core",
          allowed_values: ["core", "extended"],
          requires_strong_auth: false,
        })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    const select = screen.getByTestId("flag-change-enum") as HTMLSelectElement;
    expect(select).toBeInTheDocument();
    expect(Array.from(select.options).map((o) => o.value)).toEqual([
      "",
      "core",
      "extended",
    ]);
  });

  it("renders a numeric input for integer-typed flags", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({
          value_type: "integer",
          current_value: 12,
          default_value: 12,
          requires_strong_auth: false,
        })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    const input = screen.getByTestId("flag-change-value") as HTMLInputElement;
    expect(input.type).toBe("number");
    expect(input.value).toBe("12");
  });

  it("clears the form between opens for different flags", => {
    const { rerender } = render(
      <FlagChangeRequestDialog
        flag={flag({ key: "FLAG_A", current_value: false })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: "An older reason that was used previously." },
    });

    rerender(
      <FlagChangeRequestDialog
        flag={flag({ key: "FLAG_B", current_value: true })}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );

    const textarea = screen.getByTestId("flag-change-reason") as HTMLTextAreaElement;
    expect(textarea.value).toBe("");
  });

  it("shows a counter for the reason length", => {
    render(
      <FlagChangeRequestDialog
        flag={flag({})}
        open={true}
        onClose={() => {}}
        onSubmitted={() => {}}
      />
    );
    fireEvent.change(screen.getByTestId("flag-change-reason"), {
      target: { value: "abcdef" },
    });
    expect(screen.getByTestId("flag-change-reason-counter")).toHaveTextContent(
      "6/20"
    );
  });
});
