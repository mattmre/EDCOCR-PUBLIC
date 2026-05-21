import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CertifyDialog } from "@/components/CertifyDialog";
import { setApiKey } from "@/lib/auth";
import type { ReviewItem } from "@/lib/types";

function itemFixture(partial: Partial<ReviewItem> = {}): ReviewItem {
  return {
    review_id: "rev_ccccccccccca",
    job_id: "job_xyz",
    reason: "low_confidence",
    confidence: 0.6,
    quality_classification: "review_required",
    status: "approved",
    reviewer: "operator@example.com",
    decision_notes: "",
    created_at: "2026-04-25T10:00:00Z",
    reviewed_at: "2026-04-25T10:05:00Z",
    metadata: { certified: true },
    ...partial,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("<CertifyDialog />", => {
  beforeEach(() => {
    setApiKey("test-key");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does not render when closed", => {
    render(
      <CertifyDialog
        reviewId="rev_x"
        open={false}
        onClose={() => {}}
        onCertified={() => {}}
      />
    );
    expect(screen.queryByTestId("certify-dialog")).toBeNull();
  });

  it("requires the operator to pick an auth method before confirming", async => {
    render(
      <CertifyDialog
        reviewId="rev_ccccccccccca"
        open={true}
        onClose={() => {}}
        onCertified={() => {}}
      />
    );

    expect(screen.getByTestId("certify-warning")).toBeInTheDocument();
    // Confirm is disabled until an auth method is selected.
    expect(screen.getByTestId("certify-confirm")).toBeDisabled();

    fireEvent.click(screen.getByTestId("certify-auth-piv_cac"));
    await waitFor(() =>
      expect(screen.getByTestId("certify-confirm")).not.toBeDisabled()
    );
  });

  it("posts to /certify with the selected method and notifies on success", async => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(itemFixture()));
    const onCertified = vi.fn();
    const onClose = vi.fn();

    render(
      <CertifyDialog
        reviewId="rev_ccccccccccca"
        open={true}
        onClose={onClose}
        onCertified={onCertified}
      />
    );

    fireEvent.click(screen.getByTestId("certify-auth-oidc_mfa"));
    fireEvent.change(screen.getByTestId("certify-auth-token"), {
      target: { value: "tok-xyz" },
    });
    fireEvent.click(screen.getByTestId("certify-confirm"));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toMatch(/\/api\/v1\/review\/rev_ccccccccccca\/certify$/);
    expect((init as RequestInit).method).toBe("POST");
    const body = JSON.parse(String((init as RequestInit).body));
    expect(body).toMatchObject({
      auth_method: "oidc_mfa",
      auth_token: "tok-xyz",
    });

    await waitFor(() => {
      expect(onCertified).toHaveBeenCalledTimes(1);
    });
    const updated = onCertified.mock.calls[0]?.[0] as ReviewItem;
    expect(updated.metadata.certified).toBe(true);
    expect(onClose).toHaveBeenCalled();
  });

  it("shows a method-specific message when the backend returns 401", async => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse({ detail: "strong auth required" }, 401)
    );

    render(
      <CertifyDialog
        reviewId="rev_ccccccccccca"
        open={true}
        onClose={() => {}}
        onCertified={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("certify-auth-hardware_token"));
    fireEvent.click(screen.getByTestId("certify-confirm"));

    await waitFor(() => {
      expect(screen.getByTestId("certify-error")).toHaveTextContent(
        /Strong authentication required.*hardware token/i
      );
    });
  });
});
