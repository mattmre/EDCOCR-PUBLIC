"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ApiError, post } from "@/lib/api-client";
import type {
  CertifyAuthMethod,
  CertifyRequest,
  ReviewItem,
} from "@/lib/types";

const AUTH_METHODS: Array<{ value: CertifyAuthMethod; label: string; help: string }> = [
  {
    value: "piv_cac",
    label: "PIV / CAC",
    help: "Smartcard-based federal identity verification.",
  },
  {
    value: "oidc_mfa",
    label: "OIDC + MFA",
    help: "Identity provider login with second factor (TOTP / push).",
  },
  {
    value: "hardware_token",
    label: "Hardware token",
    help: "FIDO2 / WebAuthn-backed token signature.",
  },
];

const METHOD_LABEL: Record<CertifyAuthMethod, string> = {
  piv_cac: "PIV/CAC",
  oidc_mfa: "OIDC with MFA",
  hardware_token: "hardware token",
};

export interface CertifyDialogProps {
  reviewId: string;
  open: boolean;
  onClose: => void;
  /** Called with the updated item on successful certification. */
  onCertified: (updated: ReviewItem) => void;
}

/**
 * Strong-auth certification modal. Posts to /api/v1/review/{id}/certify.
 *
 * - Operator MUST pick an auth method explicitly (no silent default).
 * - On 401, surfaces the required method so the operator knows to refresh.
 * - The dialog warns that certification creates an immutable forensic event.
 */
export function CertifyDialog({
  reviewId,
  open,
  onClose,
  onCertified,
}: CertifyDialogProps) {
  const [method, setMethod] = useState<CertifyAuthMethod | "">("");
  const [authToken, setAuthToken] = useState<string>("");
  const [notes, setNotes] = useState<string>("");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  function reset() {
    setMethod("");
    setAuthToken("");
    setNotes("");
    setError(null);
    setSubmitting(false);
  }

  function handleClose() {
    if (submitting) return;
    reset();
    onClose();
  }

  async function handleConfirm() {
    if (!method) {
      setError("Select a strong-authentication method to continue.");
      return;
    }
    setSubmitting(true);
    setError(null);
    const body: CertifyRequest = {
      auth_method: method,
      auth_token: authToken,
      notes,
    };
    try {
      const updated = await post<ReviewItem>(
        `/api/v1/review/${encodeURIComponent(reviewId)}/certify`,
        body
      );
      onCertified(updated);
      reset();
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 401) {
        setError(
          `Strong authentication required. This action requires ${METHOD_LABEL[method]} verification.`
        );
      } else if (err instanceof ApiError) {
        setError(`${err.status} ${err.message}`);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Certification failed");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="certify-dialog-heading"
      data-testid="certify-dialog"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <div className="w-full max-w-lg rounded-lg border border-border bg-background p-6 shadow-lg">
        <h2
          id="certify-dialog-heading"
          className="text-lg font-semibold"
        >
          Certify translation
        </h2>
        <p
          className="mt-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900"
          data-testid="certify-warning"
        >
          Certification creates an immutable forensic event. This action cannot be undone.
        </p>

        <fieldset className="mt-4 space-y-2" aria-label="Strong authentication method">
          <legend className="text-sm font-medium">Strong authentication method</legend>
          {AUTH_METHODS.map((opt) => (
            <label
              key={opt.value}
              className="flex cursor-pointer items-start gap-2 rounded border border-border p-2 text-sm hover:bg-muted/40"
            >
              <input
                type="radio"
                name="certify-auth-method"
                value={opt.value}
                checked={method === opt.value}
                onChange={() => setMethod(opt.value)}
                data-testid={`certify-auth-${opt.value}`}
                disabled={submitting}
                className="mt-0.5"
              />
              <span className="flex-1">
                <span className="block font-medium">{opt.label}</span>
                <span className="block text-xs text-muted-foreground">{opt.help}</span>
              </span>
            </label>
          ))}
        </fieldset>

        <label className="mt-4 block text-xs font-medium text-muted-foreground">
          Auth token / assertion
          <input
            type="password"
            value={authToken}
            onChange={(e) => setAuthToken(e.target.value)}
            disabled={submitting}
            data-testid="certify-auth-token"
            className="mt-1 block h-10 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Auth token"
            autoComplete="off"
          />
        </label>

        <label className="mt-3 block text-xs font-medium text-muted-foreground">
          Certification notes (optional)
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            disabled={submitting}
            data-testid="certify-notes"
            rows={2}
            maxLength={2000}
            className="mt-1 block w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            aria-label="Certification notes"
          />
        </label>

        {error ? (
          <p
            role="alert"
            className="mt-3 text-xs text-destructive"
            data-testid="certify-error"
          >
            {error}
          </p>
        ) : null}

        <div className="mt-5 flex justify-end gap-2">
          <Button
            variant="ghost"
            onClick={handleClose}
            disabled={submitting}
            data-testid="certify-cancel"
          >
            Cancel
          </Button>
          <Button
            variant="default"
            onClick={() => void handleConfirm()}
            disabled={submitting || !method}
            data-testid="certify-confirm"
          >
            {submitting ? "Certifying..." : "Confirm Certification"}
          </Button>
        </div>
      </div>
    </div>
  );
}
