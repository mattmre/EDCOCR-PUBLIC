"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ApiError } from "@/lib/api-client";
import { submitChangeRequest } from "@/lib/feature-flags-api";
import type {
  CertifyAuthMethod,
  FeatureFlag,
  FeatureFlagHistoryEntry,
} from "@/lib/types";

const MIN_REASON_LENGTH = 20;

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

const METHOD_GUIDANCE: Record<CertifyAuthMethod, string> = {
  piv_cac:
    "Insert your PIV/CAC card and approve the credential prompt before retrying.",
  oidc_mfa:
    "Re-authenticate with your IdP and complete the second-factor challenge before retrying.",
  hardware_token:
    "Touch your hardware token to sign the assertion before retrying.",
};

export interface FlagChangeRequestDialogProps {
  flag: FeatureFlag;
  open: boolean;
  onClose: => void;
  onSubmitted: (entry: FeatureFlagHistoryEntry) => void;
}

/**
 * Modal that files a change request for a single flag. The request is
 * advisory -- the backend remains the only authority on whether the flip is
 * actually allowed. Strong-auth fields are conditional on
 * `flag.requires_strong_auth`.
 */
export function FlagChangeRequestDialog({
  flag,
  open,
  onClose,
  onSubmitted,
}: FlagChangeRequestDialogProps) {
  const [newValueRaw, setNewValueRaw] = useState<string>("");
  const [boolValue, setBoolValue] = useState<boolean>(false);
  const [reason, setReason] = useState<string>("");
  const [authMethod, setAuthMethod] = useState<CertifyAuthMethod | "">("");
  const [authToken, setAuthToken] = useState<string>("");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Re-seed the form whenever the dialog re-opens for a different flag.
  useEffect(() => {
    if (!open) return;
    if (flag.value_type === "boolean") {
      setBoolValue(flag.current_value !== true);
      setNewValueRaw("");
    } else if (flag.value_type === "integer") {
      setNewValueRaw(
        flag.current_value === null || flag.current_value === undefined
          ? ""
          : String(flag.current_value)
      );
    } else if (flag.value_type === "enum") {
      setNewValueRaw(
        flag.current_value === null || flag.current_value === undefined
          ? ""
          : String(flag.current_value)
      );
    } else {
      setNewValueRaw(
        flag.current_value === null || flag.current_value === undefined
          ? ""
          : String(flag.current_value)
      );
    }
    setReason("");
    setAuthMethod("");
    setAuthToken("");
    setError(null);
    setSubmitting(false);
  }, [open, flag.key, flag.value_type, flag.current_value]);

  const proposedValue: boolean | string | number | null = useMemo(() => {
    if (flag.value_type === "boolean") return boolValue;
    if (flag.value_type === "integer") {
      if (newValueRaw.trim() === "") return null;
      const n = Number(newValueRaw);
      return Number.isFinite(n) ? n : null;
    }
    if (newValueRaw === "") return null;
    return newValueRaw;
  }, [flag.value_type, boolValue, newValueRaw]);

  const valueDiffers = useMemo(() => {
    return JSON.stringify(proposedValue) !== JSON.stringify(flag.current_value);
  }, [proposedValue, flag.current_value]);

  const reasonValid = reason.trim().length >= MIN_REASON_LENGTH;

  const strongAuthValid = !flag.requires_strong_auth
    ? true
    : !!authMethod && authToken.trim().length > 0;

  const canSubmit =
    !submitting && reasonValid && valueDiffers && strongAuthValid;

  if (!open) return null;

  function handleClose() {
    if (submitting) return;
    onClose();
  }

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const entry = await submitChangeRequest(flag.key, {
        flag_key: flag.key,
        new_value: proposedValue,
        reason: reason.trim(),
        auth_method:
          flag.requires_strong_auth && authMethod ? authMethod : undefined,
        auth_token:
          flag.requires_strong_auth && authToken ? authToken : undefined,
      });
      onSubmitted(entry);
      onClose();
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        const body = err.body as { error_code?: string; detail?: string } | null;
        if (
          err.status === 403 &&
          body &&
          body.error_code === "strong_auth_required"
        ) {
          const guidance = authMethod
            ? `Strong authentication required (${METHOD_LABEL[authMethod]}). ${METHOD_GUIDANCE[authMethod]}`
            : "Strong authentication required. Pick an auth method and supply a token.";
          setError(guidance);
        } else if (err.status === 422) {
          setError(
            `Validation rejected: ${typeof body?.detail === "string" ? body.detail : err.message}`
          );
        } else {
          setError(`${err.status} ${err.message}`);
        }
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to submit change request");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="flag-change-heading"
      data-testid="flag-change-dialog"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <div className="w-full max-w-lg rounded-lg border border-border bg-background p-6 shadow-lg">
        <h2 id="flag-change-heading" className="text-lg font-semibold">
          Request flag change
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          Filing a change request appends a custody event. The backend may
          reject the request if strong-auth or bake requirements are not met.
        </p>

        <div className="mt-4 rounded border border-border bg-muted/30 p-3 text-xs">
          <div className="font-mono">{flag.key}</div>
          <div className="mt-1 text-muted-foreground">
            Current: <span className="font-mono">{String(flag.current_value)}</span>
            {" "}· Default: <span className="font-mono">{String(flag.default_value)}</span>
          </div>
        </div>

        <div className="mt-4 space-y-3">
          <label className="block text-xs font-medium text-muted-foreground">
            New value
            {flag.value_type === "boolean" ? (
              <div className="mt-1 flex gap-3" data-testid="flag-change-bool-group">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="flag-change-bool"
                    checked={boolValue === true}
                    onChange={() => setBoolValue(true)}
                    data-testid="flag-change-bool-on"
                    disabled={submitting}
                  />
                  ON
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="radio"
                    name="flag-change-bool"
                    checked={boolValue === false}
                    onChange={() => setBoolValue(false)}
                    data-testid="flag-change-bool-off"
                    disabled={submitting}
                  />
                  OFF
                </label>
              </div>
            ) : flag.value_type === "enum" ? (
              <select
                value={newValueRaw}
                onChange={(e) => setNewValueRaw(e.target.value)}
                disabled={submitting}
                data-testid="flag-change-enum"
                className="mt-1 block h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="">— select —</option>
                {(flag.allowed_values ?? []).map((v) => (
                  <option key={String(v)} value={String(v)}>
                    {String(v)}
                  </option>
                ))}
              </select>
            ) : (
              <Input
                type={flag.value_type === "integer" ? "number" : "text"}
                value={newValueRaw}
                onChange={(e) => setNewValueRaw(e.target.value)}
                disabled={submitting}
                data-testid="flag-change-value"
                className="mt-1"
              />
            )}
          </label>

          <label className="block text-xs font-medium text-muted-foreground">
            Reason (min {MIN_REASON_LENGTH} chars)
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              disabled={submitting}
              data-testid="flag-change-reason"
              rows={3}
              maxLength={2000}
              className="mt-1 block w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
            <span
              className="mt-1 block text-[10px] text-muted-foreground"
              data-testid="flag-change-reason-counter"
            >
              {reason.trim().length}/{MIN_REASON_LENGTH}
            </span>
          </label>

          {flag.requires_strong_auth ? (
            <fieldset
              className="space-y-2 rounded border border-amber-300 bg-amber-50/40 p-3"
              data-testid="flag-change-strongauth-section"
            >
              <legend className="text-xs font-medium text-amber-900">
                Strong-authentication required
              </legend>
              {AUTH_METHODS.map((opt) => (
                <label
                  key={opt.value}
                  className="flex cursor-pointer items-start gap-2 text-xs"
                >
                  <input
                    type="radio"
                    name="flag-change-auth-method"
                    checked={authMethod === opt.value}
                    onChange={() => setAuthMethod(opt.value)}
                    data-testid={`flag-change-auth-${opt.value}`}
                    disabled={submitting}
                    className="mt-0.5"
                  />
                  <span className="flex-1">
                    <span className="block font-medium">{opt.label}</span>
                    <span className="block text-[11px] text-muted-foreground">
                      {opt.help}
                    </span>
                  </span>
                </label>
              ))}
              <Input
                type="password"
                value={authToken}
                onChange={(e) => setAuthToken(e.target.value)}
                disabled={submitting}
                data-testid="flag-change-auth-token"
                placeholder="auth token / signed assertion"
                autoComplete="off"
              />
            </fieldset>
          ) : null}
        </div>

        {error ? (
          <p
            role="alert"
            className="mt-3 text-xs text-destructive"
            data-testid="flag-change-error"
          >
            {error}
          </p>
        ) : null}

        {!valueDiffers ? (
          <p
            className="mt-3 text-xs text-muted-foreground"
            data-testid="flag-change-novalue"
          >
            New value matches the current value. Pick a different value to
            file a change request.
          </p>
        ) : null}

        <div className="mt-5 flex justify-end gap-2">
          <Button
            variant="ghost"
            onClick={handleClose}
            disabled={submitting}
            data-testid="flag-change-cancel"
          >
            Cancel
          </Button>
          <Button
            variant="default"
            onClick={() => void handleSubmit()}
            disabled={!canSubmit}
            data-testid="flag-change-submit"
          >
            {submitting ? "Submitting…" : "Submit change request"}
          </Button>
        </div>
      </div>
    </div>
  );
}
