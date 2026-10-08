"use client";

import { type FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { apiClient, probeApiKey } from "@/lib/api-client";
import {
  clearApiKey,
  hasValidHttpOnlySession,
  HttpOnlySessionError,
  isHttpOnlySessionEnabled,
  setApiKey,
  setApiKeyHttpOnly,
} from "@/lib/auth";
import { LOGIN_PROBE_PATH } from "@/lib/login-contract";

const STORAGE_UNAVAILABLE_MESSAGE =
  "Browser storage is unavailable — sign-in cannot be saved in this browser.";
const UNREACHABLE_MESSAGE =
  "Could not reach the API. Check your connection and try again.";
const POLICY_DENIED_MESSAGE =
  "API access is forbidden by deployment policy. Check the allowed-client settings or contact your administrator.";

export default function LoginPage() {
  const router = useRouter();
  const httpOnlySession = isHttpOnlySessionEnabled();
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [probing, setProbing] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    const trimmed = value.trim();
    if (trimmed.length === 0) {
      setError("API key cannot be empty");
      return;
    }

    setProbing(true);
    try {
      // Candidate-key probe via the centralized client (timeout + shared
      // header construction): never persists the candidate key. Cookie mode
      // also includes the existing cookie, which is checked separately below
      // if this candidate is rejected.
      let response: Response;
      try {
        response = await probeApiKey(LOGIN_PROBE_PATH, trimmed);
      } catch {
        // Network failure / CORS / offline / probe timeout — the API was not
        // reached, so nothing was persisted and prior auth state is unknown.
        // Do not navigate or clear credentials.
        setError(UNREACHABLE_MESSAGE);
        return;
      }

      if (response.status === 401) {
        if (httpOnlySession) {
          // The candidate header can be rejected while a previously issued
          // HttpOnly cookie remains valid. Resolve cookie state without the
          // candidate header before emitting a global signed-out transition.
          try {
            if (!(await hasValidHttpOnlySession(apiClient.baseUrl))) {
              clearApiKey();
            }
          } catch {
            // A failed cookie-only check is inconclusive; preserve auth state.
          }
        } else {
          clearApiKey();
        }
        setError("Invalid API key");
        return;
      }
      if (response.status === 403) {
        setError(POLICY_DENIED_MESSAGE);
        return;
      }

      if (!response.ok) {
        // 5xx or other non-auth failures: do not navigate, do not persist.
        setError(UNREACHABLE_MESSAGE);
        return;
      }

      // Network probe passed. Persisting the key is a separate local step
      // with its own error message: a storage failure here must NOT be
      // misreported as a network failure.
      try {
        if (httpOnlySession) {
          await setApiKeyHttpOnly(trimmed, apiClient.baseUrl);
        } else {
          setApiKey(trimmed);
        }
      } catch (err) {
        if (
          err instanceof HttpOnlySessionError &&
          err.stage === "create" &&
          err.status === 401
        ) {
          clearApiKey();
          setError("Invalid API key");
          return;
        }
        if (err instanceof HttpOnlySessionError && err.status === 403) {
          setError(POLICY_DENIED_MESSAGE);
          return;
        }
        if (err instanceof HttpOnlySessionError && err.stage === "validate") {
          if (err.status === 401) {
            clearApiKey();
            setError(
              "Could not establish a secure session. Check browser cookie and API origin settings, then try again."
            );
            return;
          }
          setError(UNREACHABLE_MESSAGE);
          return;
        }
        if (err instanceof Error && /storage/i.test(err.message)) {
          setError(STORAGE_UNAVAILABLE_MESSAGE);
          return;
        }
        setError(UNREACHABLE_MESSAGE);
        return;
      }
      router.push("/dashboard");
    } finally {
      setProbing(false);
    }
  }

  return (
    <div className="mx-auto flex min-h-[60vh] max-w-md items-center">
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Sign in</CardTitle>
          <CardDescription>
            {httpOnlySession
              ? "Paste an OCR_LOCAL operator API key. It will be exchanged for an HttpOnly session cookie and not stored in browser-readable storage."
              : "Paste an OCR_LOCAL operator API key. The key is stored only in this browser's sessionStorage (tab-scoped; not long-lived localStorage) and is sent as the X-API-Key header on every request."}
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-3">
            <label htmlFor="api-key" className="block text-sm font-medium">
              API key
            </label>
            <Input
              id="api-key"
              name="api-key"
              type="password"
              autoComplete="off"
              required
              disabled={probing}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              placeholder="ocr-..."
              aria-invalid={error ? true : undefined}
            />
            {error ? (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            ) : null}
          </CardContent>
          <CardFooter className="justify-end">
            <Button type="submit" disabled={probing}>
              {probing ? "Checking…" : "Continue"}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
