"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/lib/api-client";
import {
  AUTH_STATE_EVENT,
  type AuthStateEventDetail,
  getApiKey,
  hasValidHttpOnlySession,
  isHttpOnlySessionEnabled,
  signOut,
} from "@/lib/auth";

/**
 * Role badge label derived from whether an API key is cached.
 *
 * The console is not a read-only viewer: operators with a key can submit
 * jobs, review, and mutate settings. Showing a static "read-only" badge
 * was dishonest (maintenance review). Until the API returns a real
 * role claim, key-present maps to "operator" and key-absent to "signed out".
 */
export function roleBadgeLabel(keyPresent: boolean): "operator" | "signed out" {
  return keyPresent ? "operator" : "signed out";
}

type AuthDisplayState =
  | "checking"
  | "authenticated"
  | "signed-out"
  | "unknown";

export function Topbar() {
  const router = useRouter();
  const [authState, setAuthState] = useState<AuthDisplayState>("checking");
  const [signingOut, setSigningOut] = useState(false);
  const [signOutError, setSignOutError] = useState<string | null>(null);
  const httpOnlySession = isHttpOnlySessionEnabled();

  useEffect(() => {
    let active = true;
    let authGeneration = 0;
    const onAuthState = (event: Event) => {
      const detail = (event as CustomEvent<AuthStateEventDetail>).detail;
      if (typeof detail?.authenticated === "boolean") {
        authGeneration += 1;
        setAuthState(detail.authenticated ? "authenticated" : "signed-out");
        setSignOutError(null);
      }
    };
    window.addEventListener(AUTH_STATE_EVENT, onAuthState);

    if (httpOnlySession) {
      const probeGeneration = authGeneration;
      void hasValidHttpOnlySession(apiClient.baseUrl)
        .then((valid) => {
          if (active && authGeneration === probeGeneration) {
            setAuthState(valid ? "authenticated" : "signed-out");
          }
        })
        .catch(() => {
          if (active && authGeneration === probeGeneration) {
            setAuthState("unknown");
          }
        });
    } else {
      setAuthState(getApiKey() !== null ? "authenticated" : "signed-out");
    }

    return () => {
      active = false;
      window.removeEventListener(AUTH_STATE_EVENT, onAuthState);
    };
  }, [httpOnlySession]);

  async function handleSignOut() {
    setSigningOut(true);
    setSignOutError(null);
    try {
      await signOut(apiClient.baseUrl);
      setAuthState("signed-out");
      router.push("/login");
    } catch {
      // Do not claim signed-out when the cookie-clearing request failed.
      setAuthState("unknown");
      setSignOutError("Sign out failed. Retry before leaving this browser session.");
    } finally {
      setSigningOut(false);
    }
  }

  const authenticated = authState === "authenticated";
  const canSignOut = authenticated || authState === "unknown";
  const badgeLabel = {
    checking: "checking session",
    authenticated: roleBadgeLabel(true),
    "signed-out": roleBadgeLabel(false),
    unknown: "session unknown",
  }[authState];
  const authSummary = {
    checking: "Checking authentication",
    authenticated: httpOnlySession ? "HttpOnly session" : "API key cached",
    "signed-out": "No API key",
    unknown: "Authentication unknown",
  }[authState];
  const cleanupLabel = !httpOnlySession && !authenticated;
  const actionLabel = signingOut
    ? cleanupLabel
      ? "Clearing session…"
      : "Signing out…"
    : cleanupLabel
      ? "Clear prior session"
      : "Sign out";

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-background px-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold">OCR_LOCAL Console</span>
        <span
          data-testid="topbar-role-badge"
          className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground"
        >
          {badgeLabel}
        </span>
      </div>
      <div className="flex items-center gap-3 text-sm">
        {signOutError ? (
          <span className="text-destructive" role="alert">
            {signOutError}
          </span>
        ) : null}
        <span className="text-muted-foreground">{authSummary}</span>
        {canSignOut ? (
          <Button
            variant="outline"
            size="sm"
            onClick={() => void handleSignOut()}
            disabled={signingOut}
          >
            {actionLabel}
          </Button>
        ) : authState === "checking" ? (
          <Button variant="outline" size="sm" disabled>
            Checking…
          </Button>
        ) : (
          <>
            <Button variant="outline" size="sm" onClick={() => router.push("/login")}>
              Sign in
            </Button>
            {!httpOnlySession ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => void handleSignOut()}
                disabled={signingOut}
              >
                Clear prior session
              </Button>
            ) : null}
          </>
        )}
      </div>
    </header>
  );
}
