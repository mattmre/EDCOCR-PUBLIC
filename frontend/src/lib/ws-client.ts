"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { resolveApiBaseUrl, resolveConfiguredApiBaseUrl } from "./api-client";
import { getApiKey, isHttpOnlySessionEnabled } from "./auth";
import type { JobWSMessage, WSStatus } from "./types";

const WEBSOCKET_BASE_PROTOCOLS = new Set(["http:", "https:", "ws:", "wss:"]);

function currentPageOrigin(): string | undefined {
  return typeof window !== "undefined" && window.location?.origin
    ? window.location.origin
    : undefined;
}

function selectWebSocketBaseUrl(baseUrl?: string): string {
  if (baseUrl !== undefined) {
    return baseUrl.trim();
  }

  return resolveConfiguredApiBaseUrl() ?? currentPageOrigin() ?? resolveApiBaseUrl();
}

function parseWebSocketBaseUrl(baseUrl: string, pageOrigin?: string): URL {
  let parsed: URL;
  try {
    if (!baseUrl) {
      if (!pageOrigin) {
        throw new Error("missing page origin");
      }
      parsed = new URL(pageOrigin);
    } else if (baseUrl.startsWith("/") && !baseUrl.startsWith("//")) {
      if (!pageOrigin) {
        throw new Error("missing page origin");
      }
      parsed = new URL(baseUrl, pageOrigin);
    } else {
      parsed = new URL(baseUrl);
    }
  } catch {
    throw new Error("Invalid WebSocket base URL");
  }

  if (!WEBSOCKET_BASE_PROTOCOLS.has(parsed.protocol)) {
    throw new Error("Invalid WebSocket base URL");
  }
  if (parsed.search || parsed.hash) {
    throw new Error("Invalid WebSocket base URL");
  }
  return parsed;
}

/**
 * Compute the WebSocket URL for a given job.
 *
 * Derives a ws:// (or wss://) URL from the shared API base resolver
 * (settings → env → localhost), or an explicit override. When the resolved
 * HTTP base is unavailable in-browser without env, falls back to the current
 * page origin. Credentials are sent via the auth frame (maintenance review), not query.
 */
export function buildJobWebSocketUrl(jobId: string, baseUrl?: string): string {
  const pageOrigin = currentPageOrigin();
  const parsed = parseWebSocketBaseUrl(selectWebSocketBaseUrl(baseUrl), pageOrigin);

  parsed.protocol = parsed.protocol === "https:" || parsed.protocol === "wss:" ? "wss:" : "ws:";
  const basePath = parsed.pathname.replace(/\/+$/, "");
  parsed.pathname = `${basePath}/ws/jobs/${encodeURIComponent(jobId)}`;

  const output = parsed.toString();
  if (!/^wss?:\/\//.test(output)) {
    throw new Error("Invalid WebSocket URL");
  }
  return output;
}

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;
const MAX_BUFFERED_MESSAGES = 200;
const COOKIE_AUTH_PROTOCOL = "ocr-local.auth.cookie.v1";
const FRAME_AUTH_PROTOCOL = "ocr-local.auth.frame.v1";
const AUTH_PROTOCOL_MISMATCH_CLOSE_CODE = 4002;
const AUTH_PROTOCOL_MISMATCH_REASON = "WebSocket authentication protocol mismatch";
const AUTHENTICATION_FAILURE_CLOSE_CODE = 4001;

export interface UseJobWebSocketResult {
  status: WSStatus;
  messages: JobWSMessage[];
  lastMessage: JobWSMessage | null;
  reconnect: () => void;
  send: (data: string | Record<string, unknown>) => void;
}

export interface UseJobWebSocketOptions {
  /** Inject a custom WebSocket constructor. Used by tests with a mock. */
  webSocketImpl?: typeof WebSocket;
  /** Override base URL (e.g. when running the API on a different host). */
  baseUrl?: string;
  /** Inject the API key (default: read from auth.ts). */
  apiKey?: string | null;
  /** Disable auto reconnect (default: enabled). */
  reconnect?: boolean;
}

interface Backoff {
  attempt: number;
}

function nextBackoffDelay(attempt: number): number {
  const base = Math.min(RECONNECT_BASE_MS * 2 ** attempt, RECONNECT_MAX_MS);
  // Add modest jitter (+/- 20%) to avoid thundering herd.
  const jitter = 0.8 + Math.random() * 0.4;
  return Math.round(base * jitter);
}

/**
 * React hook that manages a WebSocket connection to /ws/jobs/{jobId}.
 *
 * - Negotiates an explicit frame or HttpOnly-cookie authentication mode.
 * - Reconnects with exponential backoff (1s, 2s, 4s, 8s, ..., capped at 30s).
 * - Pauses (closes) when the document becomes hidden and reconnects when
 *   it becomes visible again.
 * - Cleans up on unmount.
 */
export function useJobWebSocket(
  jobId: string,
  options: UseJobWebSocketOptions = {}
): UseJobWebSocketResult {
  const [status, setStatus] = useState<WSStatus>("idle");
  const [messages, setMessages] = useState<JobWSMessage[]>([]);
  const [lastMessage, setLastMessage] = useState<JobWSMessage | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef<Backoff>({ attempt: 0 });
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedByUserRef = useRef<boolean>(false);
  const visibilityPausedRef = useRef<boolean>(false);
  const authenticationRejectedRef = useRef<boolean>(false);

  const reconnectEnabled = options.reconnect !== false;
  const WSImpl = options.webSocketImpl ?? (typeof WebSocket !== "undefined" ? WebSocket : undefined);
  const explicitApiKeyProvided = Object.prototype.hasOwnProperty.call(options, "apiKey");

  const cleanupSocket = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    const ws = wsRef.current;
    if (ws) {
      try {
        ws.onopen = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.onmessage = null;
        if (
          ws.readyState === 0 /* CONNECTING */ ||
          ws.readyState === 1 /* OPEN */
        ) {
          ws.close();
        }
      } catch {
        // Ignore -- the socket is being torn down.
      }
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!WSImpl || !jobId || authenticationRejectedRef.current) {
      return;
    }
    cleanupSocket();
    closedByUserRef.current = false;
    setStatus(backoffRef.current.attempt > 0 ? "reconnecting" : "connecting");

    // Snapshot the credential mode before each handshake. An explicitly
    // supplied empty value stays in frame mode and cannot fall back to cookie.
    const cookieAuthSelected = isHttpOnlySessionEnabled() && !explicitApiKeyProvided;
    const authProtocol = cookieAuthSelected ? COOKIE_AUTH_PROTOCOL : FRAME_AUTH_PROTOCOL;
    const frameApiKey = cookieAuthSelected
      ? null
      : explicitApiKeyProvided
        ? (options.apiKey ?? "")
        : (getApiKey() ?? "");

    let ws: WebSocket;
    try {
      ws = new WSImpl(buildJobWebSocketUrl(jobId, options.baseUrl), authProtocol);
    } catch {
      setStatus("error");
      scheduleReconnect();
      return;
    }
    wsRef.current = ws;
    let protocolRejected = false;

    ws.onopen = () => {
      if (ws.protocol !== authProtocol) {
        protocolRejected = true;
        setStatus("error");
        try {
          // Browser APIs reserve close code 1002 for the user agent.
          ws.close(AUTH_PROTOCOL_MISMATCH_CLOSE_CODE, AUTH_PROTOCOL_MISMATCH_REASON);
        } catch {
          try {
            ws.close();
          } catch {
            // Keep the socket credential-silent even if teardown fails.
          }
        }
        return;
      }
      setStatus("authenticating");
      // Native browser WebSockets carry the cookie. Cookie mode sends no
      // placeholder auth frame; frame mode sends the captured credential.
      if (frameApiKey === null) {
        return;
      }
      // Send auth frame first; backend rejects all other frames before auth.
      try {
        ws.send(JSON.stringify({ type: "auth", api_key: frameApiKey }));
      } catch {
        // If send fails, the onclose handler will trigger reconnect.
      }
      // Promote to "open" once we have actually exchanged an auth frame.
      // The server responds with the first "connected" message on success.
    };

    ws.onmessage = (event: MessageEvent) => {
      let parsed: JobWSMessage | null = null;
      try {
        const data = typeof event.data === "string" ? event.data : "";
        parsed = JSON.parse(data) as JobWSMessage;
      } catch {
        return;
      }
      if (!parsed || typeof parsed !== "object" || !("type" in parsed)) {
        return;
      }
      if (parsed.type === "connected") {
        backoffRef.current = { attempt: 0 };
        setStatus("open");
      }
      setLastMessage(parsed);
      setMessages((prev) => {
        const next = [...prev, parsed as JobWSMessage];
        return next.length > MAX_BUFFERED_MESSAGES
          ? next.slice(next.length - MAX_BUFFERED_MESSAGES)
          : next;
      });
    };

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onclose = (event: CloseEvent) => {
      wsRef.current = null;
      if (protocolRejected) {
        setStatus("error");
        return;
      }
      if (event.code === AUTHENTICATION_FAILURE_CLOSE_CODE) {
        // Authentication failures and session logout are terminal until the
        // user explicitly reconnects or switches to another job. Retrying the
        // rejected credential only creates a loop of failed handshakes.
        authenticationRejectedRef.current = true;
        setStatus("error");
        return;
      }
      if (closedByUserRef.current) {
        setStatus("closed");
        return;
      }
      if (visibilityPausedRef.current) {
        setStatus("closed");
        return;
      }
      if (!reconnectEnabled) {
        setStatus("closed");
        return;
      }
      scheduleReconnect();
    };
    // Inline scheduleReconnect to keep closure references explicit.
    function scheduleReconnect() {
      const delay = nextBackoffDelay(backoffRef.current.attempt);
      backoffRef.current.attempt += 1;
      setStatus("reconnecting");
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      reconnectTimerRef.current = setTimeout(() => {
        connect();
      }, delay);
    }
  }, [
    WSImpl,
    jobId,
    options.apiKey,
    options.baseUrl,
    reconnectEnabled,
    cleanupSocket,
    explicitApiKeyProvided,
    authenticationRejectedRef,
  ]);

  // (Re)connect when the jobId changes, and tear down on unmount.
  useEffect(() => {
    if (!jobId) {
      return;
    }
    authenticationRejectedRef.current = false;
    backoffRef.current = { attempt: 0 };
    setMessages([]);
    setLastMessage(null);
    connect();
    return () => {
      closedByUserRef.current = true;
      cleanupSocket();
    };
    // We deliberately depend on jobId only -- connect is stable per render
    // because of useCallback dependencies, and re-running it would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  // Pause when the tab is hidden; reconnect when it becomes visible again.
  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    function onVisibilityChange() {
      if (document.visibilityState === "hidden") {
        visibilityPausedRef.current = true;
        cleanupSocket();
        setStatus("closed");
      } else if (document.visibilityState === "visible") {
        if (visibilityPausedRef.current) {
          visibilityPausedRef.current = false;
          backoffRef.current = { attempt: 0 };
          connect();
        }
      }
    }
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [cleanupSocket, connect]);

  const reconnect = useCallback(() => {
    backoffRef.current = { attempt: 0 };
    closedByUserRef.current = false;
    visibilityPausedRef.current = false;
    authenticationRejectedRef.current = false;
    connect();
  }, [connect]);

  const send = useCallback((data: string | Record<string, unknown>) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== 1) {
      return;
    }
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    try {
      ws.send(payload);
    } catch {
      // Ignored -- onclose will surface the failure.
    }
  }, []);

  return { status, messages, lastMessage, reconnect, send };
}

export const __testing = {
  nextBackoffDelay,
  RECONNECT_BASE_MS,
  RECONNECT_MAX_MS,
};
