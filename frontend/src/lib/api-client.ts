import { clearApiKey, getApiKey } from "./auth";

/**
 * Base URL of the FastAPI backend. Configured at build time via
 * NEXT_PUBLIC_API_BASE_URL. Falls back to localhost for dev.
 */
const API_BASE_URL =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_BASE_URL) ||
  "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, message: string, body: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/**
 * UnauthorizedError is thrown specifically on 401/403. Callers (or a route
 * boundary) should clear the cached key and redirect to /login.
 */
export class UnauthorizedError extends ApiError {
  constructor(status: number, body: unknown = null) {
    super(status, status === 401 ? "Unauthorized" : "Forbidden", body);
    this.name = "UnauthorizedError";
  }
}

function buildUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const base = API_BASE_URL.replace(/\/+$/, "");
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

function buildHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }
  const apiKey = getApiKey();
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  return headers;
}

async function parseBody(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  try {
    return await response.text();
  } catch {
    return null;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (response.status === 401 || response.status === 403) {
    // Drop the cached key so subsequent requests don't loop on the same
    // failing credential. The route boundary / hook is responsible for the
    // actual redirect.
    clearApiKey();
    const body = await parseBody(response);
    throw new UnauthorizedError(response.status, body);
  }
  if (!response.ok) {
    const body = await parseBody(response);
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`;
    throw new ApiError(response.status, detail, body);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  const body = await parseBody(response);
  return body as T;
}

export interface RequestOptions {
  signal?: AbortSignal;
  headers?: HeadersInit;
}

export async function get<T = unknown>(path: string, options?: RequestOptions): Promise<T> {
  const response = await fetch(buildUrl(path), {
    method: "GET",
    headers: buildHeaders(options?.headers),
    signal: options?.signal,
    cache: "no-store",
  });
  return handleResponse<T>(response);
}

export async function post<T = unknown>(
  path: string,
  body: unknown,
  options?: RequestOptions
): Promise<T> {
  const headers = buildHeaders(options?.headers);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(buildUrl(path), {
    method: "POST",
    headers,
    signal: options?.signal,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return handleResponse<T>(response);
}

export async function put<T = unknown>(
  path: string,
  body: unknown,
  options?: RequestOptions
): Promise<T> {
  const headers = buildHeaders(options?.headers);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(buildUrl(path), {
    method: "PUT",
    headers,
    signal: options?.signal,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return handleResponse<T>(response);
}

export const apiClient = {
  get,
  post,
  put,
  baseUrl: API_BASE_URL,
};
