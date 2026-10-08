import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import LoginPage from "@/app/login/page";
import { Topbar } from "@/components/topbar";
import { AUTH_STATE_EVENT, AUTH_STORAGE_KEY } from "@/lib/auth";
import { LOGIN_PROBE_PATH } from "@/lib/login-contract";
import {
  SETTINGS_STORAGE_KEY,
  getDefaultSettings,
  saveSettings,
} from "@/lib/settings-store";

const pushMock = vi.fn();
const originalHttpOnlySession = process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
  }),
}));

function jsonResponse(status: number, body: unknown = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("LoginPage", () => {
  beforeEach(() => {
    pushMock.mockClear();
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "false";
    window.sessionStorage.clear();
    window.localStorage.clear();
    vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
    if (originalHttpOnlySession === undefined) {
      delete process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;
    } else {
      process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = originalHttpOnlySession;
    }
  });

  it("renders the API key input and submit button", () => {
    render(<LoginPage />);
    expect(screen.getByLabelText(/api key/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeInTheDocument();
  });

  it("uses the default sessionStorage/X-API-Key path and routes on 200", async () => {
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { jobs: [] })
    );
    const user = userEvent.setup();
    render(<LoginPage />);
    const input = screen.getByLabelText(/api key/i);
    await user.type(input, "ocr-login-test");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith("/dashboard");
    });

    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("ocr-login-test");
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/auth/session/probe");
    expect(url).not.toContain("/api/v1/health");
    expect(LOGIN_PROBE_PATH).not.toContain("health");
    expect(String(url)).not.toMatch(/health/i);

    const headers = new Headers(init.headers);
    expect(headers.get("X-API-Key")).toBe("ocr-login-test");
    expect(headers.get("Accept")).toBe("application/json");
    expect(init.method).toBe("GET");
    expect(init.credentials).toBe("omit");
  });

  it("establishes an HttpOnly session after the credentialed probe", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const cookieLoginKey = "ocr-cookie-login";
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }))
      .mockResolvedValueOnce(jsonResponse(200, { storage: "httpOnly_cookie" }))
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }));
    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), cookieLoginKey);
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith("/dashboard");
    });

    expect(fetchMock).toHaveBeenCalledTimes(3);
    const [probeUrl, probeInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(probeUrl).toContain("/api/v1/auth/session/probe");
    expect(probeInit.credentials).toBe("include");
    expect(new Headers(probeInit.headers).get("X-API-Key")).toBe(cookieLoginKey);

    const [sessionUrl, sessionInit] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(sessionUrl).toBe("http://localhost:8000/api/v1/auth/session");
    expect(sessionInit.method).toBe("POST");
    expect(sessionInit.credentials).toBe("include");
    expect(new Headers(sessionInit.headers).get("Content-Type")).toBe(
      "application/json"
    );
    expect(sessionInit.body).toBe(JSON.stringify({ api_key: cookieLoginKey }));
    expect(new Headers(sessionInit.headers).get("X-API-Key")).toBeNull();
    const [validationUrl, validationInit] = fetchMock.mock.calls[2] as [string, RequestInit];
    expect(validationUrl).toBe("http://localhost:8000/api/v1/auth/session/probe");
    expect(validationInit.method).toBe("GET");
    expect(validationInit.credentials).toBe("include");
    expect(new Headers(validationInit.headers).has("X-API-Key")).toBe(false);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("does not navigate when the issued cookie fails the protected auth probe", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }))
      .mockResolvedValueOnce(jsonResponse(200, { storage: "httpOnly_cookie" }))
      .mockResolvedValueOnce(jsonResponse(401, { error: "invalid_api_key" }));
    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "accepted-but-cookie-blocked");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(await screen.findByText(/could not establish a secure session/i)).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("reports policy denial when protected cookie validation returns 403", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }))
      .mockResolvedValueOnce(jsonResponse(200, { storage: "httpOnly_cookie" }))
      .mockResolvedValueOnce(jsonResponse(403, { error: "origin_forbidden" }));
    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "accepted-but-policy-blocked");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(await screen.findByText(/forbidden by deployment policy/i)).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("keeps the operator on login when HttpOnly session creation is rejected", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }))
      .mockResolvedValueOnce(jsonResponse(401, { error: "invalid_key" }));
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "stale-key");

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-bad-cookie-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/invalid api key/i);
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("shows Invalid API key on 401 and does not navigate or persist", async () => {
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(401, { detail: "Unauthorized" })
    );
    // Pre-seed a stale key to ensure failure clears it.
    window.localStorage.setItem(AUTH_STORAGE_KEY, "stale-key");

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-bad-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/invalid api key/i);
    });

    expect(pushMock).not.toHaveBeenCalled();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0] as [string];
    expect(url).toContain("/api/v1/auth/session/probe");
    expect(url).not.toContain("/health");
  });

  it("preserves a valid HttpOnly session when a replacement candidate is rejected", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "existing-session-marker");
    const authTransitions: boolean[] = [];
    const onAuthState = (event: Event) => {
      authTransitions.push(
        (event as CustomEvent<{ authenticated: boolean }>).detail.authenticated
      );
    };
    window.addEventListener(AUTH_STATE_EVENT, onAuthState);
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { authenticated: true }))
      .mockResolvedValueOnce(jsonResponse(401, { error: "invalid_api_key" }))
      .mockResolvedValueOnce(jsonResponse(200, { authenticated: true }));

    const user = userEvent.setup();
    render(
      <>
        <Topbar />
        <LoginPage />
      </>
    );
    expect(await screen.findByText("HttpOnly session")).toBeInTheDocument();
    await user.type(screen.getByLabelText(/api key/i), "invalid-replacement-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(await screen.findByText(/invalid api key/i)).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
    expect(screen.getByText("HttpOnly session")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    const [candidateUrl, candidateInit] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(candidateUrl).toContain("/api/v1/auth/session/probe");
    expect(candidateInit.credentials).toBe("include");
    expect(new Headers(candidateInit.headers).get("X-API-Key")).toBe(
      "invalid-replacement-key"
    );
    const [sessionUrl, sessionInit] = fetchMock.mock.calls[2] as [string, RequestInit];
    expect(sessionUrl).toContain("/api/v1/auth/session/probe");
    expect(sessionInit.credentials).toBe("include");
    expect(new Headers(sessionInit.headers).has("X-API-Key")).toBe(false);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe(
      "existing-session-marker"
    );
    expect(authTransitions).not.toContain(false);
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).endsWith("/auth/session"))
    ).toBe(false);
    window.removeEventListener(AUTH_STATE_EVENT, onAuthState);
  });

  it("does not clear auth state when cookie revalidation after 401 is unavailable", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const authTransitions: boolean[] = [];
    const onAuthState = (event: Event) => {
      authTransitions.push(
        (event as CustomEvent<{ authenticated: boolean }>).detail.authenticated
      );
    };
    window.addEventListener(AUTH_STATE_EVENT, onAuthState);
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { authenticated: true }))
      .mockResolvedValueOnce(jsonResponse(401, { error: "invalid_api_key" }))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"));

    const user = userEvent.setup();
    render(
      <>
        <Topbar />
        <LoginPage />
      </>
    );
    expect(await screen.findByText("HttpOnly session")).toBeInTheDocument();
    await user.type(screen.getByLabelText(/api key/i), "invalid-replacement-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(await screen.findByText(/invalid api key/i)).toBeInTheDocument();
    expect(screen.getByText("HttpOnly session")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(authTransitions).not.toContain(false);
    window.removeEventListener(AUTH_STATE_EVENT, onAuthState);
  });

  it("reports signed out when both candidate and cookie-only probes return 401", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "stale-session-marker");
    const authTransitions: boolean[] = [];
    const onAuthState = (event: Event) => {
      authTransitions.push(
        (event as CustomEvent<{ authenticated: boolean }>).detail.authenticated
      );
    };
    window.addEventListener(AUTH_STATE_EVENT, onAuthState);
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(401, { error: "unauthorized" }))
      .mockResolvedValueOnce(jsonResponse(401, { error: "invalid_api_key" }))
      .mockResolvedValueOnce(jsonResponse(401, { error: "unauthorized" }));

    const user = userEvent.setup();
    render(
      <>
        <Topbar />
        <LoginPage />
      </>
    );
    expect(await screen.findByText("No API key")).toBeInTheDocument();
    await user.type(screen.getByLabelText(/api key/i), "invalid-replacement-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(await screen.findByText(/invalid api key/i)).toBeInTheDocument();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(authTransitions).toContain(false);
    expect(screen.getByRole("button", { name: "Sign in" })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    window.removeEventListener(AUTH_STATE_EVENT, onAuthState);
  });

  it("preserves a stored key and explains policy denial on 403", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(403, { detail: "Forbidden" })
    );
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "previous-key");

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-forbidden");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/forbidden by deployment policy/i);
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("previous-key");
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("preserves prior auth state when the HttpOnly preflight cannot reach the API", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
      new TypeError("Failed to fetch")
    );
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "previous-key");

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-candidate-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/could not reach the api/i);
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("previous-key");
  });

  it("preserves prior auth state when HttpOnly session creation has a server error", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }))
      .mockResolvedValueOnce(jsonResponse(503, { error: "unavailable" }));
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "previous-key");

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-candidate-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/could not reach the api/i);
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("previous-key");
  });

  it("preserves prior auth state when HttpOnly session creation is forbidden", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const fetchMock = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(jsonResponse(200, { jobs: [] }))
      .mockResolvedValueOnce(jsonResponse(403, { detail: "Origin forbidden" }));
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, "previous-key");

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-allowed-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/forbidden by deployment policy/i);
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBe("previous-key");
  });

  it("shows a connection error on network failure and does not navigate", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
      new TypeError("Failed to fetch")
    );

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-offline");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/could not reach the api/i);
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("shows a connection error on 5xx and does not navigate", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(503, { detail: "unavailable" })
    );

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-server-down");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/could not reach the api/i);
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  });

  it("shows an inline error when the key is empty", async () => {
    const fetchMock = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
    const user = userEvent.setup();
    render(<LoginPage />);
    // Bypass the form's `required` attribute by typing whitespace, then submit.
    const input = screen.getByLabelText(/api key/i);
    await user.type(input, "   ");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(screen.getByRole("alert")).toHaveTextContent(/empty/i);
    expect(pushMock).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("disables submit while the probe is in flight", async () => {
    let resolveFetch: (value: Response) => void = () => undefined;
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        })
    );

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-slow");
    await user.click(screen.getByRole("button", { name: /continue|checking/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /checking/i })).toBeDisabled();
    });

    resolveFetch(jsonResponse(200, { jobs: [] }));

    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith("/dashboard");
    });
  });

  it("never persists the candidate key while the probe is in flight", async () => {
    let resolveFetch: (value: Response) => void = () => undefined;
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        })
    );

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-unverified");
    await user.click(screen.getByRole("button", { name: /continue|checking/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /checking/i })).toBeDisabled();
    });
    // Probe sent but unresolved: nothing may be persisted yet.
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();

    resolveFetch(jsonResponse(200, { jobs: [] }));
    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith("/dashboard");
    });
  });

  it("maps denied browser storage to the storage message, not the network message", async () => {
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      jsonResponse(200, { jobs: [] })
    );
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("Browser storage unavailable");
    });

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-valid-key");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/storage is unavailable/i);
    });
    expect(screen.getByRole("alert")).not.toHaveTextContent(/could not reach/i);
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("aborts a hung probe within the configured timeout", async () => {
    const settings = getDefaultSettings();
    settings.api.timeoutMs = 1000;
    saveSettings(settings);
    (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true }
          );
        })
    );

    const user = userEvent.setup();
    render(<LoginPage />);
    await user.type(screen.getByLabelText(/api key/i), "ocr-hung");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    await screen.findByText(/could not reach the api/i, undefined, {
      timeout: 5000,
    });
    expect(pushMock).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
  }, 15000);
});
