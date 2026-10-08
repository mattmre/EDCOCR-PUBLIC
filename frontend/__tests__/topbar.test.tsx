import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Topbar, roleBadgeLabel } from "@/components/topbar";
import { AUTH_STATE_EVENT, AUTH_STORAGE_KEY, setApiKey } from "@/lib/auth";

const pushMock = vi.fn();
const originalHttpOnlySession = process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
  }),
}));

describe("roleBadgeLabel", () => {
  it("returns 'signed out' when no API key is present", () => {
    expect(roleBadgeLabel(false)).toBe("signed out");
  });

  it("returns 'operator' when an API key is present", () => {
    expect(roleBadgeLabel(true)).toBe("operator");
  });

  it("never returns 'read-only'", () => {
    expect(roleBadgeLabel(false)).not.toBe("read-only");
    expect(roleBadgeLabel(true)).not.toBe("read-only");
  });
});

describe("<Topbar /> role badge", () => {
  beforeEach(() => {
    pushMock.mockClear();
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "false";
    window.sessionStorage.clear();
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    if (originalHttpOnlySession === undefined) {
      delete process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION;
    } else {
      process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = originalHttpOnlySession;
    }
  });

  it("exposes data-testid=topbar-role-badge", async () => {
    render(<Topbar />);
    expect(await screen.findByTestId("topbar-role-badge")).toBeInTheDocument();
  });

  it("shows 'signed out' when no API key is cached", async () => {
    render(<Topbar />);
    const badge = await screen.findByTestId("topbar-role-badge");
    await waitFor(() => {
      expect(badge).toHaveTextContent("signed out");
    });
    expect(badge).not.toHaveTextContent("read-only");
    expect(screen.getByText("No API key")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /clear prior session/i })
    ).toBeInTheDocument();
  });

  it("shows 'operator' when an API key is cached", async () => {
    setApiKey("test-operator-key");
    render(<Topbar />);
    const badge = await screen.findByTestId("topbar-role-badge");
    await waitFor(() => {
      expect(badge).toHaveTextContent("operator");
    });
    expect(badge).not.toHaveTextContent("read-only");
    expect(screen.getByText("API key cached")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign out/i })).toBeInTheDocument();
  });

  it("does not render the dishonest 'read-only' label in either state", async () => {
    const { unmount } = render(<Topbar />);
    await screen.findByTestId("topbar-role-badge");
    expect(screen.queryByText("read-only")).toBeNull();
    unmount();

    setApiKey("another-key");
    render(<Topbar />);
    await waitFor(() => {
      expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("operator");
    });
    expect(screen.queryByText("read-only")).toBeNull();
  });

  it("updates the badge to 'signed out' after sign out", async () => {
    setApiKey("session-key");
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ cleared: true }), { status: 200 }));
    const user = userEvent.setup();
    render(<Topbar />);

    await waitFor(() => {
      expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("operator");
    });

    await user.click(screen.getByRole("button", { name: /sign out/i }));

    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBeNull();
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/auth/session"),
      expect.objectContaining({ method: "DELETE", credentials: "include" })
    );
    await waitFor(() => {
      expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("signed out");
      expect(pushMock).toHaveBeenCalledWith("/login");
    });
  });

  it("clears a prior HttpOnly cookie from API-key mode without a readable key", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ cleared: true }), { status: 200 }));
    const user = userEvent.setup();
    render(<Topbar />);

    await screen.findByRole("button", { name: /clear prior session/i });
    await user.click(screen.getByRole("button", { name: /clear prior session/i }));

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/auth/session"),
      expect.objectContaining({ method: "DELETE", credentials: "include" })
    );
    await waitFor(() => {
      expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("signed out");
      expect(pushMock).toHaveBeenCalledWith("/login");
    });
  });

  it("keeps prior-cookie cleanup retryable when off-mode deletion fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("unavailable", { status: 503 })
    );
    const user = userEvent.setup();
    render(<Topbar />);

    await user.click(await screen.findByRole("button", { name: /clear prior session/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/sign out failed/i);
    expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("session unknown");
    expect(
      screen.getByRole("button", { name: /clear prior session/i })
    ).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("probes an HttpOnly cookie and clears it through the server on sign out", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ jobs: [] }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ cleared: true }), { status: 200 }));
    const user = userEvent.setup();
    render(<Topbar />);

    await screen.findByRole("button", { name: /sign out/i });
    expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("operator");
    expect(screen.getByText("HttpOnly session")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /sign out/i }));

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("/api/v1/auth/session/probe"),
      expect.objectContaining({ method: "GET", credentials: "include" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining("/api/v1/auth/session"),
      expect.objectContaining({ method: "DELETE", credentials: "include" })
    );
    expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("signed out");
    expect(pushMock).toHaveBeenCalledWith("/login");
  });

  it("does not let a stale cookie probe overwrite a newer auth event", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    let finishProbe!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch").mockReturnValue(
      new Promise<Response>((resolve) => {
        finishProbe = resolve;
      })
    );
    render(<Topbar />);
    await screen.findByText("Checking authentication");

    act(() => {
      window.dispatchEvent(
        new CustomEvent(AUTH_STATE_EVENT, { detail: { authenticated: true } })
      );
      finishProbe(new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 }));
    });

    await waitFor(() => {
      expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("operator");
    });
  });

  it("does not claim signed out when server logout fails", async () => {
    process.env.NEXT_PUBLIC_OCR_HTTPONLY_SESSION = "true";
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ jobs: [] }), { status: 200 }))
      .mockResolvedValueOnce(new Response("unavailable", { status: 503 }));
    const user = userEvent.setup();
    render(<Topbar />);
    await screen.findByRole("button", { name: /sign out/i });

    await user.click(screen.getByRole("button", { name: /sign out/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/sign out failed/i);
    expect(screen.getByTestId("topbar-role-badge")).toHaveTextContent("session unknown");
    expect(pushMock).not.toHaveBeenCalled();
  });
});
