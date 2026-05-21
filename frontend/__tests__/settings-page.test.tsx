import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import SettingsPage from "@/app/settings/page";
import { setApiKey } from "@/lib/auth";
import { SETTINGS_STORAGE_KEY } from "@/lib/settings-store";

vi.mock("next/navigation", => ({
  useRouter: => ({ push: vi.fn() }),
  usePathname: => "/settings",
}));

describe("<SettingsPage />", => {
  beforeEach(() => {
    window.localStorage.clear();
    setApiKey("test-key-1234abcdEFGH");
    document.documentElement.classList.remove("theme-light", "theme-dark");
    delete document.documentElement.dataset.theme;
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the General tab by default", async => {
    render(<SettingsPage />);
    await screen.findByTestId("settings-form-general");
    expect(screen.getByTestId("settings-tab-general")).toHaveAttribute(
      "aria-selected",
      "true"
    );
  });

  it("switches to the Display tab when clicked", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-display"));
    expect(await screen.findByTestId("settings-form-display")).toBeInTheDocument();
    expect(screen.queryByTestId("settings-form-general")).toBeNull();
  });

  it("toggles the theme to dark and applies it to documentElement", async => {
    render(<SettingsPage />);
    fireEvent.click(await screen.findByTestId("settings-theme-dark"));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("rejects an invalid API base URL on save", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-api"));
    const input = await screen.findByTestId("settings-base-url");
    fireEvent.change(input, { target: { value: "not a url" } });
    fireEvent.submit(screen.getByTestId("settings-form-api"));
    expect(await screen.findByTestId("settings-base-url-error")).toBeInTheDocument();
  });

  it("accepts a valid API base URL on save", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-api"));
    const input = await screen.findByTestId("settings-base-url");
    fireEvent.change(input, { target: { value: "https://ocr.example.com" } });
    fireEvent.submit(screen.getByTestId("settings-form-api"));
    await waitFor(() => {
      expect(screen.queryByTestId("settings-base-url-error")).toBeNull();
    });
    const stored = JSON.parse(
      window.localStorage.getItem(SETTINGS_STORAGE_KEY) as string
    );
    expect(stored.api.baseUrl).toBe("https://ocr.example.com");
  });

  it("rejects out-of-range timeoutMs on save", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-api"));
    const input = await screen.findByTestId("settings-timeout-ms");
    fireEvent.change(input, { target: { value: "10" } });
    fireEvent.submit(screen.getByTestId("settings-form-api"));
    expect(await screen.findByTestId("settings-timeout-error")).toBeInTheDocument();
  });

  it("auto-refresh slider drives the displayed seconds value", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-display"));
    const slider = await screen.findByTestId("settings-auto-refresh");
    fireEvent.change(slider, { target: { value: "120" } });
    expect(screen.getByLabelText(/Auto-refresh interval/)).toBeInTheDocument();
    expect(screen.getByText(/120/)).toBeInTheDocument();
  });

  it("saves a new pageSize and shows the 'Last saved' label", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-display"));
    const select = await screen.findByTestId("settings-page-size");
    fireEvent.change(select, { target: { value: "100" } });
    fireEvent.submit(screen.getByTestId("settings-form-display"));
    expect(await screen.findByTestId("settings-last-saved")).toBeInTheDocument();
    const stored = JSON.parse(
      window.localStorage.getItem(SETTINGS_STORAGE_KEY) as string
    );
    expect(stored.display.pageSize).toBe(100);
  });

  it("Reset to defaults requires two clicks", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-display"));
    const select = await screen.findByTestId("settings-page-size");
    fireEvent.change(select, { target: { value: "100" } });
    fireEvent.submit(screen.getByTestId("settings-form-display"));
    await screen.findByTestId("settings-last-saved");

    const resetBtn = screen.getByTestId("settings-reset-button");
    fireEvent.click(resetBtn);
    expect(resetBtn).toHaveTextContent(/Confirm reset/);
    fireEvent.click(resetBtn);
    await waitFor(() => {
      expect(window.localStorage.getItem(SETTINGS_STORAGE_KEY)).toBeNull();
    });
    expect(screen.getByTestId("settings-reset-button")).toHaveTextContent(/Reset to defaults/);
  });

  it("Cancel button aborts a pending reset", async => {
    render(<SettingsPage />);
    const resetBtn = screen.getByTestId("settings-reset-button");
    fireEvent.click(resetBtn);
    expect(resetBtn).toHaveTextContent(/Confirm reset/);
    fireEvent.click(screen.getByTestId("settings-reset-cancel"));
    expect(screen.getByTestId("settings-reset-button")).toHaveTextContent(/Reset to defaults/);
  });

  it("surfaces a save error when localStorage write throws", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-display"));
    await screen.findByTestId("settings-form-display");
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("QuotaExceededError");
    });
    fireEvent.submit(screen.getByTestId("settings-form-display"));
    expect(await screen.findByTestId("settings-save-error")).toBeInTheDocument();
    spy.mockRestore();
  });

  it("syncs the redacted API key preview from the auth layer", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-api"));
    const sync = await screen.findByTestId("settings-api-key-sync");
    fireEvent.click(sync);
    const preview = screen.getByTestId("settings-api-key-preview");
    expect(preview.textContent).toContain("test");
    expect(preview.textContent).toContain("EFGH");
  });

  it("disables sub-toggles when desktop notifications are off", async => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-notifications"));
    const sub = await screen.findByTestId("settings-notif-sound");
    // Default state: desktopEnabled is false, so sub-toggles must be disabled.
    expect(sub).toBeDisabled();
  });

  it("enables sub-toggles after desktop notifications are turned on", async => {
    const NotificationStub = vi.fn() as unknown as typeof Notification;
    Object.defineProperty(NotificationStub, "permission", {
      value: "granted",
      configurable: true,
    });
    Object.defineProperty(NotificationStub, "requestPermission", {
      value: vi.fn().mockResolvedValue("granted"),
      configurable: true,
    });
    (globalThis as unknown as { Notification: typeof Notification }).Notification = NotificationStub;

    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-notifications"));
    const desktop = await screen.findByTestId("settings-notif-desktop");
    fireEvent.click(desktop);
    await waitFor(() => {
      expect(screen.getByTestId("settings-notif-sound")).not.toBeDisabled();
    });
  });

  it("does not show the request-permission button after grant", async => {
    const NotificationStub = vi.fn() as unknown as typeof Notification;
    Object.defineProperty(NotificationStub, "permission", {
      value: "granted",
      configurable: true,
    });
    Object.defineProperty(NotificationStub, "requestPermission", {
      value: vi.fn().mockResolvedValue("granted"),
      configurable: true,
    });
    (globalThis as unknown as { Notification: typeof Notification }).Notification = NotificationStub;

    render(<SettingsPage />);
    fireEvent.click(screen.getByTestId("settings-tab-notifications"));
    await screen.findByTestId("settings-notif-desktop");
    expect(screen.queryByTestId("settings-notif-request-permission")).toBeNull();
  });
});
