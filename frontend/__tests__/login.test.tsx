import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import LoginPage from "@/app/login/page";
import { AUTH_STORAGE_KEY } from "@/lib/auth";

const pushMock = vi.fn();

vi.mock("next/navigation", => ({
  useRouter: => ({
    push: pushMock,
  }),
}));

describe("LoginPage", => {
  it("renders the API key input and submit button", => {
    render(<LoginPage />);
    expect(screen.getByLabelText(/api key/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /continue/i })).toBeInTheDocument();
  });

  it("stores the entered key and routes to /dashboard on submit", async => {
    pushMock.mockClear();
    const user = userEvent.setup();
    render(<LoginPage />);
    const input = screen.getByLabelText(/api key/i);
    await user.type(input, "ocr-login-test");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(window.localStorage.getItem(AUTH_STORAGE_KEY)).toBe("ocr-login-test");
    expect(pushMock).toHaveBeenCalledWith("/dashboard");
  });

  it("shows an inline error when the key is empty", async => {
    pushMock.mockClear();
    const user = userEvent.setup();
    render(<LoginPage />);
    // Bypass the form's `required` attribute by typing whitespace, then submit.
    const input = screen.getByLabelText(/api key/i);
    await user.type(input, "   ");
    await user.click(screen.getByRole("button", { name: /continue/i }));

    expect(screen.getByRole("alert")).toHaveTextContent(/empty/i);
    expect(pushMock).not.toHaveBeenCalled();
  });
});
