"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { clearApiKey, getApiKey } from "@/lib/auth";

export function Topbar() {
  const router = useRouter();
  const [keyPresent, setKeyPresent] = useState(false);

  useEffect(() => {
    setKeyPresent(getApiKey() !== null);
  }, []);

  function handleSignOut() {
    clearApiKey();
    router.push("/login");
  }

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-background px-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold">EDCOCR Console</span>
        <span className="rounded bg-muted px-2 py-0.5 text-xs text-muted-foreground">
          read-only
        </span>
      </div>
      <div className="flex items-center gap-3 text-sm">
        <span className="text-muted-foreground">
          {keyPresent ? "API key cached" : "No API key"}
        </span>
        {keyPresent ? (
          <Button variant="outline" size="sm" onClick={handleSignOut}>
            Sign out
          </Button>
        ) : (
          <Button variant="outline" size="sm" onClick={() => router.push("/login")}>
            Sign in
          </Button>
        )}
      </div>
    </header>
  );
}
