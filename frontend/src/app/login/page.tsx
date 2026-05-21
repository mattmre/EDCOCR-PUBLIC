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
import { setApiKey } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    try {
      setApiKey(value);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save API key");
      return;
    }
    router.push("/dashboard");
  }

  return (
    <div className="mx-auto flex min-h-[60vh] max-w-md items-center">
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Sign in</CardTitle>
          <CardDescription>
            Paste an EDCOCR operator API key. The key is stored only in this browser&apos;s
            localStorage and is sent as the X-API-Key header on every request.
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
            <Button type="submit">Continue</Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}
