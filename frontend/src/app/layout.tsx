import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";
import { Topbar } from "@/components/topbar";
import { publicRuntimeConfig, RUNTIME_CONFIG_ID, serializeRuntimeConfig } from "@/lib/runtime-config";

// Read deployment settings when serving HTML, not while building the image.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "OCR_LOCAL Console",
  description: "Operator console for the OCR_LOCAL pipeline",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <script
          id={RUNTIME_CONFIG_ID}
          type="application/json"
          dangerouslySetInnerHTML={{ __html: serializeRuntimeConfig(publicRuntimeConfig(process.env)) }}
        />
      </head>
      <body className="min-h-screen bg-background text-foreground">
        <div className="flex min-h-screen">
          <Sidebar />
          <div className="flex min-h-screen flex-1 flex-col">
            <Topbar />
            <main className="flex-1 p-6">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
