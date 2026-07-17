import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { SiteHeader } from "@/components/layout/site-header";

import "./globals.css";

// These variable names must match what globals.css reads in its `@theme inline`
// block (`--font-sans` / `--font-mono`). The scaffold emitted them as
// `--font-geist-sans`, so the theme was silently falling back to system fonts.
const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FormVision — Squat Analysis",
  description:
    "Upload a squat video for pose tracking, rep counting, joint angle "
    + "measurement, and rule-based coaching feedback.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      // The app commits to a dark theme: it is a video-review tool, and a dark
      // surround keeps attention on the footage rather than competing with it.
      className={`dark ${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="bg-background text-foreground flex min-h-full flex-col">
        <SiteHeader />
        <main className="flex-1">{children}</main>
        <footer className="border-border/60 text-muted-foreground border-t px-6 py-6 text-center text-xs">
          FormVision V1 · Back squat analysis · Rule-based, no machine learning
        </footer>
      </body>
    </html>
  );
}
