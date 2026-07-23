"use client";

import Link from "next/link";

import { buttonVariants } from "@/components/ui/button";

/**
 * Live coaching page.
 *
 * A Client Component: it drives the webcam, runs pose estimation and the
 * analysis engine in the browser, and renders a skeleton overlay on a canvas.
 * None of that touches the server beyond fetching the shared analysis
 * thresholds once (`GET /config`).
 *
 * This is the shell. The camera stage, live metrics, coaching, and session
 * summary are layered in over the following milestones; for now it establishes
 * the route and the layout so the two-mode home page has somewhere to go.
 */
export default function LivePage() {
  return (
    <div className="mx-auto w-full max-w-6xl px-6 py-10">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <span className="border-primary/30 bg-primary/10 text-primary inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium">
            <span className="bg-primary size-1.5 rounded-full" aria-hidden />
            Live
          </span>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight">
            Live coaching
          </h1>
          <p className="text-muted-foreground mt-1 text-sm">
            Real-time rep counting, depth, tempo, and spoken feedback from your
            webcam. Everything runs in your browser.
          </p>
        </div>
        <Link
          href="/"
          className={buttonVariants({ variant: "outline", size: "sm" })}
        >
          Back to home
        </Link>
      </header>

      <div className="border-border/60 bg-card/40 mt-6 grid aspect-video w-full place-items-center rounded-xl border">
        <p className="text-muted-foreground text-sm">
          Camera stage — coming online in the next step.
        </p>
      </div>
    </div>
  );
}
