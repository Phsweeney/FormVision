import Link from "next/link";

import { buttonVariants } from "@/components/ui/button";
import { UploadPanel } from "@/components/upload/upload-panel";

/**
 * Landing page.
 *
 * A Server Component. Two modes share this page: offline analysis of an
 * uploaded video (the interactive `UploadPanel`, the only JavaScript shipped
 * here) and real-time coaching from a live webcam (a link to `/live`). Only the
 * upload panel needs the client; the live page loads its own bundle on demand.
 */
export default function HomePage() {
  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-14 sm:py-20">
      <section className="space-y-4 text-center">
        <span className="border-border/70 bg-card/60 text-muted-foreground inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs">
          <span className="bg-primary/80 size-1.5 rounded-full" aria-hidden />
          Back squat
        </span>

        <h1 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
          See what your squat actually looks like
        </h1>

        <p className="text-muted-foreground mx-auto max-w-xl text-sm text-pretty sm:text-base">
          FormVision measures your depth, counts your reps, tracks your joint
          angles, and tells you what to work on — using explicit biomechanical
          rules, not a black box. Analyse a recorded set, or get coached live
          from your webcam.
        </p>
      </section>

      {/* Mode chooser: the two ways in, given equal weight. */}
      <section className="mt-10 grid gap-4 sm:grid-cols-2">
        <div className="border-border/60 bg-card/40 flex flex-col rounded-xl border p-5">
          <span className="border-primary/30 bg-primary/10 text-primary inline-flex w-fit items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium">
            <span className="bg-primary size-1.5 rounded-full" aria-hidden />
            Live
          </span>
          <h2 className="mt-3 text-base font-semibold">Live webcam coaching</h2>
          <p className="text-muted-foreground mt-1.5 flex-1 text-xs leading-relaxed">
            Turn on your camera and get real-time rep counting, depth and tempo,
            and spoken coaching as you lift. Nothing is uploaded — it all runs in
            your browser.
          </p>
          <Link
            href="/live"
            className={buttonVariants({ size: "lg", className: "mt-4 w-full" })}
          >
            Start a live session
          </Link>
        </div>

        <div className="border-border/60 bg-card/40 flex flex-col rounded-xl border p-5">
          <span className="border-border/70 bg-muted/40 text-muted-foreground inline-flex w-fit items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium">
            Upload
          </span>
          <h2 className="mt-3 text-base font-semibold">Analyze a video</h2>
          <p className="text-muted-foreground mt-1.5 flex-1 text-xs leading-relaxed">
            Have a set already filmed? Upload an MP4 or MOV and get a full report:
            skeleton overlay, per-rep breakdown, joint-angle charts, and coaching.
          </p>
          <Link
            href="#upload"
            className={buttonVariants({
              variant: "outline",
              size: "lg",
              className: "mt-4 w-full",
            })}
          >
            Upload a video
          </Link>
        </div>
      </section>

      <section id="upload" className="mt-8 scroll-mt-20">
        <UploadPanel />
      </section>

      <section className="mt-8">
        <div className="border-border/60 bg-card/30 rounded-xl border p-5">
          <h2 className="text-sm font-medium">Filming tips</h2>
          <ul className="text-muted-foreground mt-3 grid gap-2 text-xs sm:grid-cols-2">
            <li>· Keep your whole body in frame for the entire set.</li>
            <li>· Film side-on for the most accurate depth and lean.</li>
            <li>· Film front-on to check left-right balance.</li>
            <li>· One person in shot, in even lighting.</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
