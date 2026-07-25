"use client";

import { use } from "react";
import Link from "next/link";

import { AnalysisCharts } from "@/components/dashboard/analysis-charts";
import {
  ErrorState,
  LoadingState,
  ProcessingState,
  TrackingWarning,
} from "@/components/dashboard/analysis-status";
import { FeedbackList } from "@/components/dashboard/feedback-list";
import { MetricCards } from "@/components/dashboard/metric-cards";
import { RepTable } from "@/components/dashboard/rep-table";
import { VideoPanel } from "@/components/dashboard/video-panel";
import { buttonVariants } from "@/components/ui/button";
import { useAnalysis } from "@/hooks/use-analysis";
import { formatTimestamp } from "@/lib/format";
import { isPending } from "@/lib/types";

/**
 * The analysis dashboard.
 *
 * A Client Component, because it polls for status and owns the video player.
 * Route params are a Promise in Next 16 and a Client Component cannot be
 * `async`, so they are unwrapped with React's `use()` — the pattern the bundled
 * Next.js docs prescribe for exactly this case.
 *
 * Section order follows the spec: video, then overlay, then metrics, then
 * graphs, then coaching feedback.
 */
export default function AnalysisPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { analysis, error, isLoading, refresh } = useAnalysis(id);

  if (isLoading && !analysis) return <LoadingState />;

  if (error && !analysis) {
    return (
      <ErrorState
        title="Could not load this analysis"
        message={error}
        onRetry={refresh}
      />
    );
  }

  if (!analysis) {
    return (
      <ErrorState
        title="Analysis not found"
        message="This analysis does not exist. It may have been removed."
      />
    );
  }

  if (analysis.status === "failed") {
    return (
      <ErrorState
        title="Analysis failed"
        message={
          analysis.error_message
          ?? "Something went wrong while analysing this video."
        }
      />
    );
  }

  if (isPending(analysis)) {
    return <ProcessingState filename={analysis.filename} />;
  }

  const { metrics, reps, feedback, series } = analysis;

  // `completed` without metrics should not happen, but rendering a broken
  // dashboard would be worse than saying so plainly.
  if (!metrics) {
    return (
      <ErrorState
        title="No results available"
        message="This analysis completed but produced no measurements."
        onRetry={refresh}
      />
    );
  }

  const showTrackingWarning = metrics.tracking_quality < 0.6;

  return (
    <div className="mx-auto w-full max-w-6xl space-y-10 px-6 py-10">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold tracking-tight">
            {analysis.filename}
          </h1>
          <p className="text-muted-foreground mt-1 text-xs">
            Analysed {formatTimestamp(analysis.updated_at)}
            {analysis.processing_seconds !== null
              && ` · took ${analysis.processing_seconds.toFixed(1)}s`}
          </p>
        </div>

        {/* `buttonVariants` on the link rather than a slotted Button: this
            shadcn build is on Base UI, which has no `asChild`, and styling the
            anchor keeps it a real link with correct keyboard and context-menu
            behaviour. */}
        <Link
          href="/"
          className={buttonVariants({ variant: "outline", size: "sm" })}
        >
          Analyse another video
        </Link>
      </header>

      {showTrackingWarning && (
        <TrackingWarning quality={metrics.tracking_quality} />
      )}

      <section aria-labelledby="video-heading">
        <h2 id="video-heading" className="sr-only">
          Video and skeleton overlay
        </h2>
        <VideoPanel
          videoUrl={analysis.video_url}
          overlayUrl={analysis.overlay_url}
          series={series}
        />
      </section>

      <section aria-labelledby="metrics-heading">
        <h2
          id="metrics-heading"
          className="text-muted-foreground mb-3 text-xs font-medium tracking-wide uppercase"
        >
          Summary
        </h2>
        <MetricCards metrics={metrics} />
      </section>

      {series && reps && (
        <section aria-labelledby="charts-heading">
          <h2
            id="charts-heading"
            className="text-muted-foreground mb-3 text-xs font-medium tracking-wide uppercase"
          >
            Movement over time
          </h2>
          <AnalysisCharts series={series} reps={reps} />
        </section>
      )}

      {reps && reps.length > 0 && (
        <section aria-labelledby="reps-heading">
          <h2
            id="reps-heading"
            className="text-muted-foreground mb-3 text-xs font-medium tracking-wide uppercase"
          >
            Repetition breakdown
          </h2>
          <RepTable reps={reps} />
        </section>
      )}

      {feedback && feedback.length > 0 && (
        <section aria-labelledby="feedback-heading">
          <h2
            id="feedback-heading"
            className="text-muted-foreground mb-3 text-xs font-medium tracking-wide uppercase"
          >
            Coaching feedback
          </h2>
          <FeedbackList items={feedback} />
        </section>
      )}
    </div>
  );
}
