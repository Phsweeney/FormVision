import Link from "next/link";

/**
 * Application header.
 *
 * A Server Component — it holds no state and needs no interactivity, so there
 * is no reason to ship it to the browser as JavaScript.
 */
export function SiteHeader() {
  return (
    <header className="border-border/60 bg-background/80 sticky top-0 z-40 border-b backdrop-blur-sm">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center justify-between px-6">
        <Link
          href="/"
          className="group flex items-center gap-2.5"
          aria-label="FormVision home"
        >
          <span
            aria-hidden
            className="from-primary/90 to-primary/50 ring-primary/20 flex size-7 items-center justify-center rounded-md bg-gradient-to-br ring-1"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              className="text-primary-foreground size-4"
            >
              <circle cx="12" cy="4.5" r="2" />
              <path d="M12 7v5m0 0-3 5m3-5 3 5M8 9.5h8" />
            </svg>
          </span>
          <span className="text-[15px] font-semibold tracking-tight">
            Form<span className="text-muted-foreground font-normal">Vision</span>
          </span>
        </Link>

        <span className="text-muted-foreground hidden text-xs sm:inline">
          Back squat analysis
        </span>
      </div>
    </header>
  );
}
