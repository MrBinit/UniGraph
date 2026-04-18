import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { DislikeIcon, LikeIcon, SparklesIcon } from "./Icons";
import type { ChatMessage, ReactionType } from "../types";

interface MessageCardProps {
  message: ChatMessage;
  onRegenerate: (sourcePrompt: string) => void;
  onRegenerateInDeep: (sourcePrompt: string) => void;
  onRetryWebOnly: (sourcePrompt: string) => void;
  onReaction: (messageId: string, reaction: ReactionType) => void;
}

function avatarForRole(role: ChatMessage["role"]): string {
  return role === "assistant" ? "AI" : "ME";
}

function cardClass(role: ChatMessage["role"]): string {
  if (role === "assistant") {
    return "border border-blue-100 bg-white shadow-soft dark:border-slate-700 dark:bg-slate-900";
  }
  return "border border-blue-100 bg-gradient-to-br from-blue-50/80 via-white to-rose-50/50 shadow-sm dark:border-slate-700 dark:bg-slate-800/80";
}

function extractUrlsFromText(input: string): string[] {
  const matches = input.match(/https?:\/\/[^\s)"']+/gi) ?? [];
  const urls = matches
    .map((item) => normalizeUrlCandidate(item))
    .filter((item): item is string => Boolean(item));
  return uniqueUrls(urls);
}

function normalizeUrlCandidate(raw: string): string | null {
  const trimmed = String(raw ?? "").trim().replace(/[),.;\]]+$/g, "");
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = new URL(trimmed);
    const normalizedPath = parsed.pathname === "/" ? "" : parsed.pathname.replace(/\/+$/, "");
    return `${parsed.protocol}//${parsed.host}${normalizedPath}${parsed.search}`;
  } catch {
    return null;
  }
}

function uniqueUrls(values: string[]): string[] {
  const deduped: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    const normalized = normalizeUrlCandidate(value);
    if (!normalized) {
      continue;
    }
    const key = normalized.toLowerCase();
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(normalized);
  }
  return deduped;
}

function displayWebsite(value: string): string {
  try {
    const hostname = new URL(value).hostname.trim().toLowerCase();
    return hostname.replace(/^www\./, "");
  } catch {
    return value.trim().toLowerCase();
  }
}

function compactUrlPath(url: string): string {
  try {
    const parsed = new URL(url);
    const path = `${parsed.pathname}${parsed.search}`.trim();
    return path && path !== "/" ? path : "/";
  } catch {
    return "/";
  }
}

function splitAnswerSections(content: string): { primary: string; sourcesBlock: string } {
  const text = String(content ?? "");
  const marker = text.match(/(?:^|\n)\s*(?:#{1,3}\s*)?Sources?:?\s*(?:\n|$)/i);
  if (!marker || typeof marker.index !== "number") {
    return { primary: text.trim(), sourcesBlock: "" };
  }
  const primary = text.slice(0, marker.index).trim();
  const sourcesBlock = text.slice(marker.index).trim();
  return { primary, sourcesBlock };
}

function sourceUrlsFromBlock(sourcesBlock: string): string[] {
  return extractUrlsFromText(sourcesBlock || "");
}

function compactPercent(value?: number): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return `${pct}%`;
}

function compactThoughtStep(step: string, maxChars = 140): string {
  const normalized = String(step).replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxChars - 1))}…`;
}

function isPlannerThoughtStep(step: string): boolean {
  const normalized = String(step).trim().toLowerCase();
  return (
    normalized.startsWith("planner:") ||
    normalized.startsWith("planner queries:") ||
    normalized.startsWith("planner checks:") ||
    normalized.startsWith("searching:") ||
    normalized.startsWith("coverage gaps:") ||
    normalized.startsWith("follow-up queries:") ||
    normalized.startsWith("domain coverage:")
  );
}

function partitionThoughtSteps(steps: string[]): { planner: string[]; progress: string[] } {
  const planner: string[] = [];
  const progress: string[] = [];
  for (const step of steps) {
    if (isPlannerThoughtStep(step)) {
      planner.push(step);
      continue;
    }
    progress.push(step);
  }
  return {
    planner: planner.slice(-5),
    progress: progress.slice(-6),
  };
}

export function MessageCard({
  message,
  onRegenerate,
  onRegenerateInDeep,
  onRetryWebOnly,
  onReaction,
}: MessageCardProps) {
  const [copied, setCopied] = useState(false);
  const [showCitations, setShowCitations] = useState(false);
  const [showAllCitations, setShowAllCitations] = useState(false);
  const [showThought, setShowThought] = useState(false);
  const copyResetTimerRef = useRef<number | null>(null);
  const isErrorMessage =
    message.role === "assistant" && message.content.trim().toLowerCase().startsWith("error:");
  const modeLabel = message.executionMode ? message.executionMode.toUpperCase() : "";
  const { primary: primaryAnswer, sourcesBlock } = splitAnswerSections(message.content);
  const citations = useMemo(() => {
    return uniqueUrls([
      ...(message.sourceUrls ?? []),
      ...sourceUrlsFromBlock(sourcesBlock),
      ...extractUrlsFromText(message.content),
    ]);
  }, [message.content, message.sourceUrls, sourcesBlock]);
  const citationDomains = useMemo(() => {
    return Array.from(new Set(citations.map((url) => displayWebsite(url)))).slice(0, 12);
  }, [citations]);
  const citationPreview = showAllCitations ? citations : citations.slice(0, 8);
  const thoughtSteps = (message.reasoningSteps ?? [])
    .map((step) => compactThoughtStep(step))
    .filter((step) => Boolean(step))
    .slice(-18);
  const thoughtSites = (message.searchedWebsites ?? []).slice(0, 10);
  const thoughtGroups = useMemo(() => partitionThoughtSteps(thoughtSteps), [thoughtSteps]);
  const isStreamingThought = message.content.includes("Still working") || message.content.endsWith("▍");
  const hasThoughtContent =
    thoughtGroups.planner.length > 0 ||
    thoughtGroups.progress.length > 0 ||
    thoughtSites.length > 0;

  useEffect(() => {
    return () => {
      if (copyResetTimerRef.current !== null) {
        window.clearTimeout(copyResetTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (isStreamingThought) {
      setShowThought(true);
    }
  }, [isStreamingThought]);

  useEffect(() => {
    if (!showCitations) {
      setShowAllCitations(false);
    }
  }, [showCitations]);

  const handleCopy = async () => {
    const text = message.content.trim();
    if (!text) {
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (copyResetTimerRef.current !== null) {
        window.clearTimeout(copyResetTimerRef.current);
      }
      copyResetTimerRef.current = window.setTimeout(() => {
        setCopied(false);
      }, 1400);
    } catch {
      // Clipboard permission can fail in some browsers; ignore silently.
    }
  };

  return (
    <article className="animate-[fade-up_240ms_ease-out] space-y-1.5">
      <div className="flex items-start gap-3">
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-brand-blue to-brand-red text-[11px] font-bold text-white">
          {avatarForRole(message.role)}
        </div>

        <div className={`w-full rounded-2xl p-3.5 ${cardClass(message.role)}`}>
          <div className="mb-1.5 flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{message.username}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </p>
            </div>
            {message.role === "assistant" ? (
              <div className="flex items-center gap-1.5">
                {modeLabel ? (
                  <span className="rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-medium text-rose-700 dark:bg-rose-950/50 dark:text-rose-200">
                    {modeLabel}
                  </span>
                ) : null}
                <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-medium text-brand-blue dark:bg-slate-700 dark:text-slate-200">
                  AI response
                </span>
              </div>
            ) : null}
          </div>

          {message.role === "assistant" ? (
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs leading-4 text-slate-500 dark:text-slate-400">
              <div className="w-full">
                <button
                  type="button"
                  onClick={() => setShowThought((prev) => !prev)}
                  className="inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-3 py-1.5 text-[13px] font-medium text-slate-700 transition hover:bg-blue-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  <span
                    className={`h-2 w-2 rounded-full ${
                      isStreamingThought ? "animate-pulse bg-emerald-500" : "bg-slate-400 dark:bg-slate-500"
                    }`}
                  />
                  {message.workedForLabel ? `Thought for ${message.workedForLabel}` : "Thought"}
                  <span className="text-[11px]">{showThought ? "▲" : "▼"}</span>
                </button>
                {showThought ? (
                  <div className="mt-2 rounded-xl border border-slate-200 bg-slate-50 p-3 text-[12px] text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200">
                    {hasThoughtContent ? (
                      <div className="space-y-3">
                        {thoughtGroups.planner.length ? (
                          <div>
                            <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                              Planner
                            </p>
                            <ul className="space-y-1">
                              {thoughtGroups.planner.map((step) => (
                                <li key={step} className="break-words text-slate-700 dark:text-slate-200">
                                  {step}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {thoughtGroups.progress.length ? (
                          <div>
                            <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                              Progress
                            </p>
                            <ul className="space-y-1">
                              {thoughtGroups.progress.map((step) => (
                                <li key={step} className="break-words text-slate-700 dark:text-slate-200">
                                  {step}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {thoughtSites.length ? (
                          <div>
                            <p className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                              Sources Checked
                            </p>
                            <div className="flex flex-wrap gap-1.5">
                              {thoughtSites.map((site) => (
                                <code
                                  key={site}
                                  className="rounded-md border border-blue-100 bg-white px-1.5 py-0.5 text-[11px] text-brand-blue dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200"
                                >
                                  {site}
                                </code>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <p className="text-slate-500 dark:text-slate-400">
                        {isStreamingThought ? "Collecting planning steps..." : "No thought trace captured."}
                      </p>
                    )}
                  </div>
                ) : null}
              </div>
              {compactPercent(message.trustConfidence) ? (
                <span className="rounded-full border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:border-emerald-900/70 dark:bg-emerald-950/40 dark:text-emerald-200">
                  {compactPercent(message.trustConfidence)} confidence
                </span>
              ) : null}
              {message.trustFreshness ? (
                <span className="rounded-full border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium capitalize text-brand-blue dark:border-blue-900/70 dark:bg-blue-950/40 dark:text-blue-200">
                  {message.trustFreshness}
                </span>
              ) : null}
              {message.trustContradiction ? (
                <span className="rounded-full border border-amber-200 bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:border-amber-900/70 dark:bg-amber-950/40 dark:text-amber-200">
                  conflict
                </span>
              ) : null}
              {citations.length ? (
                <button
                  type="button"
                  onClick={() => setShowCitations((prev) => !prev)}
                  className="text-[13px] text-brand-blue hover:underline dark:text-blue-300"
                >
                  {showCitations ? "Hide" : "Show"} citations ({citations.length})
                </button>
              ) : null}
              {citationDomains.length ? (
                <span className="rounded-full border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-medium text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
                  {citationDomains.length} domains
                </span>
              ) : null}
            </div>
          ) : null}

          {message.role === "assistant" ? (
            <div className="text-[15px] leading-6 text-slate-700 dark:text-slate-200">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ children }) => (
                    <h1 className="mb-3 mt-1 text-2xl font-semibold text-slate-900 dark:text-slate-100">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="mb-2 mt-4 text-xl font-semibold text-slate-900 dark:text-slate-100">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="mb-2 mt-3 text-lg font-semibold text-slate-900 dark:text-slate-100">
                      {children}
                    </h3>
                  ),
                  p: ({ children }) => <p className="my-2 whitespace-pre-wrap">{children}</p>,
                  strong: ({ children }) => (
                    <strong className="font-semibold text-slate-900 dark:text-slate-100">{children}</strong>
                  ),
                  ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-6">{children}</ul>,
                  ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-6">{children}</ol>,
                  li: ({ children }) => <li>{children}</li>,
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noreferrer"
                      className="break-all text-brand-blue underline decoration-blue-300 underline-offset-2 hover:text-blue-700 dark:text-blue-300 dark:decoration-blue-500 dark:hover:text-blue-200"
                    >
                      {children}
                    </a>
                  ),
                  hr: () => <hr className="my-5 border-blue-100 dark:border-slate-700" />,
                  code: ({ children }) => (
                    <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[13px] text-slate-800 dark:bg-slate-800 dark:text-slate-100">
                      {children}
                    </code>
                  ),
                  pre: ({ children }) => (
                    <pre className="my-4 overflow-x-auto rounded-xl bg-slate-950 p-3 text-[13px] text-slate-100">
                      {children}
                    </pre>
                  ),
                  table: ({ children }) => (
                    <div className="my-4 overflow-x-auto">
                      <table className="min-w-full border-collapse text-left text-sm">{children}</table>
                    </div>
                  ),
                  thead: ({ children }) => (
                    <thead className="bg-blue-50/70 dark:bg-slate-800/80">{children}</thead>
                  ),
                  th: ({ children }) => (
                    <th className="border border-blue-100 px-3 py-2 font-semibold text-slate-900 dark:border-slate-700 dark:text-slate-100">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-blue-100 px-3 py-2 align-top dark:border-slate-700">
                      {children}
                    </td>
                  ),
                }}
              >
                {primaryAnswer || message.content}
              </ReactMarkdown>
            </div>
          ) : (
            <p className="whitespace-pre-wrap text-[15px] leading-6 text-slate-700 dark:text-slate-200">
              {message.content}
            </p>
          )}

          {message.role === "assistant" && showCitations && citations.length ? (
            <section className="mt-2 rounded-xl border border-blue-100 bg-white/70 p-2.5 dark:border-slate-700 dark:bg-slate-900/60">
              <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
                Supporting Evidence
              </p>
              <div className="space-y-2">
                {citationPreview.map((url, index) => (
                  <a
                    key={url}
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="block rounded-lg border border-blue-100 bg-white px-2.5 py-2 hover:bg-blue-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
                        [{index + 1}] {displayWebsite(url)}
                      </span>
                    </div>
                    <span className="mt-0.5 block break-all text-[11px] text-brand-blue dark:text-blue-300">
                      {compactUrlPath(url)}
                    </span>
                  </a>
                ))}
                {citations.length > citationPreview.length ? (
                  <button
                    type="button"
                    onClick={() => setShowAllCitations((prev) => !prev)}
                    className="text-[11px] font-medium text-brand-blue hover:underline dark:text-blue-300"
                  >
                    {showAllCitations
                      ? "Show fewer citations"
                      : `Show ${citations.length - citationPreview.length} more citations`}
                  </button>
                ) : null}
                {sourcesBlock && !sourceUrlsFromBlock(sourcesBlock).length ? (
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-2 text-[11px] text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{sourcesBlock}</ReactMarkdown>
                  </div>
                ) : null}
              </div>
            </section>
          ) : null}

          {message.role === "assistant" && message.sourcePrompt ? (
            <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-blue-100 pt-3 dark:border-slate-700">
              <button
                type="button"
                onClick={() => onRegenerate(message.sourcePrompt || "")}
                className="inline-flex items-center gap-1 rounded-lg border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-brand-blue transition hover:bg-blue-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
              >
                <SparklesIcon className="h-3.5 w-3.5" />
                Regenerate
              </button>

              <button
                type="button"
                onClick={() => onRegenerateInDeep(message.sourcePrompt || "")}
                className="rounded-lg border border-rose-200 bg-rose-50 px-2.5 py-1.5 text-xs text-rose-700 transition hover:bg-rose-100 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-200"
              >
                Regenerate in Deep
              </button>

              <button
                type="button"
                onClick={handleCopy}
                className={`rounded-lg border px-2.5 py-1.5 text-xs transition ${
                  copied
                    ? "border-emerald-500 bg-emerald-600 text-white"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-800 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200"
                }`}
              >
                {copied ? "Copied" : "Copy"}
              </button>

              <button
                type="button"
                onClick={() => onReaction(message.id, message.reaction === "like" ? null : "like")}
                className={`rounded-lg border px-2.5 py-1.5 text-xs transition ${
                  message.reaction === "like"
                    ? "border-brand-blue bg-brand-blue text-white"
                    : "border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:text-brand-blue dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200"
                }`}
              >
                <LikeIcon className="h-3.5 w-3.5" />
              </button>

              <button
                type="button"
                onClick={() => onReaction(message.id, message.reaction === "dislike" ? null : "dislike")}
                className={`rounded-lg border px-2.5 py-1.5 text-xs transition ${
                  message.reaction === "dislike"
                    ? "border-rose-500 bg-rose-600 text-white"
                    : "border-slate-200 bg-white text-slate-600 hover:border-rose-200 hover:text-rose-600 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200"
                }`}
              >
                <DislikeIcon className="h-3.5 w-3.5" />
              </button>

              {isErrorMessage ? (
                <>
                  <button
                    type="button"
                    onClick={() => onRegenerate(message.sourcePrompt || "")}
                    className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 transition hover:bg-slate-100 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                  >
                    Retry same
                  </button>
                  <button
                    type="button"
                    onClick={() => onRetryWebOnly(message.sourcePrompt || "")}
                    className="rounded-lg border border-cyan-200 bg-cyan-50 px-2.5 py-1.5 text-xs text-cyan-700 transition hover:bg-cyan-100 dark:border-cyan-800 dark:bg-cyan-950/40 dark:text-cyan-200"
                  >
                    Retry web-first
                  </button>
                </>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </article>
  );
}
