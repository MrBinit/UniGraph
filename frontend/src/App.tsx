import { useEffect, useMemo, useRef, useState } from "react";

import { BrandIcon, BrandLogo } from "./components/Brand";
import { ChatInput } from "./components/ChatInput";
import { MenuIcon } from "./components/Icons";
import { LoginPanel } from "./components/LoginPanel";
import { MessageCard } from "./components/MessageCard";
import { Sidebar } from "./components/Sidebar";
import { clearChatHistory, fetchChatJobTrace, fetchConversations, loginWithPassword, streamChatResponse } from "./lib/api";
import type {
  AuthSession,
  ChatExecutionMode,
  ChatMessage,
  ConversationItem,
  ReactionType,
  StreamEvent,
  TraceEventItem,
} from "./types";

const AUTH_STORAGE_KEY = "ai.chat.frontend.auth";
const THEME_STORAGE_KEY = "ai.chat.frontend.theme";
const ACTIVE_JOB_STORAGE_KEY = "ai.chat.frontend.active_job_id";
const CHAT_MODE_STORAGE_KEY = "ai.chat.frontend.mode";
const CONVERSATION_META_STORAGE_KEY = "ai.chat.frontend.conversation_meta";

type ConversationDateFilter = "all" | "7d" | "30d";

interface ConversationMeta {
  pinned?: boolean;
  starred?: boolean;
  customTitle?: string;
}

interface PromptCommandResult {
  prompt: string;
  mode: ChatExecutionMode | null;
}

function makeId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getStoredAuth(): AuthSession | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as AuthSession;
  } catch {
    return null;
  }
}

function getStoredTheme(): boolean {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) === "dark";
  } catch {
    return false;
  }
}

function getStoredChatMode(): ChatExecutionMode {
  try {
    const candidate = String(localStorage.getItem(CHAT_MODE_STORAGE_KEY) ?? "")
      .trim()
      .toLowerCase();
    if (candidate === "fast" || candidate === "standard") {
      return "standard";
    }
    if (candidate === "deep" || candidate === "auto") {
      return candidate;
    }
  } catch {
    // ignore localStorage access failures and use default
  }
  return "auto";
}

function getStoredConversationMeta(): Record<string, ConversationMeta> {
  try {
    const raw = localStorage.getItem(CONVERSATION_META_STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as Record<string, ConversationMeta>;
    if (!parsed || typeof parsed !== "object") {
      return {};
    }
    return parsed;
  } catch {
    return {};
  }
}

function parsePromptCommands(rawInput: string, defaultMode: ChatExecutionMode): PromptCommandResult {
  const text = rawInput.trim();
  if (!text.startsWith("/")) {
    return { prompt: text, mode: null };
  }

  const [rawCommand, ...rest] = text.split(/\s+/);
  const command = rawCommand.toLowerCase();
  const remaining = rest.join(" ").trim();
  let modeOverride: ChatExecutionMode | null = null;
  let prompt = remaining;

  if (command === "/standard" || command === "/fast") {
    modeOverride = "standard";
  } else if (command === "/deep") {
    modeOverride = "deep";
  } else if (command === "/auto") {
    modeOverride = "auto";
  } else if (command === "/cite") {
    prompt = remaining ? `${remaining}\n\nPlease include clear source citations.` : "";
  } else if (command === "/summarize") {
    prompt = remaining ? `Summarize clearly and concisely:\n\n${remaining}` : "";
  } else if (command === "/web") {
    prompt = remaining
      ? `${remaining}\n\nPrioritize web retrieval and provide source links for all key claims.`
      : "";
    modeOverride = defaultMode === "fast" || defaultMode === "standard" ? "standard" : "deep";
  }

  return { prompt: prompt.trim(), mode: modeOverride };
}

function mergeStreamText(current: string, incoming: string): string {
  const next = incoming;
  if (!next) {
    return current;
  }
  if (!current) {
    return next;
  }
  if (next.startsWith(current)) {
    return next;
  }
  if (current.endsWith(next)) {
    return current;
  }
  return `${current}${next}`;
}

function streamEventText(event: StreamEvent): string {
  return String(event.text ?? event.delta ?? event.token ?? event.content ?? event.message ?? "");
}

function pendingPhaseText(status: string, elapsedSeconds: number): string {
  const normalized = status.trim().toLowerCase();
  if (normalized.includes("queued")) {
    return "Preparing request";
  }
  if (normalized.includes("search")) {
    return "Searching web and knowledge sources";
  }
  if (normalized.includes("retriev")) {
    return "Reviewing retrieved documents";
  }
  if (normalized.includes("rank") || normalized.includes("rerank")) {
    return "Ranking relevant evidence";
  }
  if (normalized.includes("draft") || normalized.includes("generat")) {
    return "Drafting response";
  }
  if (normalized.includes("check") || normalized.includes("validat")) {
    return "Cross-checking response";
  }

  if (elapsedSeconds < 4) {
    return "Searching web and knowledge sources";
  }
  if (elapsedSeconds < 8) {
    return "Reviewing relevant context";
  }
  if (elapsedSeconds < 12) {
    return "Drafting response";
  }
  return "Finalizing answer";
}

function mergeUniqueStrings(current: string[], incoming: string[], limit = 10): string[] {
  const deduped = new Set(current);
  for (const item of incoming) {
    const clean = item.trim();
    if (clean) {
      deduped.add(clean);
    }
  }
  return Array.from(deduped).slice(0, limit);
}

function displayWebsite(value: string): string {
  try {
    const hostname = new URL(value).hostname.trim().toLowerCase();
    return hostname.replace(/^www\./, "");
  } catch {
    return value.trim().toLowerCase();
  }
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

function extractUrlsFromText(input: string): string[] {
  const matches = input.match(/https?:\/\/[^\s)"']+/gi) ?? [];
  const urls = matches
    .map((item) => normalizeUrlCandidate(item))
    .filter((item): item is string => Boolean(item));
  return Array.from(new Set(urls));
}

const TRACE_EVENT_LABELS: Record<string, string> = {
  request_received: "Question received",
  query_intent_classified: "Detected query intent",
  retrieval_query_decomposed: "Expanded search queries",
  query_plan_created: "Search strategy decided",
  query_planner_started: "Planning the search",
  query_planner_completed: "Search plan ready",
  query_planner_skipped: "Skipped planning",
  search_started: "Searching trusted sources",
  search_results: "Collected candidate sources",
  pages_read: "Read top pages",
  facts_extracted: "Extracted evidence",
  gaps_identified: "Found evidence gaps",
  retrieval_verification: "Coverage check",
  source_ranking_completed: "Ranked source quality",
  retrieval_vector_started: "Checking memory index",
  retrieval_vector_completed: "Memory results ready",
  web_retrieval_skipped: "Skipped web expansion",
  web_fallback_started: "Expanded to web search",
  web_fallback_completed: "Web evidence collected",
  retrieval_reranked: "Prioritized strongest evidence",
  retrieval_selective_filter: "Kept highest-signal evidence",
  evidence_selected: "Final evidence selected",
  evidence_trust_scored: "Scored evidence trust",
  citation_grounding_ready: "Citation mapping ready",
  answer_planning_started: "Planning answer structure",
  answer_plan_created: "Answer plan drafted",
  answer_planning_completed: "Answer plan approved",
  model_round_started: "Drafting answer",
  model_round_completed: "Draft completed",
  answer_verification_completed: "Checked claims and citations",
  fast_refine_started: "Refining draft with citations",
  fast_refine_completed: "Refinement completed",
  fast_refine_failed: "Refinement skipped",
  answer_uncertainty_flagged: "Marked uncertainty",
  answer_synthesis_completed: "Answer synthesized",
  answer_finalized: "Answer finalized",
  job_processing_started: "Job started",
  job_completed: "Job completed",
  job_failed: "Job failed",
};

const THOUGHT_TRACE_TYPES = new Set<string>([
  "query_intent_classified",
  "retrieval_query_decomposed",
  "query_planner_started",
  "query_plan_created",
  "query_planner_completed",
  "query_planner_skipped",
  "search_started",
  "search_results",
  "pages_read",
  "facts_extracted",
  "gaps_identified",
  "retrieval_verification",
  "source_ranking_completed",
  "answer_planning_started",
  "answer_planning_completed",
  "answer_verification_completed",
  "answer_finalized",
]);

function normalizeConfidence(value: unknown): number | undefined {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return undefined;
  }
  return Math.max(0, Math.min(1, numeric));
}

function normalizeCoverage(value: unknown): number | undefined {
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    return undefined;
  }
  return Math.max(0, Math.min(1, numeric));
}

function extractTraceTrustSignals(payload?: Record<string, unknown>) {
  if (!payload) {
    return {};
  }
  const freshnessRaw = String(payload.freshness ?? "").trim().toLowerCase();
  const trustFreshness =
    freshnessRaw && freshnessRaw !== "unknown" ? freshnessRaw : undefined;
  const uncertaintyRaw = payload.uncertainty_reasons;
  const uncertaintyReasons = Array.isArray(uncertaintyRaw)
    ? uncertaintyRaw
        .map((item) => String(item).trim())
        .filter((item) => Boolean(item))
        .slice(0, 3)
    : [];
  return {
    trustConfidence: normalizeConfidence(payload.confidence),
    trustFreshness,
    trustContradiction:
      typeof payload.contradiction_flag === "boolean" ? payload.contradiction_flag : undefined,
    claimCitationCoverage: normalizeCoverage(payload.claim_citation_coverage),
    uncertaintyReasons: uncertaintyReasons.length ? uncertaintyReasons : undefined,
  };
}

function traceLabel(type: string): string {
  const normalized = type.trim().toLowerCase();
  return TRACE_EVENT_LABELS[normalized] ?? normalized.replace(/_/g, " ");
}

function isThoughtTraceEvent(type: string): boolean {
  return THOUGHT_TRACE_TYPES.has(String(type).trim().toLowerCase());
}

function normalizeTraceTextList(value: unknown, limit = 3): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => String(item ?? "").trim())
    .filter((item) => Boolean(item))
    .slice(0, limit);
}

function compactThoughtFragment(text: string, maxChars = 72): string {
  const normalized = String(text).replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxChars - 1))}…`;
}

function compactThoughtLine(prefix: string, values: string[]): string {
  if (!values.length) {
    return "";
  }
  const compact = values
    .map((item) => compactThoughtFragment(item))
    .filter((item) => Boolean(item))
    .join(" | ");
  if (!compact) {
    return "";
  }
  return `${prefix}: ${compact}`.slice(0, 180);
}

function traceThoughtDetails(event: TraceEventItem): string[] {
  const payload =
    event.payload && typeof event.payload === "object"
      ? (event.payload as Record<string, unknown>)
      : null;
  if (!payload) {
    return [];
  }
  const type = String(event.type).trim().toLowerCase();
  const details: string[] = [];

  if (type === "query_plan_created" || type === "query_planner_completed") {
    const planner = String(payload.planner ?? "").trim();
    const llmUsed = typeof payload.llm_used === "boolean" ? payload.llm_used : undefined;
    if (planner) {
      details.push(
        `Planner: ${planner}${llmUsed === true ? " (llm)" : llmUsed === false ? " (heuristic)" : ""}`
      );
    }
    const queryLine = compactThoughtLine("Planner queries", normalizeTraceTextList(payload.queries, 4));
    if (queryLine) {
      details.push(queryLine);
    }
    const subquestionLine = compactThoughtLine(
      "Planner checks",
      normalizeTraceTextList(payload.subquestions, 4)
    );
    if (subquestionLine) {
      details.push(subquestionLine);
    }
  }

  if (type === "search_started") {
    const searchLine = compactThoughtLine("Searching", normalizeTraceTextList(payload.queries, 4));
    if (searchLine) {
      details.push(searchLine);
    }
  }

  if (type === "gaps_identified" || type === "retrieval_verification") {
    const missing = normalizeTraceTextList(payload.missing_subquestions, 3);
    const missingLine = compactThoughtLine("Coverage gaps", missing);
    if (missingLine) {
      details.push(missingLine);
    }
    const followUps = normalizeTraceTextList(payload.follow_up_queries, 3);
    const followUpLine = compactThoughtLine("Follow-up queries", followUps);
    if (followUpLine) {
      details.push(followUpLine);
    }
    const uniqueDomainCount = Number(payload.unique_domain_count);
    const minUniqueDomains = Number(payload.min_unique_domains);
    if (Number.isFinite(uniqueDomainCount) && Number.isFinite(minUniqueDomains)) {
      details.push(`Domain coverage: ${uniqueDomainCount}/${minUniqueDomains}`);
    }
  }

  return details.slice(0, 4);
}

function formatDurationLabel(totalSeconds: number): string {
  const safeSeconds = Math.max(1, Math.round(totalSeconds));
  if (safeSeconds < 60) {
    return `${safeSeconds}s`;
  }
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds % 60;
  if (minutes < 60) {
    return remainingSeconds ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
}

function mergeTraceEvents(current: TraceEventItem[], incoming: TraceEventItem[]): TraceEventItem[] {
  const seen = new Set(current.map((event) => `${event.type}|${event.timestamp ?? ""}`));
  const merged = [...current];
  for (const event of incoming) {
    const key = `${event.type}|${event.timestamp ?? ""}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(event);
  }
  return merged.slice(-120);
}

function extractTraceWebsites(payload?: Record<string, unknown>): string[] {
  if (!payload) {
    return [];
  }
  const urls = new Set<string>();
  const collect = (value: unknown) => {
    if (typeof value === "string") {
      for (const match of extractUrlsFromText(value)) {
        urls.add(displayWebsite(match));
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        collect(item);
      }
      return;
    }
    if (value && typeof value === "object") {
      for (const nested of Object.values(value as Record<string, unknown>)) {
        collect(nested);
      }
    }
  };
  const sourceKeys = ["urls", "source_urls", "sources", "websites"];
  for (const key of sourceKeys) {
    collect(payload[key]);
  }
  return Array.from(urls);
}

function extractTraceUrls(payload?: Record<string, unknown>): string[] {
  if (!payload) {
    return [];
  }
  const urls = new Set<string>();
  const collect = (value: unknown) => {
    if (typeof value === "string") {
      for (const match of extractUrlsFromText(value)) {
        urls.add(match);
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        collect(item);
      }
      return;
    }
    if (value && typeof value === "object") {
      for (const nested of Object.values(value as Record<string, unknown>)) {
        collect(nested);
      }
    }
  };

  const urlKeys = ["urls", "source_urls", "sources", "websites", "citations", "results", "items"];
  for (const key of urlKeys) {
    collect(payload[key]);
  }
  return Array.from(urls);
}

function isLowSignalReasoningStep(step: string): boolean {
  const normalized = step.trim().toLowerCase();
  return (
    normalized === "queued" ||
    normalized === "processing" ||
    normalized === "status" ||
    normalized === "request received" ||
    normalized === "job processing started"
  );
}

export default function App() {
  const [auth, setAuth] = useState<AuthSession | null>(() => getStoredAuth());
  const [darkMode, setDarkMode] = useState<boolean>(() => getStoredTheme());
  const [chatMode, setChatMode] = useState<ChatExecutionMode>(() => getStoredChatMode());
  const [statusText, setStatusText] = useState("Ready");
  const [statusError, setStatusError] = useState(false);
  const [canCancelQueued, setCanCancelQueued] = useState(false);
  const [isAwaitingFirstChunk, setIsAwaitingFirstChunk] = useState(false);
  const [reasoningSteps, setReasoningSteps] = useState<string[]>([]);
  const [searchedWebsites, setSearchedWebsites] = useState<string[]>([]);
  const [workLogEvents, setWorkLogEvents] = useState<TraceEventItem[]>([]);
  const [activeJobId, setActiveJobId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [conversationMetaById, setConversationMetaById] = useState<Record<string, ConversationMeta>>(
    () => getStoredConversationMeta()
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [conversationDateFilter, setConversationDateFilter] = useState<ConversationDateFilter>("all");
  const [activeConversationId, setActiveConversationId] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [loginError, setLoginError] = useState("");
  const [isLoadingLogin, setIsLoadingLogin] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [deletingConversationId, setDeletingConversationId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const [autoScrollPaused, setAutoScrollPaused] = useState(false);

  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const shouldAutoScrollRef = useRef(true);
  const streamAbortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = (behavior: ScrollBehavior, force = false) => {
    if (!force && autoScrollPaused) {
      return;
    }
    if (!force && !shouldAutoScrollRef.current) {
      return;
    }
    if (force) {
      shouldAutoScrollRef.current = true;
    }
    messageEndRef.current?.scrollIntoView({ behavior, block: "end" });
    setShowJumpToLatest(false);
  };

  const handleChatScroll = () => {
    const container = chatContainerRef.current;
    if (!container) {
      return;
    }
    const threshold = 120;
    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    const isNearBottom = distanceFromBottom <= threshold;
    shouldAutoScrollRef.current = isNearBottom;
    setShowJumpToLatest(!isNearBottom);
  };

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
    localStorage.setItem(THEME_STORAGE_KEY, darkMode ? "dark" : "light");
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem(CHAT_MODE_STORAGE_KEY, chatMode);
  }, [chatMode]);

  useEffect(() => {
    localStorage.setItem(CONVERSATION_META_STORAGE_KEY, JSON.stringify(conversationMetaById));
  }, [conversationMetaById]);

  useEffect(() => {
    scrollToBottom(isSending ? "auto" : "smooth");
  }, [autoScrollPaused, isSending, messages]);

  useEffect(() => {
    if (!isSending && !isAwaitingFirstChunk) {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      scrollToBottom("auto");
    });
    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [isAwaitingFirstChunk, isSending, reasoningSteps, searchedWebsites, workLogEvents]);

  useEffect(() => {
    const session = auth;
    if (!session) {
      setConversations([]);
      return;
    }

    const sessionUserId = session.userId;
    const sessionToken = session.token;
    let cancelled = false;

    async function loadHistory() {
      try {
        const history = await fetchConversations(sessionUserId, sessionToken, 80);
        if (cancelled) return;
        setConversations(history);
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : "Failed to load conversations.";
        setStatusText(message);
        setStatusError(true);
      }
    }

    void loadHistory();

    return () => {
      cancelled = true;
    };
  }, [auth]);

  useEffect(() => {
    const session = auth;
    if (!session) {
      setWorkLogEvents([]);
      setActiveJobId("");
      return;
    }
    const sessionToken = session.token;
    const savedJobId = String(localStorage.getItem(ACTIVE_JOB_STORAGE_KEY) ?? "").trim();
    if (!savedJobId) {
      return;
    }

    let cancelled = false;
    async function replayTrace() {
      try {
        const replay = await fetchChatJobTrace(sessionToken, savedJobId);
        if (cancelled) return;
        setActiveJobId(replay.jobId);
        setWorkLogEvents(replay.traceEvents);
        setReasoningSteps(
          mergeUniqueStrings([], replay.traceEvents.map((event) => traceLabel(event.type)), 12)
        );
        setSearchedWebsites((prev) =>
          mergeUniqueStrings(prev, replay.websites.map(displayWebsite), 12)
        );
        const terminal = replay.traceEvents.some((event) => {
          const kind = event.type.toLowerCase();
          return kind === "job_completed" || kind === "job_failed";
        });
        if (terminal) {
          localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
          setActiveJobId("");
        }
        setStatusText(`Recovered work log for job ${savedJobId.slice(0, 8)}...`);
        setStatusError(false);
      } catch {
        if (cancelled) return;
        localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
        setActiveJobId("");
      }
    }

    void replayTrace();
    return () => {
      cancelled = true;
    };
  }, [auth]);

  useEffect(() => {
    if (!auth || !activeJobId || !isSending) {
      return;
    }

    let cancelled = false;
    const pollTrace = async () => {
      try {
        const replay = await fetchChatJobTrace(auth.token, activeJobId);
        if (cancelled) return;
        setWorkLogEvents((prev) => mergeTraceEvents(prev, replay.traceEvents));
        setSearchedWebsites((prev) =>
          mergeUniqueStrings(prev, replay.websites.map(displayWebsite), 12)
        );
        const terminal = replay.traceEvents.some((event) => {
          const kind = event.type.toLowerCase();
          return kind === "job_completed" || kind === "job_failed";
        });
        if (terminal) {
          localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
          setActiveJobId("");
        }
      } catch {
        // ignore polling failures; live stream may still continue
      }
    };

    void pollTrace();
    const intervalId = window.setInterval(() => {
      void pollTrace();
    }, 2500);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [auth, activeJobId, isSending]);

  const enrichedConversations = useMemo(() => {
    return conversations.map((item) => {
      const meta = conversationMetaById[item.conversationId] ?? {};
      const customTitle = String(meta.customTitle ?? "").trim();
      return {
        ...item,
        title: customTitle || item.title,
      };
    });
  }, [conversationMetaById, conversations]);

  const filteredConversations = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const now = Date.now();
    const cutoffMs =
      conversationDateFilter === "7d"
        ? 7 * 24 * 60 * 60 * 1000
        : conversationDateFilter === "30d"
          ? 30 * 24 * 60 * 60 * 1000
          : null;
    return enrichedConversations
      .filter((item) => {
        if (query) {
          const matches =
            item.title.toLowerCase().includes(query) ||
            item.prompt.toLowerCase().includes(query) ||
            item.answer.toLowerCase().includes(query);
          if (!matches) {
            return false;
          }
        }
        if (cutoffMs === null) {
          return true;
        }
        const createdAtMs = new Date(item.createdAt).getTime();
        return Number.isFinite(createdAtMs) && now - createdAtMs <= cutoffMs;
      })
      .sort((a, b) => {
        const aMeta = conversationMetaById[a.conversationId] ?? {};
        const bMeta = conversationMetaById[b.conversationId] ?? {};
        if (Boolean(aMeta.pinned) !== Boolean(bMeta.pinned)) {
          return aMeta.pinned ? -1 : 1;
        }
        if (Boolean(aMeta.starred) !== Boolean(bMeta.starred)) {
          return aMeta.starred ? -1 : 1;
        }
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });
  }, [conversationDateFilter, conversationMetaById, enrichedConversations, searchQuery]);

  const yourConversations = useMemo(() => filteredConversations.slice(0, 24), [filteredConversations]);

  const lastSevenDays = useMemo(() => {
    const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
    return filteredConversations.filter((item) => {
      const time = new Date(item.createdAt).getTime();
      return Number.isFinite(time) && time >= cutoff;
    });
  }, [filteredConversations]);

  const followUpChips = useMemo(() => {
    const hasAssistantAnswer = messages.some((message) => message.role === "assistant");
    if (!hasAssistantAnswer) {
      return [
        "Give me a quick overview",
        "What should I ask next?",
        "Summarize this topic in bullets",
      ];
    }
    return [
      "Summarize this in 5 bullets",
      "What are the risks and tradeoffs?",
      "Compare top options in a table",
      "Give me a step-by-step action plan",
    ];
  }, [messages]);
  const updateMessage = (messageId: string, updater: (message: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((item) => (item.id === messageId ? updater(item) : item)));
  };

  const refreshHistory = async () => {
    if (!auth) return;
    const history = await fetchConversations(auth.userId, auth.token, 80);
    setConversations(history);
  };

  const sendPrompt = async (
    prompt: string,
    includeUserMessage = true,
    options?: { modeOverride?: ChatExecutionMode; statusLabel?: string }
  ) => {
    if (!auth || isSending) {
      return;
    }

    const cleanPrompt = prompt.trim();
    if (!cleanPrompt) {
      return;
    }

    const requestedMode = options?.modeOverride ?? chatMode;

    setIsSending(true);
    setStatusError(false);
    setStatusText(options?.statusLabel ?? (includeUserMessage ? "Submitting prompt..." : "Regenerating response..."));
    setCanCancelQueued(false);
    setIsAwaitingFirstChunk(true);
    setReasoningSteps([]);
    setSearchedWebsites([]);
    setWorkLogEvents([]);
    setActiveJobId("");
    localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
    const streamAbortController = new AbortController();
    streamAbortControllerRef.current = streamAbortController;

    const assistantId = makeId();

    if (includeUserMessage) {
      const userMessage: ChatMessage = {
        id: makeId(),
        role: "user",
        username: auth.username,
        content: cleanPrompt,
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);
    }

    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      username: "UNIGRAPH",
      content: "Thinking...",
      createdAt: new Date().toISOString(),
      sourcePrompt: cleanPrompt,
      executionMode: requestedMode,
      reaction: null,
    };

    setMessages((prev) => [...prev, assistantMessage]);
    shouldAutoScrollRef.current = true;
    scrollToBottom("smooth", true);

    let streamTargetText = "";
    let streamRenderedText = "";
    let doneReceived = false;
    let firstTokenReceived = false;
    let pumpResolved = false;
    const waitingFrames = ["", ".", "..", "..."];
    let waitingFrameIndex = 0;
    const pendingStartedAt = Date.now();
    const traceSourceUrls = new Set<string>();
    const traceTimestampMs: number[] = [];
    let messageReasoningSteps: string[] = [];
    let messageSearchedWebsites: string[] = [];
    const messageTraceEvents: TraceEventItem[] = [];
    let trustConfidence: number | undefined;
    let trustFreshness: string | undefined;
    let trustContradiction: boolean | undefined;
    let claimCitationCoverage: number | undefined;
    let uncertaintyReasons: string[] | undefined;
    let latestServerStatus = "queued";
    const thinkingInterval = window.setInterval(() => {
      if (firstTokenReceived) {
        return;
      }
      waitingFrameIndex = (waitingFrameIndex + 1) % waitingFrames.length;
      const elapsedSeconds = Math.max(1, Math.floor((Date.now() - pendingStartedAt) / 1000));
      const phase = pendingPhaseText(latestServerStatus, elapsedSeconds);
      updateMessage(assistantId, (item) => ({
        ...item,
        workedForLabel: formatDurationLabel((Date.now() - pendingStartedAt) / 1000),
        reasoningSteps: messageReasoningSteps,
        searchedWebsites: messageSearchedWebsites,
        content: `${phase}${waitingFrames[waitingFrameIndex]}\nStill working (${elapsedSeconds}s)`,
      }));
    }, 350);
    let pumpIntervalId: number | null = null;
    let resolvePump: (() => void) | null = null;
    const pumpDone = new Promise<void>((resolve) => {
      resolvePump = resolve;
    });

    const stopThinking = () => {
      window.clearInterval(thinkingInterval);
    };
    const syncAssistantThoughtMeta = () => {
      updateMessage(assistantId, (item) => ({
        ...item,
        workedForLabel: formatDurationLabel((Date.now() - pendingStartedAt) / 1000),
        sourceUrls:
          item.sourceUrls?.length
            ? item.sourceUrls
            : Array.from(traceSourceUrls).slice(0, 12),
        reasoningSteps: messageReasoningSteps,
        searchedWebsites: messageSearchedWebsites,
        traceEvents: messageTraceEvents.slice(-50),
        trustConfidence,
        trustFreshness,
        trustContradiction,
        claimCitationCoverage,
        uncertaintyReasons,
      }));
    };
    const addReasoningSteps = (steps: string[] | undefined) => {
      if (!steps?.length) {
        return;
      }
      const filtered = steps.filter((step) => !isLowSignalReasoningStep(step));
      if (!filtered.length) {
        return;
      }
      messageReasoningSteps = mergeUniqueStrings(messageReasoningSteps, filtered, 12);
      setReasoningSteps((prev) => mergeUniqueStrings(prev, filtered, 8));
      syncAssistantThoughtMeta();
    };
    const addSearchedWebsites = (websites: string[] | undefined) => {
      if (!websites?.length) {
        return;
      }
      const normalized = websites.map(displayWebsite);
      messageSearchedWebsites = mergeUniqueStrings(messageSearchedWebsites, normalized, 15);
      setSearchedWebsites((prev) => mergeUniqueStrings(prev, normalized, 8));
      syncAssistantThoughtMeta();
    };
    const setCurrentJob = (jobId: string | undefined) => {
      const clean = String(jobId ?? "").trim();
      if (!clean) {
        return;
      }
      setActiveJobId(clean);
      localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, clean);
    };
    const appendTraceEvent = (trace: TraceEventItem | undefined) => {
      if (!trace) {
        return;
      }
      messageTraceEvents.push(trace);
      if (String(trace.type).trim().toLowerCase() === "answer_finalized") {
        const trustSignals = extractTraceTrustSignals(trace.payload);
        trustConfidence = trustSignals.trustConfidence ?? trustConfidence;
        trustFreshness = trustSignals.trustFreshness ?? trustFreshness;
        trustContradiction = trustSignals.trustContradiction ?? trustContradiction;
        claimCitationCoverage = trustSignals.claimCitationCoverage ?? claimCitationCoverage;
        uncertaintyReasons = trustSignals.uncertaintyReasons ?? uncertaintyReasons;
      }
      for (const url of extractTraceUrls(trace.payload)) {
        traceSourceUrls.add(url);
      }
      if (trace.timestamp) {
        const traceTimeMs = new Date(trace.timestamp).getTime();
        if (Number.isFinite(traceTimeMs)) {
          traceTimestampMs.push(traceTimeMs);
        }
      }
      setWorkLogEvents((prev) => mergeTraceEvents(prev, [trace]));
      const traceDetails = traceThoughtDetails(trace);
      if (isThoughtTraceEvent(trace.type)) {
        addReasoningSteps([traceLabel(trace.type), ...traceDetails]);
      } else if (traceDetails.length) {
        addReasoningSteps(traceDetails);
      }
      addSearchedWebsites(extractTraceWebsites(trace.payload));
      setStatusText(`Status: ${traceLabel(trace.type)}`);
      setStatusError(false);
      syncAssistantThoughtMeta();
    };
    const finalizeAssistantMeta = (content: string) => {
      const finalizedAt = Date.now();
      const startedAt = traceTimestampMs.length ? Math.min(pendingStartedAt, ...traceTimestampMs) : pendingStartedAt;
      const endedAt = traceTimestampMs.length ? Math.max(finalizedAt, ...traceTimestampMs) : finalizedAt;
      const workedForLabel = formatDurationLabel((endedAt - startedAt) / 1000);
      const sourceUrls = Array.from(
        new Set([...traceSourceUrls, ...extractUrlsFromText(content)])
      ).slice(0, 12);
      updateMessage(assistantId, (item) => ({
        ...item,
        workedForLabel,
        sourceUrls,
        reasoningSteps: messageReasoningSteps,
        searchedWebsites: messageSearchedWebsites,
        traceEvents: messageTraceEvents.slice(-50),
        trustConfidence,
        trustFreshness,
        trustContradiction,
        claimCitationCoverage,
        uncertaintyReasons,
      }));
    };
    const maybeResolvePump = () => {
      if (pumpResolved || !doneReceived) {
        return;
      }
      if (streamRenderedText.length < streamTargetText.length) {
        return;
      }
      if (pumpIntervalId !== null) {
        window.clearInterval(pumpIntervalId);
      }
      updateMessage(assistantId, (item) => ({
        ...item,
        content: streamRenderedText || "(No response text returned.)",
      }));
      pumpResolved = true;
      resolvePump?.();
    };
    const startRenderPump = () => {
      if (pumpIntervalId !== null) {
        return;
      }
      pumpIntervalId = window.setInterval(() => {
        if (streamRenderedText.length < streamTargetText.length) {
          const nextLength = Math.min(streamTargetText.length, streamRenderedText.length + 4);
          streamRenderedText = streamTargetText.slice(0, nextLength);
          updateMessage(assistantId, (item) => ({ ...item, content: `${streamRenderedText}▍` }));
          return;
        }
        maybeResolvePump();
      }, 16);
    };

    try {
      await streamChatResponse(
        auth.token,
        {
          userId: auth.userId,
          sessionId: auth.sessionId || auth.userId,
          prompt: cleanPrompt,
          mode: requestedMode,
        },
        (event: StreamEvent) => {
          if (event.type === "queued") {
            latestServerStatus = event.status ?? "queued";
            setCurrentJob(event.job_id);
            addSearchedWebsites(event.websites);
            setCanCancelQueued(true);
            setStatusText("Status: Preparing request");
            setStatusError(false);
            return;
          }
          if (event.type === "status") {
            latestServerStatus = event.status ?? latestServerStatus;
            setCurrentJob(event.job_id);
            addSearchedWebsites(event.websites);
            const isQueuedStatus = String(latestServerStatus).toLowerCase().includes("queued");
            setCanCancelQueued(isQueuedStatus && !firstTokenReceived);
            const elapsedSeconds = Math.max(1, Math.floor((Date.now() - pendingStartedAt) / 1000));
            setStatusText(`Status: ${pendingPhaseText(latestServerStatus, elapsedSeconds)}`);
            setStatusError(false);
            return;
          }
          if (event.type === "trace") {
            setCurrentJob(event.job_id);
            appendTraceEvent(event.trace);
            addSearchedWebsites(event.websites);
            return;
          }
          if (event.type === "search") {
            addSearchedWebsites(event.websites);
            setStatusText("Status: Searching external sources");
            setStatusError(false);
            return;
          }
          if (event.type === "reasoning") {
            addReasoningSteps(event.steps);
            if (event.steps?.[0]) {
              setStatusText(`Status: ${event.steps[0]}`);
            }
            setStatusError(false);
            return;
          }
          if (event.type === "chunk") {
            const delta = streamEventText(event);
            if (!delta) {
              return;
            }
            if (!firstTokenReceived) {
              firstTokenReceived = true;
              stopThinking();
              startRenderPump();
              setCanCancelQueued(false);
              setIsAwaitingFirstChunk(false);
            }
            streamTargetText = mergeStreamText(streamTargetText, delta);
            setStatusText("Streaming response...");
            setStatusError(false);
            return;
          }
          if (event.type === "done") {
            doneReceived = true;
            stopThinking();
            maybeResolvePump();
            setCanCancelQueued(false);
            setIsAwaitingFirstChunk(false);
            setActiveJobId("");
            localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
            setStatusText("Complete");
            setStatusError(false);
            return;
          }
          if (event.type === "error") {
            throw new Error(event.detail ?? "Chat stream failed.");
          }
        },
        { signal: streamAbortController.signal }
      );

      stopThinking();
      doneReceived = true;
      if (!firstTokenReceived) {
        streamTargetText = streamTargetText || "";
        streamRenderedText = streamTargetText;
        maybeResolvePump();
      } else {
        maybeResolvePump();
        await pumpDone;
      }

      if (streamTargetText) {
        setStatusText("Complete");
        setStatusError(false);
      }
      finalizeAssistantMeta(streamTargetText || streamRenderedText);

      setInputValue("");
      await refreshHistory();
    } catch (error) {
      stopThinking();
      if (pumpIntervalId !== null) {
        window.clearInterval(pumpIntervalId);
      }
      const isAbortError =
        (error instanceof DOMException && error.name === "AbortError") ||
        (error instanceof Error && error.name === "AbortError");
      if (isAbortError) {
        updateMessage(assistantId, (item) => ({
          ...item,
          content: "Request canceled.",
          workedForLabel: formatDurationLabel((Date.now() - pendingStartedAt) / 1000),
          sourceUrls: Array.from(traceSourceUrls),
          reasoningSteps: messageReasoningSteps,
          searchedWebsites: messageSearchedWebsites,
          traceEvents: messageTraceEvents.slice(-50),
        }));
        setActiveJobId("");
        localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
        setStatusText("Request canceled.");
        setStatusError(false);
      } else {
        const message = error instanceof Error ? error.message : "Chat request failed.";
        const errorText = `Error: ${message}`;
        updateMessage(assistantId, (item) => ({
          ...item,
          content: errorText,
          workedForLabel: formatDurationLabel((Date.now() - pendingStartedAt) / 1000),
          sourceUrls: Array.from(new Set([...traceSourceUrls, ...extractUrlsFromText(errorText)])).slice(0, 12),
          reasoningSteps: messageReasoningSteps,
          searchedWebsites: messageSearchedWebsites,
          traceEvents: messageTraceEvents.slice(-50),
        }));
        setActiveJobId("");
        localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
        setStatusText(message);
        setStatusError(true);
      }
    } finally {
      stopThinking();
      if (pumpIntervalId !== null) {
        window.clearInterval(pumpIntervalId);
      }
      setCanCancelQueued(false);
      setIsAwaitingFirstChunk(false);
      streamAbortControllerRef.current = null;
      setIsSending(false);
    }
  };

  const handleCancelQueuedRequest = () => {
    if (!canCancelQueued) {
      return;
    }
    setStatusText("Canceling request...");
    setStatusError(false);
    setCanCancelQueued(false);
    streamAbortControllerRef.current?.abort();
  };

  const handleLogin = async (username: string, password: string, sessionId: string) => {
    setIsLoadingLogin(true);
    setLoginError("");

    try {
      const payload = await loginWithPassword(username, password);
      const session: AuthSession = {
        token: payload.access_token,
        userId: payload.user_id,
        username,
        roles: payload.roles,
        sessionId: sessionId || payload.user_id,
      };
      setAuth(session);
      localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
      setStatusText("Signed in");
      setStatusError(false);
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "Sign-in failed.");
    } finally {
      setIsLoadingLogin(false);
    }
  };

  const handleConversationSelect = (conversation: ConversationItem) => {
    setActiveConversationId(conversation.conversationId);
    setReasoningSteps([]);
    setSearchedWebsites([]);
    setWorkLogEvents([]);
    setActiveJobId("");
    localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
    setMessages([
      {
        id: makeId(),
        role: "user",
        username: auth?.username || "You",
        content: conversation.prompt,
        createdAt: conversation.createdAt,
      },
      {
        id: makeId(),
        role: "assistant",
        username: "UNIGRAPH",
        content: conversation.answer,
        createdAt: conversation.createdAt,
        sourcePrompt: conversation.prompt,
        sourceUrls: extractUrlsFromText(conversation.answer).slice(0, 12),
        reaction: null,
      },
    ]);
    setStatusText(`Loaded conversation: ${conversation.title}`);
    setStatusError(false);
  };

  const handleRegenerate = async (sourcePrompt: string) => {
    await sendPrompt(sourcePrompt, false);
  };

  const handleRegenerateInDeep = async (sourcePrompt: string) => {
    await sendPrompt(sourcePrompt, false, { modeOverride: "deep", statusLabel: "Regenerating in deep mode..." });
  };

  const handleRetryWebOnly = async (sourcePrompt: string) => {
    const webFirstPrompt = `${sourcePrompt}\n\nPrioritize web retrieval and include direct source URLs for key claims.`;
    await sendPrompt(webFirstPrompt, false, { modeOverride: "deep", statusLabel: "Retrying with web-first strategy..." });
  };

  const handleReaction = (messageId: string, reaction: ReactionType) => {
    updateMessage(messageId, (item) => ({ ...item, reaction }));
  };

  const handleSubmitFromComposer = async (rawInput: string) => {
    const parsed = parsePromptCommands(rawInput, chatMode);
    if (!parsed.prompt) {
      setStatusText("Write a prompt after the slash command.");
      setStatusError(true);
      return;
    }
    if (parsed.mode && parsed.mode !== chatMode) {
      setChatMode(parsed.mode);
    }
    await sendPrompt(parsed.prompt, true, {
      modeOverride: parsed.mode ?? chatMode,
      statusLabel: parsed.mode ? `Submitting prompt in ${parsed.mode} mode...` : "Submitting prompt...",
    });
  };

  const handleStopGeneration = () => {
    if (!isSending) {
      return;
    }
    setStatusText("Stopping generation...");
    setStatusError(false);
    setCanCancelQueued(false);
    streamAbortControllerRef.current?.abort();
  };

  const handleDeleteConversation = async (conversation: ConversationItem) => {
    if (!auth || deletingConversationId) {
      return;
    }

    setDeletingConversationId(conversation.conversationId);
    setStatusError(false);
    setStatusText("Deleting conversation...");

    try {
      await clearChatHistory(auth.userId, auth.token, conversation.conversationId);
      setConversations((prev) =>
        prev.filter((item) => item.conversationId !== conversation.conversationId)
      );
      setConversationMetaById((prev) => {
        const next = { ...prev };
        delete next[conversation.conversationId];
        return next;
      });
      if (activeConversationId === conversation.conversationId) {
        setMessages([]);
        setActiveConversationId("");
        setInputValue("");
      }
      setStatusText("Conversation deleted.");
      setStatusError(false);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to delete conversation.";
      setStatusText(message);
      setStatusError(true);
    } finally {
      setDeletingConversationId(null);
    }
  };

  const isPinnedConversation = (conversationId: string) =>
    Boolean(conversationMetaById[conversationId]?.pinned);
  const isStarredConversation = (conversationId: string) =>
    Boolean(conversationMetaById[conversationId]?.starred);

  const handleTogglePinnedConversation = (conversationId: string) => {
    setConversationMetaById((prev) => {
      const current = prev[conversationId] ?? {};
      return {
        ...prev,
        [conversationId]: {
          ...current,
          pinned: !current.pinned,
        },
      };
    });
  };

  const handleToggleStarredConversation = (conversationId: string) => {
    setConversationMetaById((prev) => {
      const current = prev[conversationId] ?? {};
      return {
        ...prev,
        [conversationId]: {
          ...current,
          starred: !current.starred,
        },
      };
    });
  };

  const handleRenameConversation = (conversation: ConversationItem) => {
    const nextTitle = window.prompt("Rename conversation", conversation.title);
    if (nextTitle === null) {
      return;
    }
    const trimmed = nextTitle.trim();
    setConversationMetaById((prev) => {
      const current = prev[conversation.conversationId] ?? {};
      return {
        ...prev,
        [conversation.conversationId]: {
          ...current,
          customTitle: trimmed || undefined,
        },
      };
    });
  };

  const handleLogout = () => {
    streamAbortControllerRef.current?.abort();
    streamAbortControllerRef.current = null;
    localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setAuth(null);
    setMessages([]);
    setConversations([]);
    setActiveConversationId("");
    setInputValue("");
    setReasoningSteps([]);
    setSearchedWebsites([]);
    setWorkLogEvents([]);
    setActiveJobId("");
    setCanCancelQueued(false);
    setIsAwaitingFirstChunk(false);
    setStatusText("Signed out");
    setStatusError(false);
  };

  if (!auth) {
    return <LoginPanel loading={isLoadingLogin} error={loginError} onSubmit={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-app-gradient text-slate-800 dark:bg-app-gradient-dark dark:text-slate-100">
      <Sidebar
        auth={auth}
        conversations={yourConversations}
        recentConversations={lastSevenDays}
        activeConversationId={activeConversationId}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        dateFilter={conversationDateFilter}
        onDateFilterChange={setConversationDateFilter}
        onNewChat={() => {
          setMessages([]);
          setActiveConversationId("");
          setReasoningSteps([]);
          setSearchedWebsites([]);
          setWorkLogEvents([]);
          setActiveJobId("");
          localStorage.removeItem(ACTIVE_JOB_STORAGE_KEY);
          setStatusText("New chat started");
          setStatusError(false);
          setIsSidebarOpen(false);
        }}
        onDeleteConversation={async (conversation) => {
          await handleDeleteConversation(conversation);
          setIsSidebarOpen(false);
        }}
        deletingConversationId={deletingConversationId}
        isPinned={isPinnedConversation}
        isStarred={isStarredConversation}
        onTogglePin={handleTogglePinnedConversation}
        onToggleStar={handleToggleStarredConversation}
        onRenameConversation={handleRenameConversation}
        onSelectConversation={handleConversationSelect}
        onToggleTheme={() => setDarkMode((prev) => !prev)}
        darkMode={darkMode}
        mobileOpen={isSidebarOpen}
        onCloseMobile={() => setIsSidebarOpen(false)}
      />

      <div className="md:ml-[260px]">
        <header className="sticky top-0 z-20 border-b border-blue-100/80 bg-white/90 px-5 py-4 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80">
          <div className="mx-auto flex w-full max-w-4xl items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="rounded-lg p-1.5 text-slate-600 hover:bg-blue-50 hover:text-brand-blue dark:text-slate-300 dark:hover:bg-slate-800 md:hidden"
                onClick={() => setIsSidebarOpen(true)}
                aria-label="Open sidebar"
              >
                <MenuIcon className="h-5 w-5" />
              </button>
              <BrandIcon className="h-7 w-7 rounded-md" />
              <BrandLogo compact className="hidden sm:block" />
            </div>
            <div className="flex items-start gap-3 text-sm text-slate-500 dark:text-slate-400">
              <div className="max-w-[320px]">
                <span className={statusError ? "rounded-md bg-rose-50 px-2 py-1 text-rose-600 dark:bg-rose-950/40 dark:text-rose-300" : ""}>
                  {statusText}
                </span>
                {isAwaitingFirstChunk ? (
                  <div className="mt-1 flex items-center gap-1 text-[11px] text-slate-500 dark:text-slate-400">
                    {[0, 1, 2].map((index) => (
                      <span
                        key={index}
                        className="h-1.5 w-1.5 animate-pulse rounded-full bg-brand-blue/80 dark:bg-brand-red/80"
                        style={{ animationDelay: `${index * 160}ms` }}
                      />
                    ))}
                    <span className="ml-1">Working...</span>
                  </div>
                ) : null}
              </div>
              {isSending ? (
                <button
                  type="button"
                  onClick={canCancelQueued ? handleCancelQueuedRequest : handleStopGeneration}
                  className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-700 hover:bg-amber-100 dark:border-amber-800 dark:bg-amber-900/30 dark:text-amber-200 dark:hover:bg-amber-900/40"
                >
                  {canCancelQueued ? "Cancel queued" : "Stop generation"}
                </button>
              ) : null}
              <button
                type="button"
                onClick={handleLogout}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
              >
                Logout
              </button>
            </div>
          </div>
        </header>

        <main
          ref={chatContainerRef}
          onScroll={handleChatScroll}
          className="h-[calc(100vh-154px)] overflow-y-auto"
        >
          <div className="mx-auto flex min-h-full w-full max-w-4xl flex-col gap-4 px-4 pb-44 pt-6">

            {messages.length ? (
              messages.map((message) => (
                <MessageCard
                  key={message.id}
                  message={message}
                  onRegenerate={handleRegenerate}
                  onRegenerateInDeep={handleRegenerateInDeep}
                  onRetryWebOnly={handleRetryWebOnly}
                  onReaction={handleReaction}
                />
              ))
            ) : (
              <div className="mt-20 rounded-3xl border border-blue-200 bg-white/85 p-10 text-center shadow-soft dark:border-slate-700 dark:bg-slate-900/60">
                <h2 className="mb-2 text-2xl font-semibold">Start a new conversation</h2>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Ask anything about your data and get a streamed AI response in a clean ChatGPT-style workspace.
                </p>
              </div>
            )}
            <div ref={messageEndRef} />
          </div>
        </main>

        {showJumpToLatest && messages.length ? (
          <button
            type="button"
            onClick={() => {
              shouldAutoScrollRef.current = true;
              scrollToBottom("smooth", true);
            }}
            className="fixed bottom-28 right-4 z-30 rounded-full border border-blue-200 bg-white/95 px-3 py-1.5 text-xs font-medium text-brand-blue shadow-soft hover:bg-blue-50 dark:border-slate-700 dark:bg-slate-900/95 dark:text-blue-300 dark:hover:bg-slate-800 md:right-8"
          >
            Jump to latest
          </button>
        ) : null}

        <div className="fixed bottom-0 left-0 right-0 border-t border-blue-100 bg-white/75 backdrop-blur-md dark:border-slate-800 dark:bg-slate-950/80 md:left-[260px]">
          <ChatInput
            value={inputValue}
            disabled={isSending}
            mode={chatMode}
            suggestionChips={followUpChips}
            autoScrollPaused={autoScrollPaused}
            onChange={setInputValue}
            onModeChange={setChatMode}
            onSuggestionClick={async (chip) => {
              await handleSubmitFromComposer(chip);
            }}
            onToggleAutoScroll={() => {
              setAutoScrollPaused((prev) => {
                const next = !prev;
                if (next) {
                  shouldAutoScrollRef.current = false;
                  setShowJumpToLatest(true);
                } else {
                  shouldAutoScrollRef.current = true;
                  scrollToBottom("smooth", true);
                }
                return next;
              });
            }}
            onSubmit={async () => {
              await handleSubmitFromComposer(inputValue);
            }}
          />
        </div>
      </div>
    </div>
  );
}
