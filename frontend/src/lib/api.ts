import type {
  ChatExecutionMode,
  ConversationItem,
  LoginResponse,
  StreamEvent,
  TraceEventItem,
} from "../types";

function resolveApiBase(): string {
  if (import.meta.env.PROD) {
    return "";
  }

  const configured = String(import.meta.env.VITE_API_BASE ?? import.meta.env.VITE_API_URL ?? "").trim();
  if (!configured) {
    return "";
  }

  const normalized = configured.replace(/\/+$/, "");
  return normalized;
}

const API_BASE = resolveApiBase();

function resolveChatPathCandidates(path: string): string[] {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  const configuredPrefix = String(import.meta.env.VITE_CHAT_API_PREFIX ?? "").trim().replace(/\/+$/, "");
  if (configuredPrefix) {
    return [`${configuredPrefix}${normalized}`];
  }
  return [`/api/v1${normalized}`, normalized];
}

async function fetchWithPathFallback(
  paths: string[],
  init: RequestInit,
  retryStatuses = new Set([404, 405])
): Promise<Response> {
  let lastResponse: Response | null = null;
  let lastError: unknown = null;

  for (const [index, path] of paths.entries()) {
    try {
      const response = await fetch(`${API_BASE}${path}`, init);
      if (response.ok) {
        return response;
      }
      if (index < paths.length - 1 && retryStatuses.has(response.status)) {
        lastResponse = response;
        continue;
      }
      throw new Error(await parseError(response));
    } catch (error) {
      lastError = error;
      if (index < paths.length - 1) {
        continue;
      }
    }
  }

  if (lastResponse) {
    throw new Error(await parseError(lastResponse));
  }
  if (lastError instanceof Error) {
    throw lastError;
  }
  throw new Error("Request failed.");
}

async function parseError(response: Response): Promise<string> {
  const fallback = `Request failed (${response.status})`;
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function loginWithPassword(username: string, password: string): Promise<LoginResponse> {
  const response = await fetch(`${API_BASE}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as LoginResponse;
}

export async function fetchConversations(
  userId: string,
  token: string,
  limit = 60
): Promise<ConversationItem[]> {
  const response = await fetch(
    `${API_BASE}/api/v1/eval/conversations?user_id=${encodeURIComponent(userId)}&limit=${limit}`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    }
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  const payload = (await response.json()) as {
    conversations?: Array<{
      conversation_id?: string;
      prompt?: string;
      answer?: string;
      created_at?: string;
    }>;
  };

  return (payload.conversations ?? []).map((item) => {
    const prompt = String(item.prompt ?? "").trim();
    const answer = String(item.answer ?? "").trim();
    const title = prompt.length > 56 ? `${prompt.slice(0, 56)}...` : prompt || "Untitled chat";

    return {
      conversationId: String(item.conversation_id ?? ""),
      title,
      prompt,
      answer,
      createdAt: String(item.created_at ?? ""),
    };
  });
}

export async function clearChatHistory(
  userId: string,
  token: string,
  sessionId?: string
): Promise<void> {
  const query = new URLSearchParams({ user_id: userId });
  const cleanSessionId = String(sessionId ?? "").trim();
  if (cleanSessionId) {
    query.set("session_id", cleanSessionId);
  }

  const response = await fetch(`${API_BASE}/api/v1/chat/history?${query.toString()}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }
}

function extractText(payload: Record<string, unknown>): string {
  const directKeys = ["text", "delta", "token", "content", "message"];
  for (const key of directKeys) {
    const value = payload[key];
    if (typeof value === "string") {
      return value;
    }
  }

  const choices = payload.choices;
  if (Array.isArray(choices)) {
    const first = choices[0];
    if (first && typeof first === "object") {
      const delta = (first as Record<string, unknown>).delta;
      if (delta && typeof delta === "object") {
        const content = (delta as Record<string, unknown>).content;
        if (typeof content === "string") {
          return content;
        }
      }
    }
  }

  return "";
}

function extractUrls(input: string): string[] {
  const matches = input.match(/https?:\/\/[^\s)"']+/gi) ?? [];
  const normalized = matches
    .map((item) => normalizeUrlCandidate(item))
    .filter((item): item is string => Boolean(item));
  return Array.from(new Set(normalized));
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

function collectUrlCandidates(value: unknown, out: Set<string>): void {
  if (typeof value === "string") {
    for (const url of extractUrls(value)) {
      out.add(url);
    }
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      collectUrlCandidates(item, out);
    }
    return;
  }
  if (value && typeof value === "object") {
    for (const nested of Object.values(value as Record<string, unknown>)) {
      collectUrlCandidates(nested, out);
    }
  }
}

function extractWebsites(payload: Record<string, unknown>): string[] {
  const urls = new Set<string>();
  const sourceKeys = ["websites", "sources", "source_urls", "urls", "search_results", "citations"];
  for (const key of sourceKeys) {
    collectUrlCandidates(payload[key], urls);
  }
  collectUrlCandidates(payload.status, urls);
  collectUrlCandidates(payload.message, urls);
  return Array.from(urls);
}

function extractReasoningSteps(payload: Record<string, unknown>): string[] {
  const steps = new Set<string>();
  const candidateKeys = ["steps", "reasoning", "reasoning_summary", "progress", "phase", "status"];
  for (const key of candidateKeys) {
    const value = payload[key];
    if (typeof value === "string") {
      const clean = value.trim();
      if (clean && !/^https?:\/\//i.test(clean)) {
        steps.add(clean);
      }
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === "string") {
          const clean = item.trim();
          if (clean && !/^https?:\/\//i.test(clean)) {
            steps.add(clean);
          }
        }
      }
    }
  }
  return Array.from(steps);
}

function normalizeTraceEvent(value: unknown): TraceEventItem | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const rawType = String(record.type ?? "").trim();
  if (!rawType) {
    return null;
  }
  const timestamp = typeof record.timestamp === "string" ? record.timestamp : undefined;
  const payload =
    record.payload && typeof record.payload === "object"
      ? (record.payload as Record<string, unknown>)
      : undefined;
  return { type: rawType, timestamp, payload };
}

function normalizeStreamEvent(payload: Record<string, unknown>): StreamEvent | null {
  const rawType = String(payload.type ?? "").toLowerCase();
  const websites = extractWebsites(payload);
  const steps = extractReasoningSteps(payload);
  if (rawType === "error") {
    return {
      type: "error",
      detail:
        typeof payload.detail === "string"
          ? payload.detail
          : typeof payload.error === "string"
            ? payload.error
            : "Streaming request failed.",
    };
  }
  if (rawType === "done" || rawType === "complete" || rawType === "completed") {
    return { type: "done" };
  }
  if (rawType === "queued") {
    return {
      type: "queued",
      status: typeof payload.status === "string" ? payload.status : undefined,
      job_id: typeof payload.job_id === "string" ? payload.job_id : undefined,
      websites,
      steps,
    };
  }
  if (rawType === "status") {
    return {
      type: "status",
      status: typeof payload.status === "string" ? payload.status : undefined,
      job_id: typeof payload.job_id === "string" ? payload.job_id : undefined,
      websites,
      steps,
    };
  }
  if (rawType === "search" || rawType === "sources" || rawType === "source") {
    return {
      type: "search",
      status: typeof payload.status === "string" ? payload.status : undefined,
      websites,
    };
  }
  if (rawType === "reasoning" || rawType === "trace" || rawType === "progress") {
    const trace = normalizeTraceEvent(payload.event);
    if (trace) {
      const traceWebsites = trace.payload ? extractWebsites(trace.payload) : websites;
      return {
        type: "trace",
        job_id: typeof payload.job_id === "string" ? payload.job_id : undefined,
        trace,
        websites: traceWebsites,
      };
    }
    return {
      type: "reasoning",
      status: typeof payload.status === "string" ? payload.status : undefined,
      steps,
    };
  }
  if (rawType === "chunk" || rawType === "token" || rawType === "delta") {
    return { type: "chunk", text: extractText(payload) };
  }

  if (websites.length) {
    return {
      type: "search",
      status: typeof payload.status === "string" ? payload.status : undefined,
      websites,
    };
  }
  if (steps.length) {
    return {
      type: "reasoning",
      status: typeof payload.status === "string" ? payload.status : undefined,
      steps,
    };
  }

  const inferredText = extractText(payload);
  if (inferredText) {
    return { type: "chunk", text: inferredText };
  }
  return null;
}

function parseSseBlock(block: string): StreamEvent | null {
  const lines = block.split("\n");
  const dataLines: string[] = [];
  let eventName = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim().toLowerCase();
      continue;
    }
    if (line.startsWith("data:")) {
      let payload = line.slice(5);
      if (payload.startsWith(" ")) {
        payload = payload.slice(1);
      }
      dataLines.push(payload);
    }
  }

  if (!dataLines.length) {
    return null;
  }

  const data = dataLines.join("\n");
  if (!data) {
    return null;
  }
  if (data === "[DONE]") {
    return { type: "done" };
  }

  try {
    const parsed = JSON.parse(data) as Record<string, unknown>;
    const normalized = normalizeStreamEvent(parsed);
    if (normalized) {
      return normalized;
    }
  } catch {
    // non-JSON SSE payload; treat as plain streamed text
  }

  if (eventName === "error") {
    return { type: "error", detail: data };
  }
  return { type: "chunk", text: data };
}

export async function streamChatResponse(
  token: string,
  payload: { userId: string; sessionId: string; prompt: string; mode?: ChatExecutionMode },
  onEvent: (event: StreamEvent) => void,
  options?: { signal?: AbortSignal }
): Promise<void> {
  const response = await fetchWithPathFallback(resolveChatPathCandidates("/chat/stream"), {
    method: "POST",
    signal: options?.signal,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: payload.userId,
      session_id: payload.sessionId,
      prompt: payload.prompt,
      mode: payload.mode ?? "auto",
    }),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  if (!response.body) {
    throw new Error("No stream body received.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const contentType = String(response.headers.get("content-type") ?? "").toLowerCase();
  const serverSentEvents = contentType.includes("text/event-stream");
  let buffer = "";
  let streamMode: "unknown" | "sse" | "text" = serverSentEvents ? "sse" : "unknown";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    const chunk = decoder.decode(value, { stream: true });
    if (!chunk) {
      continue;
    }

    if (streamMode === "unknown") {
      if (chunk.includes("data:") || chunk.includes("\n\n")) {
        streamMode = "sse";
      } else {
        streamMode = "text";
      }
    }

    if (streamMode === "text") {
      onEvent({ type: "chunk", text: chunk });
      continue;
    }

    buffer += chunk.replace(/\r\n/g, "\n");
    while (buffer.includes("\n\n")) {
      const boundary = buffer.indexOf("\n\n");
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (!block.trim()) {
        continue;
      }
      const event = parseSseBlock(block);
      if (event != null) {
        onEvent(event);
      }
    }
  }

  const trailing = decoder.decode();
  if (trailing) {
    if (streamMode === "text") {
      onEvent({ type: "chunk", text: trailing });
    } else {
      buffer += trailing;
    }
  }

  const remainder = buffer.trim();
  if (remainder) {
    const tailEvent = parseSseBlock(remainder);
    if (tailEvent) {
      onEvent(tailEvent);
    } else if (streamMode !== "sse") {
      onEvent({ type: "chunk", text: remainder });
    }
  }
}

export async function fetchChatJobTrace(
  token: string,
  jobId: string
): Promise<{ jobId: string; traceEvents: TraceEventItem[]; websites: string[] }> {
  const response = await fetchWithPathFallback(
    resolveChatPathCandidates(`/chat/${encodeURIComponent(jobId)}`),
    {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    },
    new Set([404])
  );

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  const payload = (await response.json()) as {
    trace_events?: unknown[];
  };

  const traceEvents = (payload.trace_events ?? [])
    .map((item) => normalizeTraceEvent(item))
    .filter((item): item is TraceEventItem => item !== null);

  const websites = new Set<string>();
  for (const event of traceEvents) {
    if (event.payload) {
      for (const url of extractWebsites(event.payload)) {
        websites.add(url);
      }
    }
  }

  return {
    jobId,
    traceEvents,
    websites: Array.from(websites),
  };
}
