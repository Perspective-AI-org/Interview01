// Typed client for the localrag FastAPI backend.

export interface Citation {
  marker: number;
  source: string;
  chunk_index: number;
  score: number;
  snippet: string;
}

export interface AskResponse {
  answer: string;
  mode: string;
  model: string;
  citations: Citation[];
}

export interface IngestResponse {
  source: string;
  n_chunks: number;
  document_id: number;
}

export interface SourceInfo {
  source: string;
  n_chunks: number;
  created_at?: string | null;
}

export interface Stats {
  embedder: string;
  embed_dim: number | null;
  n_chunks: number;
  ollama_available: boolean;
  sources: SourceInfo[];
}

// Allow overriding the API base (e.g. when the SPA is hosted separately).
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore parse errors */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function getStats(): Promise<Stats> {
  return handle<Stats>(await fetch(`${BASE}/api/stats`));
}

export async function ask(
  question: string,
  topK?: number,
  source?: string,
): Promise<AskResponse> {
  const res = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, source: source ?? null }),
  });
  return handle<AskResponse>(res);
}

export async function ingestText(
  text: string,
  source: string,
): Promise<IngestResponse> {
  const res = await fetch(`${BASE}/api/ingest/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  });
  return handle<IngestResponse>(res);
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/ingest/file`, {
    method: "POST",
    body: form,
  });
  return handle<IngestResponse>(res);
}

export async function deleteSource(name: string): Promise<void> {
  const res = await fetch(`${BASE}/api/sources/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  await handle<unknown>(res);
}
