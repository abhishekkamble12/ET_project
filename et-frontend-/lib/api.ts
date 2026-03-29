// API client for EngageTech backend
// Base URL configurable via NEXT_PUBLIC_API_URL or defaults to localhost:8000

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const DEFAULT_COMPANY_ID = 'demo-company';

function getHeaders(contentType: string = 'application/json'): HeadersInit {
  const companyId = process.env.NEXT_PUBLIC_COMPANY_ID || DEFAULT_COMPANY_ID;
  const headers: Record<string, string> = {
    'X-Company-ID': companyId,
  };
  if (contentType) {
    headers['Content-Type'] = contentType;
  }
  return headers;
}

export interface GenerateRequest {
  query: string;
  platform: 'linkedin' | 'instagram' | 'twitter';
  tone?: string;
  locale?: string;
  session_id?: string;
}

export interface ApproveRequest {
  session_id: string;
  decision: 'publish' | 'edit' | 'reject';
  edits?: string;
}

export interface StreamEvent {
  eventType: string;
  data: any;
}

export async function generateContent(req: GenerateRequest) {
  const res = await fetch(`${API_BASE}/api/v1/generate`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function* streamContent(req: GenerateRequest): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API_BASE}/api/v1/stream`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  if (!res.body) throw new Error('No stream body');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const event of events) {
      if (!event.trim()) continue;
      const lines = event.split('\n');
      let eventType = 'message';
      let dataStr = '';
      for (const line of lines) {
        if (line.startsWith('event:')) eventType = line.slice(6).trim();
        if (line.startsWith('data:')) dataStr += line.slice(5).trim();
      }
      if (dataStr) {
        try {
          const data = JSON.parse(dataStr);
          yield { eventType, data };
        } catch {}
      }
    }
  }
}

export async function approveContent(req: ApproveRequest) {
  const res = await fetch(`${API_BASE}/api/v1/approve`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Enterprise APIs
export async function uploadKnowledge(file: File, company_id: string = process.env.NEXT_PUBLIC_COMPANY_ID || DEFAULT_COMPANY_ID, department?: string) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('company_id', company_id);
  if (department) {
    formData.append('department', department);
  }
  
  const res = await fetch(`${API_BASE}/api/v1/enterprise/data/upload`, {
    method: 'POST',
    headers: {
      'X-Company-ID': company_id,
    },
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listKnowledge() {
  const res = await fetch(`${API_BASE}/api/v1/enterprise/data/list`, {
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteKnowledge(doc_id: string) {
  const res = await fetch(`${API_BASE}/api/v1/enterprise/data/${doc_id}`, { 
    method: 'DELETE',
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


