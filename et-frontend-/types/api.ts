export interface GeneratedPost {
  caption?: string;
  image_prompt?: string;
  hashtags: string[];
  platform: string;
  locale?: string;
}

export type Platform = 'linkedin' | 'instagram' | 'twitter';

export interface GenerateRequest {
  query: string;
  platform: Platform;
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

