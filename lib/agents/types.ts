export interface Agent {
  id: string;
  name: string;
  role: AgentRole;
  capabilities: AgentCapability[];
  model: LLMModel;
  status: AgentStatus;
  metadata: Record<string, any>;
}

export enum AgentRole {
  RESEARCHER = 'researcher',
  ANALYST = 'analyst',
  CRITIC = 'critic',
  ORCHESTRATOR = 'orchestrator',
  WRITER = 'writer'
}

export enum AgentCapability {
  WEB_SEARCH = 'web_search',
  CODE_ANALYSIS = 'code_analysis',
  TEXT_ANALYSIS = 'text_analysis',
  MATH = 'math',
  REASONING = 'reasoning'
}

export enum AgentStatus {
  IDLE = 'idle',
  BUSY = 'busy',
  ERROR = 'error'
}

export enum LLMModel {
  GPT4 = 'gpt-4',
  CLAUDE = 'claude-3-opus-20240229',
  GEMINI = 'gemini-pro'
}

export interface AgentMessage {
  id: string;
  agentId: string;
  content: string;
  type: MessageType;
  timestamp: Date;
  metadata: Record<string, any>;
  citations?: Citation[];
}

export enum MessageType {
  TEXT = 'text',
  CODE = 'code',
  ERROR = 'error',
  SYSTEM = 'system',
  SEARCH_RESULT = 'search_result'
}

export interface Citation {
  url: string;
  title: string;
  snippet: string;
  timestamp: Date;
}