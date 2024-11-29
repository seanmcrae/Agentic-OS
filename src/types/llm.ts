export type LLMProvider = 'openai' | 'anthropic' | 'gemini' | 'groq' | 'openrouter'

export interface LLMConfig {
  provider: LLMProvider
  apiKey: string
  enabled: boolean
  models?: string[]
}

export interface LLMSettings {
  providers: Record<LLMProvider, LLMConfig>
  defaultProvider: LLMProvider
} 