'use client'

import { useState } from 'react'
import { LLMProvider, LLMSettings } from '@/types/llm'

const DEFAULT_SETTINGS: LLMSettings = {
  providers: {
    openai: {
      provider: 'openai',
      apiKey: '',
      enabled: false,
    },
    anthropic: {
      provider: 'anthropic',
      apiKey: '',
      enabled: false,
    },
    gemini: {
      provider: 'gemini',
      apiKey: '',
      enabled: false,
    },
    groq: {
      provider: 'groq',
      apiKey: '',
      enabled: false,
    },
    openrouter: {
      provider: 'openrouter',
      apiKey: '',
      enabled: false,
    }
  },
  defaultProvider: 'anthropic'
}

export function LLMSettings() {
  const [settings, setSettings] = useState<LLMSettings>(() => {
    // Load from localStorage if available
    const saved = localStorage.getItem('llm-settings')
    return saved ? JSON.parse(saved) : DEFAULT_SETTINGS
  })

  const updateProviderSettings = (provider: LLMProvider, apiKey: string, enabled: boolean) => {
    const newSettings = {
      ...settings,
      providers: {
        ...settings.providers,
        [provider]: {
          ...settings.providers[provider],
          apiKey,
          enabled
        }
      }
    }
    setSettings(newSettings)
    localStorage.setItem('llm-settings', JSON.stringify(newSettings))
  }

  return (
    <div className="space-y-6 p-4">
      <h2 className="text-2xl font-bold mb-4">LLM Settings</h2>
      
      {/* OpenAI */}
      <div className="space-y-2">
        <label className="text-lg font-medium">OpenAI API Key</label>
        <div className="flex gap-2">
          <input
            type="password"
            placeholder="Enter OpenAI API key"
            className="flex-1 rounded-lg border p-2 dark:bg-gray-800 dark:border-gray-700"
            value={settings.providers.openai.apiKey}
            onChange={(e) => updateProviderSettings('openai', e.target.value, settings.providers.openai.enabled)}
          />
          <button
            onClick={() => updateProviderSettings('openai', settings.providers.openai.apiKey, !settings.providers.openai.enabled)}
            className={`px-4 py-2 rounded-lg ${
              settings.providers.openai.enabled ? 'bg-green-500' : 'bg-gray-500'
            } text-white`}
          >
            {settings.providers.openai.enabled ? 'Enabled' : 'Disabled'}
          </button>
        </div>
        <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-blue-500 text-sm">
          Get OpenAI API Key
        </a>
      </div>

      {/* Anthropic */}
      <div className="space-y-2">
        <label className="text-lg font-medium">Anthropic API Key</label>
        <div className="flex gap-2">
          <input
            type="password"
            placeholder="Enter Anthropic API key"
            className="flex-1 rounded-lg border p-2 dark:bg-gray-800 dark:border-gray-700"
            value={settings.providers.anthropic.apiKey}
            onChange={(e) => updateProviderSettings('anthropic', e.target.value, settings.providers.anthropic.enabled)}
          />
          <button
            onClick={() => updateProviderSettings('anthropic', settings.providers.anthropic.apiKey, !settings.providers.anthropic.enabled)}
            className={`px-4 py-2 rounded-lg ${
              settings.providers.anthropic.enabled ? 'bg-green-500' : 'bg-gray-500'
            } text-white`}
          >
            {settings.providers.anthropic.enabled ? 'Enabled' : 'Disabled'}
          </button>
        </div>
        <a href="https://console.anthropic.com/account/keys" target="_blank" rel="noopener noreferrer" className="text-blue-500 text-sm">
          Get Anthropic API Key
        </a>
      </div>

      {/* Add similar sections for Gemini, Groq, and OpenRouter */}
    </div>
  )
} 