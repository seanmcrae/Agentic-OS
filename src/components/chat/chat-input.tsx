'use client'

import { useState, FormEvent } from 'react'

interface ChatInputProps {
  onSubmit: (message: string) => void
  disabled?: boolean
}

export function ChatInput({ onSubmit, disabled }: ChatInputProps) {
  const [input, setInput] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return
    
    onSubmit(input)
    setInput('')
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type a message..."
        className="flex-1 rounded-lg border p-2 dark:bg-gray-800 dark:border-gray-700"
        disabled={disabled}
      />
      <button
        type="submit"
        disabled={disabled || !input.trim()}
        className="bg-blue-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
      >
        Send
      </button>
    </form>
  )
} 