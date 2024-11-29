'use client'

import { useState } from 'react'
import { Message } from './message'
import { ChatInput } from './chat-input'

interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
}

export function ChatContainer() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (content: string) => {
    // Create user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content,
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    // TODO: Add API call here
    // For now, just simulate a response
    setTimeout(() => {
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: 'This is a simulated response.',
        role: 'assistant',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, assistantMessage])
      setIsLoading(false)
    }, 1000)
  }

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((message) => (
          <Message key={message.id} {...message} />
        ))}
      </div>
      <div className="border-t p-4">
        <ChatInput onSubmit={handleSubmit} disabled={isLoading} />
      </div>
    </div>
  )
} 