'use client'

import { useState } from 'react'
import { ChevronDown, Plus, Settings, Users } from 'lucide-react'
import { CreateAgent } from './create-agent'

interface Agent {
  id: string
  name: string
  avatar?: string
  description: string
}

export function AgentSelector() {
  const [isOpen, setIsOpen] = useState(false)
  const [showCreateAgent, setShowCreateAgent] = useState(false)
  const [agents] = useState<Agent[]>([
    { id: '1', name: 'TAO', description: 'Task Automation Orchestrator' },
    { id: '2', name: 'Research Assistant', description: 'Research and analysis specialist' },
    { id: '3', name: 'Code Assistant', description: 'Programming and development helper' },
  ])

  return (
    <>
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full p-2 flex items-center justify-between hover:bg-gray-800 rounded-lg transition-colors"
        >
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-500 rounded-full" />
            <div className="text-left">
              <div className="text-white font-medium">TAO</div>
              <div className="text-xs text-gray-400">Task Automation Orchestrator</div>
            </div>
          </div>
          <ChevronDown className={`text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} size={20} />
        </button>

        {isOpen && (
          <div className="absolute top-full left-0 w-full mt-2 bg-gray-900 rounded-lg border border-gray-800 shadow-lg overflow-hidden">
            <div className="p-2">
              {agents.map((agent) => (
                <button
                  key={agent.id}
                  className="w-full flex items-center gap-2 p-2 hover:bg-gray-800 rounded-lg"
                >
                  <div className="w-8 h-8 bg-blue-500 rounded-full" />
                  <div className="text-left">
                    <div className="text-white font-medium">{agent.name}</div>
                    <div className="text-xs text-gray-400">{agent.description}</div>
                  </div>
                </button>
              ))}
            </div>
            
            <div className="border-t border-gray-800 p-2">
              <button
                onClick={() => {
                  setIsOpen(false)
                  setShowCreateAgent(true)
                }}
                className="w-full flex items-center gap-2 p-2 text-blue-400 hover:bg-gray-800 rounded-lg"
              >
                <Plus size={20} />
                <span>Create New Agent</span>
              </button>
              <button
                className="w-full flex items-center gap-2 p-2 text-gray-400 hover:bg-gray-800 rounded-lg"
              >
                <Settings size={20} />
                <span>Manage Agents</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {showCreateAgent && (
        <CreateAgent onClose={() => setShowCreateAgent(false)} />
      )}
    </>
  )
} 