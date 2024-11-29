'use client'

import { useState } from 'react'
import { X } from 'lucide-react'

interface CreateAgentProps {
  onClose: () => void
}

export function CreateAgent({ onClose }: CreateAgentProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [persona, setPersona] = useState('')

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center">
      <div className="bg-gray-900 rounded-lg w-full max-w-2xl p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">Create New Agent</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X size={24} />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">
              Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2 text-white"
              placeholder="Enter agent name"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2 text-white h-24"
              placeholder="Describe your agent's purpose and capabilities"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-200 mb-1">
              Persona
            </label>
            <textarea
              value={persona}
              onChange={(e) => setPersona(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2 text-white h-24"
              placeholder="Define your agent's personality and tone"
            />
          </div>
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-gray-800 text-white hover:bg-gray-700"
          >
            Cancel
          </button>
          <button
            className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700"
          >
            Create Agent
          </button>
        </div>
      </div>
    </div>
  )
} 