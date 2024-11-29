'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Home, 
  Settings, 
  Users, 
  Tool, 
  Command, 
  Share2,
  PlusCircle,
} from 'lucide-react'
import { AgentSelector } from '@/components/agents/agent-selector'

interface SidebarItem {
  icon: React.ReactNode
  label: string
  href: string
}

const MENU_ITEMS: SidebarItem[] = [
  { icon: <Home size={20} />, label: 'Home', href: '/' },
  { icon: <Users size={20} />, label: 'Agents', href: '/agents' },
  { icon: <Tool size={20} />, label: 'Tools', href: '/tools' },
  { icon: <Command size={20} />, label: 'Commands', href: '/commands' },
  { icon: <Share2 size={20} />, label: 'Share & Embed', href: '/share' },
]

export function Sidebar() {
  return (
    <div className="w-64 bg-gray-900 h-screen flex flex-col">
      {/* Agent Selector */}
      <div className="p-4 border-b border-gray-800">
        <AgentSelector />
      </div>

      {/* Main Menu */}
      <nav className="flex-1 p-4 space-y-2">
        {MENU_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="flex items-center gap-3 text-gray-400 hover:text-white px-2 py-2 rounded-lg hover:bg-gray-800"
          >
            {item.icon}
            <span>{item.label}</span>
          </Link>
        ))}
      </nav>

      {/* Tools Quick Access */}
      <div className="p-4 border-t border-gray-800">
        <div className="text-sm text-gray-400 mb-2">Tools</div>
        <div className="grid grid-cols-4 gap-2">
          <button className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700">
            <Tool size={20} className="text-gray-400" />
          </button>
          {/* Add more tool shortcuts */}
        </div>
      </div>
    </div>
  )
} 