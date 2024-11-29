interface MessageProps {
  content: string
  role: 'user' | 'assistant'
  timestamp?: Date
}

export function Message({ content, role, timestamp }: MessageProps) {
  return (
    <div className={`flex ${role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] rounded-lg p-4 ${
        role === 'user' 
          ? 'bg-blue-500 text-white' 
          : 'bg-gray-100 dark:bg-gray-800'
      }`}>
        <p className="text-sm">{content}</p>
        {timestamp && (
          <span className="text-xs opacity-50">
            {timestamp.toLocaleTimeString()}
          </span>
        )}
      </div>
    </div>
  )
} 