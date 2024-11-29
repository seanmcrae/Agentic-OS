# Agentic OS

An advanced AI agent operating system with multi-agent orchestration capabilities.

## Overview

Agentic OS is a next-generation web application that provides an intelligent interface for interacting with multiple AI agents. Similar to Perplexity but with advanced agent orchestration capabilities.

## Features

- Multi-agent orchestration
- Real-time chat interface
- Source attribution
- Conversation branching
- Advanced search capabilities

## Technology Stack

- Frontend: Next.js 14, TypeScript, Tailwind CSS
- Backend: Node.js, Express
- Database: PostgreSQL, Prisma
- Cache: Redis
- LLM Integration: OpenAI GPT-4

## Project Structure

```
agentic-os/
├── src/
│   ├── app/                 # Next.js 14 app directory
│   ├── components/          # React components
│   ├── lib/                 # Utility functions and shared logic
│   ├── styles/             # Global styles and Tailwind config
│   └── types/              # TypeScript type definitions
├── prisma/                 # Database schema and migrations
├── public/                 # Static assets
└── tests/                 # Test files
```

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/seanmcrae/Agentic-OS.git
cd Agentic-OS
```

2. Install dependencies
```bash
npm install
```

3. Set up environment variables
```bash
cp .env.example .env
```

4. Start the development server
```bash
npm run dev
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

MIT