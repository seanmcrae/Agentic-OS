#if POCKETBRAINS_MLX
import Foundation
import MLXLLM
import MLXLMCommon

/// Optional bigger brain: a quantized open model via MLX Swift, for devices
/// without Apple Intelligence or users who want more headroom. Opt-in with
/// `-D POCKETBRAINS_MLX` plus the mlx-swift-examples package (see project.yml).
///
/// Model choice (mid-2026): Qwen3-4B-Instruct 4-bit — the best
/// quality/speed/size balance for iPhone-class hardware on MLX, which runs
/// 20–90% faster than llama.cpp on Apple Silicon. ~2.3GB download on first
/// use, cached locally; everything stays on device.
@MainActor
final class MLXBackend: ModelBackend {
    let displayName = "Qwen3 4B · MLX on-device"

    private let toolbox: ToolBox
    private let modelID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    private var container: ModelContainer?
    private var history: [(role: String, text: String)] = []

    init?(toolbox: ToolBox) {
        self.toolbox = toolbox
    }

    func resetConversation() { history.removeAll() }

    func reply(to prompt: String) -> AsyncThrowingStream<AgentEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    let container = try await loadContainerIfNeeded()
                    var transcript = prompt
                    var aggregated = ""

                    // Tool loop: generate → execute any tool call → feed
                    // results back → continue, max 4 hops.
                    for _ in 0..<4 {
                        let output = try await generate(container, userText: transcript,
                                                        continuation: continuation,
                                                        prefix: aggregated)
                        guard let call = Self.parseToolCall(output) else {
                            aggregated = output
                            break
                        }
                        let specs = AgentToolRegistry.all(toolbox: toolbox)
                        guard let spec = specs.first(where: { $0.name == call.name }) else {
                            aggregated = output
                            break
                        }
                        continuation.yield(.toolStarted(name: spec.name, summary: spec.description))
                        let result = spec.run(call.arguments)
                        continuation.yield(.toolFinished(result.record(toolName: spec.name)))
                        transcript = "<tool_response>\n\(result.detail)\n</tool_response>"
                    }

                    history.append(("user", prompt))
                    history.append(("assistant", aggregated))
                    continuation.yield(.done(finalText: aggregated))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func loadContainerIfNeeded() async throws -> ModelContainer {
        if let container { return container }
        let configuration = ModelConfiguration(id: modelID)
        let loaded = try await LLMModelFactory.shared.loadContainer(configuration: configuration)
        container = loaded
        return loaded
    }

    private func generate(_ container: ModelContainer, userText: String,
                          continuation: AsyncThrowingStream<AgentEvent, Error>.Continuation,
                          prefix: String) async throws -> String {
        let system = AgentVoice.instructions() + "\n\n" + Self.toolPrompt(toolbox: toolbox)
        var messages: [[String: String]] = [["role": "system", "content": system]]
        for turn in history.suffix(8) {
            messages.append(["role": turn.role, "content": turn.text])
        }
        messages.append(["role": "user", "content": userText])

        var collected = ""
        try await container.perform { modelContext in
            let input = try await modelContext.processor.prepare(
                input: .init(messages: messages))
            _ = try MLXLMCommon.generate(
                input: input, parameters: .init(temperature: 0.6),
                context: modelContext
            ) { tokens in
                let text = modelContext.tokenizer.decode(tokens: tokens)
                collected = text
                if !text.contains("<tool_call>") {
                    Task { @MainActor in
                        continuation.yield(.text(prefix + text))
                    }
                }
                return tokens.count > 1024 ? .stop : .more
            }
        }
        return collected
    }

    private static func toolPrompt(toolbox: ToolBox) -> String {
        let specs = AgentToolRegistry.all(toolbox: toolbox)
        let toolJSON = specs.map { spec in
            let params = spec.parameters
                .map { "\"\($0.name)\": \"\($0.description)\"" }
                .joined(separator: ", ")
            return "{\"name\": \"\(spec.name)\", \"description\": \"\(spec.description)\", \"parameters\": {\(params)}}"
        }.joined(separator: "\n")
        return """
        You can call tools. Available tools:
        \(toolJSON)
        To call one, reply with exactly:
        <tool_call>{"name": "toolName", "arguments": {"param": "value"}}</tool_call>
        After receiving <tool_response>, answer the user in plain prose.
        """
    }

    static func parseToolCall(_ text: String) -> (name: String, arguments: [String: String])? {
        guard let open = text.range(of: "<tool_call>"),
              let close = text.range(of: "</tool_call>"),
              open.upperBound < close.lowerBound
        else { return nil }
        let json = String(text[open.upperBound..<close.lowerBound])
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = object["name"] as? String
        else { return nil }
        let rawArgs = object["arguments"] as? [String: Any] ?? [:]
        let args = rawArgs.compactMapValues { "\($0)" }
        return (name, args)
    }
}
#endif
