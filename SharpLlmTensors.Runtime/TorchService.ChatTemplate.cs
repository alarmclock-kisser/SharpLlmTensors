using System.Text.Json;
using SharpLlmTensors.Shared;

namespace SharpLlmTensors.Runtime
{
    public enum ChatTemplateType { ChatML, Llama3, Gemma, Fallback }

    public partial class TorchService
    {
        private ChatTemplateType _currentChatTemplate = ChatTemplateType.Fallback;

        /// <summary>
        /// Liest die tokenizer_config.json und analysiert das Jinja-Template.
        /// </summary>
        public async Task InitializeChatTemplateAsync(string configFilePath)
        {
            if (!File.Exists(configFilePath))
            {
                await StaticLogger.LogAsync("[TorchService] No tokenizer_config.json found. Using fallback template.");
                return;
            }

            try
            {
                var jsonText = await File.ReadAllTextAsync(configFilePath);
                var config = JsonSerializer.Deserialize<JsonElement>(jsonText);

                if (config.TryGetProperty("chat_template", out var templateProp))
                {
                    string templateString = templateProp.GetString() ?? "";

                    // Dynamische Mustererkennung des Jinja-Templates
                    if (templateString.Contains("<|im_start|>"))
                    {
                        this._currentChatTemplate = ChatTemplateType.ChatML; // Qwen, Yi, Smaug
                        await StaticLogger.LogAsync("[TorchService] Chat template detected: ChatML (Qwen Format)");
                    }
                    else if (templateString.Contains("<|start_header_id|>"))
                    {
                        this._currentChatTemplate = ChatTemplateType.Llama3; // Llama 3
                        await StaticLogger.LogAsync("[TorchService] Chat template detected: Llama 3");
                    }
                    else if (templateString.Contains("<start_of_turn>"))
                    {
                        this._currentChatTemplate = ChatTemplateType.Gemma; // Gemma
                        await StaticLogger.LogAsync("[TorchService] Chat template detected: Gemma");
                    }
                    else
                    {
                        this._currentChatTemplate = ChatTemplateType.Fallback;
                        await StaticLogger.LogAsync("[TorchService] Chat template unknown. Using fallback.");
                    }
                }
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"[TorchService] Error parsing chat template: {ex.Message}");
            }
        }

        /// <summary>
        /// Formatiert den rohen User-Text in das korrekte Format für das geladene Modell.
        /// </summary>
        public string ApplyChatTemplate(string userMessage, string systemMessage = "You are a helpful assistant.")
        {
            return this._currentChatTemplate switch
            {
                ChatTemplateType.ChatML =>
                    $"<|im_start|>system\n{systemMessage}<|im_end|>\n<|im_start|>user\n{userMessage}<|im_end|>\n<|im_start|>assistant\n",

                ChatTemplateType.Llama3 =>
                    $"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{systemMessage}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{userMessage}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",

                ChatTemplateType.Gemma =>
                    $"<bos><start_of_turn>user\n{userMessage}<end_of_turn>\n<start_of_turn>model\n",

                _ =>
                    $"{systemMessage}\n\nUser: {userMessage}\nAssistant: "
            };
        }
    }
}