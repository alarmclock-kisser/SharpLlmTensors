using SharpLlmTensors.Shared;
using System.Runtime.CompilerServices;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace SharpLlmTensors.Runtime
{
    public partial class TorchService
    {
        public GenerationStats? LastGenerationStats { get; private set; } = null;

        /// <summary>
        /// Generiert Text basierend auf einem Prompt und gibt das komplette Ergebnis am Ende zurück.
        /// </summary>
        public async Task<string> GenerateTextAsync(string prompt, int maxNewTokens = 64, CancellationToken ct = default)
        {
            StringBuilder sb = new StringBuilder();
            await foreach (var token in GenerateTextStreamAsync(prompt, maxNewTokens, false, ct))
            {
                sb.Append(token);
            }
            return sb.ToString();
        }

        /// <summary>
        /// Generiert einen Token-Stream für Echtzeit-Ausgabe.
        /// </summary>
        public async unsafe IAsyncEnumerable<string> GenerateTextStreamAsync(
            string prompt,
            int maxNewTokens = 128,
            bool logTokens = true,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            if (this._activeModel is null)
            {
                throw new InvalidOperationException("Model not loaded.");
            }

            if (!this.IsTokenizerLoaded)
            {
                throw new InvalidOperationException("Tokenizer not loaded.");
            }

            LogVerbose($"[Inference] Starting generation. Max tokens: {maxNewTokens}");

            // 1. Prompt-Formatierung (Chat-Template)
            string formattedPrompt = this.ApplyChatTemplate(prompt);
            LogVerbose($"[Inference] Formatted Prompt: {formattedPrompt.Replace("\n", "\\n")}");

            // 2. Tokenisierung
            var inputTokenIds = this.Tokenize(formattedPrompt).ToList();
            LogVerbose($"[Inference] Input tokens: {string.Join(", ", inputTokenIds)} (Count: {inputTokenIds.Count})");

            this.LastGenerationStats = new GenerationStats { GenerationStarted = DateTime.UtcNow };

            using var no_grad = torch.no_grad();
            int startSeqLen = inputTokenIds.Count;

            for (int step = 0; step < maxNewTokens; step++)
            {
                if (ct.IsCancellationRequested)
                {
                    break;
                }

                // 3. Tensor-Vorbereitung
                using var inputTensor = torch.tensor(inputTokenIds.ToArray(), dtype: ScalarType.Int64, device: this._currentDevice).unsqueeze(0);
                LogVerbose($"[Inference] Step {step}: Input tensor shape {string.Join(',', inputTensor.shape)}");

                Tensor logits;

                // 4. GENERISCHER FORWARD PASS
                // Wir casten auf die Basisklasse, die alle unsere Text-Modelle (Llama, Gemma, Qwen) nutzen.
                if (this._activeModel is nn.Module<Tensor, Tensor> model)
                {
                    logits = model.forward(inputTensor);
                }
                else
                {
                    // Fallback für VL-Modelle oder spezielle Architekturen
                    LogVerbose($"[Inference] Warning: Model is not a standard Module<Tensor, Tensor>. Trying dynamic forward.");
                    using var _ = logits = (Tensor) this._activeModel.GetType().GetMethod("forward")?.Invoke(this._activeModel, new object[] { inputTensor })!;
                }

                if (logits is null)
                {
                    throw new Exception("Model forward pass failed to produce logits.");
                }

                // 5. Sampling (Wir nehmen den letzten Token der Sequenz)
                long lastTokenIdx = logits.shape[1] - 1;
                using var lastTokenLogits = logits.select(1, lastTokenIdx);

                // Greedy Search: Einfach das wahrscheinlichste Token nehmen
                using var nextTokenTensor = lastTokenLogits.argmax(-1);
                int nextTokenId = (int) nextTokenTensor.item<long>();

                // Cleanup Logits
                logits.Dispose();

                // 6. Detokenisierung
                string currentWord = this.Detokenize(new long[] { nextTokenId });

                // 7. Beendigungs-Logik (EOS Tokens)
                // 128009 = Llama3 <|eot_id|>, 151643 = Qwen <|im_end|>, 1 = Gemma <eos>
                if (nextTokenId == 128009 || nextTokenId == 151643 || nextTokenId == 1 || nextTokenId == 107)
                {
                    LogVerbose($"[Inference] EOS detected (ID: {nextTokenId}). Stopping.");
                    break;
                }

                inputTokenIds.Add(nextTokenId);
                this.LastGenerationStats.TotalTokensGenerated++;

                if (logTokens)
                {
                    await StaticLogger.LogAsync($"[TorchService] Step {step} -> Token: '{currentWord.Replace("\n", "\\n")}' (ID: {nextTokenId})");
                }

                yield return currentWord;

                // 8. Kontext-Fenster Schutz (Optional)
                if (inputTokenIds.Count > 4096)
                {
                    LogVerbose("[Inference] Max context reached. Truncating.");
                    inputTokenIds.RemoveAt(0);
                }
            }

            // 9. Statistik-Abschluss
            this.LastGenerationStats.GenerationFinished = DateTime.UtcNow;
            await LogGenerationSummary();
        }

        private async Task LogGenerationSummary()
        {
            if (this.LastGenerationStats is null)
            {
                return;
            }

            await StaticLogger.LogAsync($"[TorchService] Generation finished.");
            await StaticLogger.LogAsync($"[TorchService] Total tokens: {this.LastGenerationStats.TotalTokensGenerated}");
            await StaticLogger.LogAsync($"[TorchService] Tokens/sec: {this.LastGenerationStats.TokensPerSecond:F2}");
        }
    }
}