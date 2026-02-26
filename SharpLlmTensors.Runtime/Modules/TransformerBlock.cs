using SharpLlmTensors.Shared;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpLlmTensors.Runtime.Modules
{
    public class TransformerBlock : Module<Tensor, Tensor, Tensor>
    {
        private readonly RMSNorm input_layernorm;
        private readonly RMSNorm post_attention_layernorm;
        private readonly SwiGLUMLP mlp;
        private readonly SelfAttention self_attn;

        public TransformerBlock(JsonElement config, int layerIndex = -1)
            : base(layerIndex >= 0 ? $"TransformerBlock_{layerIndex}" : "TransformerBlock")
        {
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            double eps = config.TryGetProperty("rms_norm_eps", out var e) ? e.GetDouble() : 1e-6;
            TorchService.LogVerbose($"[TransformerBlock] Initializing block {(layerIndex >= 0 ? layerIndex.ToString() : "")} with hidden_size: {hiddenSize}, eps: {eps}");

            // FIX: Prüfen, ob wir ein Gemma-Modell laden
            string modelType = "";
            if (config.TryGetProperty("model_type", out var mt) && mt.ValueKind == JsonValueKind.String)
            {
                modelType = mt.GetString()?.ToLower() ?? "";
            }
            bool isGemma = modelType.Contains("gemma");

            this.input_layernorm = new RMSNorm([hiddenSize], eps, isGemma);
            this.post_attention_layernorm = new RMSNorm([hiddenSize], eps, isGemma);
            this.mlp = new SwiGLUMLP(config);
            this.self_attn = new SelfAttention(config);

            this.register_module("input_layernorm", this.input_layernorm);
            TorchService.LogVerbose($"[TransformerBlock] input_layernorm initialized with hidden_size: {hiddenSize}, eps: {eps}, isGemma: {isGemma}");
            this.register_module("post_attention_layernorm", this.post_attention_layernorm);
            TorchService.LogVerbose($"[TransformerBlock] post_attention_layernorm initialized with hidden_size: {hiddenSize}, eps: {eps}, isGemma: {isGemma}");
            this.register_module("mlp", this.mlp);
            TorchService.LogVerbose($"[TransformerBlock] MLP initialized with hidden_size: {hiddenSize}, eps: {eps}, isGemma: {isGemma}");
            this.register_module("self_attn", this.self_attn);
            TorchService.LogVerbose($"[TransformerBlock] SelfAttention initialized with hidden_size: {hiddenSize}, eps: {eps}, isGemma: {isGemma}");

            TorchService.LogVerbose($"[TransformerBlock] Initialized {(isGemma ? "Gemma" : "Standard")} block with hidden_size: {hiddenSize}, eps: {eps}");
        }

        public override Tensor forward(Tensor x, Tensor attentionMask)
        {
            using var normed1 = this.input_layernorm.forward(x);
            using var attnOut = this.self_attn.forward(normed1, attentionMask);
            using var h = x + attnOut;

            using var normed2 = this.post_attention_layernorm.forward(h);
            using var mlpOut = this.mlp.forward(normed2);

            return h + mlpOut;
        }
    }
}