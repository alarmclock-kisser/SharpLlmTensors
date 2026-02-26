using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpLlmTensors.Runtime.Modules
{
    // Die Basisklasse MUSS 3 Tensoren definieren: Input1(x), Input2(mask), Output
    public class TransformerBlock : Module<Tensor, Tensor, Tensor>
    {
        private readonly RMSNorm input_layernorm;
        private readonly RMSNorm post_attention_layernorm;
        private readonly SwiGLUMLP mlp;
        private readonly SelfAttention self_attn;

        // FIX: Optionaler layerIndex, damit LlamaModel (config, i) aufrufen kann
        public TransformerBlock(JsonElement config, int layerIndex = -1)
            : base(layerIndex >= 0 ? $"TransformerBlock_{layerIndex}" : "TransformerBlock")
        {
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            double eps = config.TryGetProperty("rms_norm_eps", out var e) ? e.GetDouble() : 1e-6;

            TorchService.LogVerbose($"[TransformerBlock] Initializing (layer {layerIndex}) hidden_size: {hiddenSize}, eps: {eps}");

            this.input_layernorm = new RMSNorm([hiddenSize], eps);
            this.post_attention_layernorm = new RMSNorm([hiddenSize], eps);
            this.mlp = new SwiGLUMLP(config);
            this.self_attn = new SelfAttention(config);

            TorchService.LogVerbose($"[TransformerBlock] Modules created for layer {layerIndex}");

            this.register_module("input_layernorm", this.input_layernorm);
            this.register_module("post_attention_layernorm", this.post_attention_layernorm);
            this.register_module("mlp", this.mlp);
            this.register_module("self_attn", this.self_attn);
            TorchService.LogVerbose($"[TransformerBlock] Modules registered for layer {layerIndex}");
        }

        public override Tensor forward(Tensor x, Tensor attentionMask)
        {
            TorchService.LogVerbose($"[TransformerBlock] Forward called. input shape: {string.Join(',', x.shape)}");
            // Attention Branch with Residual
            using var normed1 = this.input_layernorm.forward(x);
            TorchService.LogVerbose($"[TransformerBlock] input_layernorm applied. shape: {string.Join(',', normed1.shape)}");
            using var attnOut = this.self_attn.forward(normed1, attentionMask);
            TorchService.LogVerbose($"[TransformerBlock] self_attn forward completed. shape: {string.Join(',', attnOut.shape)}");
            using var h = x + attnOut;

            // MLP Branch with Residual
            using var normed2 = this.post_attention_layernorm.forward(h);
            TorchService.LogVerbose($"[TransformerBlock] post_attention_layernorm applied. shape: {string.Join(',', normed2.shape)}");
            using var mlpOut = this.mlp.forward(normed2);
            TorchService.LogVerbose($"[TransformerBlock] mlp forward completed. shape: {string.Join(',', mlpOut.shape)}");

            return h + mlpOut;
        }
    }
}