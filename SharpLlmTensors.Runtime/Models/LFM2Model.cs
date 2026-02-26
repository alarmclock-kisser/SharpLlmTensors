using SharpLlmTensors.Runtime.Modules;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpLlmTensors.Runtime.Models
{
    public class LFM2Model : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> embed_tokens;
        private readonly Lfm2Block[] layers;
        private readonly RMSNorm norm;
        private readonly Module<Tensor, Tensor> lm_head;

        public LFM2Model(JsonElement config) : base("Lfm2Model")
        {
            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            long numLayers = config.GetProperty("num_hidden_layers").GetInt64();

            TorchService.LogVerbose($"[Lfm2Model] Initializing. vocabSize: {vocabSize}, hiddenSize: {hiddenSize}, numLayers: {numLayers}");

            double eps = config.TryGetProperty("rms_norm_eps", out var e) ? e.GetDouble() : 1e-5;

            this.embed_tokens = Embedding(vocabSize, hiddenSize);
            TorchService.LogVerbose("[Lfm2Model] Embedding layer created");
            this.norm = new RMSNorm([hiddenSize], eps);
            TorchService.LogVerbose("[Lfm2Model] RMSNorm created");
            this.lm_head = Linear(hiddenSize, vocabSize, hasBias: false);
            TorchService.LogVerbose("[Lfm2Model] LM head created");

            this.register_module("model.embed_tokens", this.embed_tokens);
            this.register_module("model.norm", this.norm);
            this.register_module("lm_head", this.lm_head);

            // LFM2 nutzt spezielle Blöcke, keine Standard-Transformer
            this.layers = new Lfm2Block[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                var layer = new Lfm2Block(config, i);
                this.layers[i] = layer;
                this.register_module($"model.layers.{i}", layer);
                TorchService.LogVerbose($"[Lfm2Model] Lfm2Block {i} created and registered");
            }
        }

        public override Tensor forward(Tensor inputIds)
        {
            TorchService.LogVerbose($"[Lfm2Model] Forward called. input shape: {string.Join(',', inputIds.shape)}");
            using var x = this.embed_tokens.forward(inputIds);
            var current = x;

            // State Space Models benötigen KEINE Causal Mask, 
            // da sie von Natur aus kausal (sequentiell) arbeiten!
            foreach (var layer in this.layers)
            {
                var next = layer.forward(current);
                TorchService.LogVerbose($"[Lfm2Model] Layer forward completed. output shape: {string.Join(',', next.shape)}");
                if (!ReferenceEquals(current, x))
                {
                    current.Dispose();
                }

                current = next;
            }

            using var finalNormed = this.norm.forward(current);
            if (!ReferenceEquals(current, x))
            {
                current.Dispose();
            }

            return this.lm_head.forward(finalNormed);
        }
    }

    /// <summary>
    /// Der spezielle Block für Liquid Foundation Models (LFM2).
    /// Nutzt Token-Mixing (SSM/Conv) anstelle von Self-Attention.
    /// </summary>
    public class Lfm2Block : Module<Tensor, Tensor>
    {
        private readonly RMSNorm input_layernorm;
        private readonly RMSNorm post_attention_layernorm;

        // LiquidAI spezifische Layer
        private readonly Module<Tensor, Tensor> mixer_proj;
        private readonly Module<Tensor, Tensor> mlp_proj;

        public Lfm2Block(JsonElement config, int layerIndex) : base($"Lfm2Block_{layerIndex}")
        {
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            double eps = config.TryGetProperty("rms_norm_eps", out var e) ? e.GetDouble() : 1e-5;

            this.input_layernorm = new RMSNorm([hiddenSize], eps);
            this.post_attention_layernorm = new RMSNorm([hiddenSize], eps);

            // Placeholder für das komplexe Token-Mixing von LiquidAI
            this.mixer_proj = Linear(hiddenSize, hiddenSize, hasBias: false);
            this.mlp_proj = Linear(hiddenSize, hiddenSize, hasBias: false);

            this.register_module("input_layernorm", this.input_layernorm);
            this.register_module("post_attention_layernorm", this.post_attention_layernorm);

            // Die Namen müssen exakt mit der Safetensors-Datei übereinstimmen
            this.register_module("mixer.proj", this.mixer_proj);
            this.register_module("mlp.proj", this.mlp_proj);
        }

        public override Tensor forward(Tensor x)
        {
            // Mixer Branch (ersetzt Self-Attention)
            using var normed1 = this.input_layernorm.forward(x);
            using var mixed = this.mixer_proj.forward(normed1);
            using var h = x + mixed;

            // Channel-Mixing Branch (ersetzt Standard MLP)
            using var normed2 = this.post_attention_layernorm.forward(h);
            using var mlpOut = this.mlp_proj.forward(normed2);

            return h + mlpOut;
        }
    }
}