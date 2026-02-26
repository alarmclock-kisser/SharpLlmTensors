using SharpLlmTensors.Runtime.Modules;
using SharpLlmTensors.Shared;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpLlmTensors.Runtime.Models
{
    public class Qwen2VLModel : Module<Tensor, Tensor>
    {
        private readonly QwenModelInternal model;
        private readonly Module<Tensor, Tensor> lm_head;

        public Qwen2VLModel(JsonElement config) : base("Qwen2VLModel")
        {
            TorchService.LogVerbose($"[Qwen2VLModel] Initializing with JSON config...");
            this.model = new QwenModelInternal(config);
            this.register_module("model", this.model);
            TorchService.LogVerbose("[Qwen2VLModel] <1/2> register_module(model) SUCCESS");

            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            TorchService.LogVerbose($"[Qwen2VLModel] Config hidden_size: {hiddenSize}, vocab_size: {vocabSize}");

            this.lm_head = Linear(hiddenSize, vocabSize, hasBias: false);
            TorchService.LogVerbose("[Qwen2VLModel] Linear layer for lm_head created successfully");
            this.register_module("lm_head", this.lm_head);
            TorchService.LogVerbose("[Qwen2VLModel] <2/2> register_module(lm_head) SUCCESS");
        }

        public override Tensor forward(Tensor inputIds)
        {
            using var x = this.model.forward(inputIds);
            TorchService.LogVerbose($"[Qwen2VLModel] Forward pass through model completed. Output shape: {x.shape}");
            return this.lm_head.forward(x);
        }
    }

    internal class QwenModelInternal : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> embed_tokens;
        private readonly ModuleList<TransformerBlock> layers;
        private readonly RMSNorm norm;

        public QwenModelInternal(JsonElement config) : base("QwenModelInternal")
        {
            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            long numLayers = config.GetProperty("num_hidden_layers").GetInt64();
            TorchService.LogVerbose($"[QwenModelInternal] Initializing with vocab_size: {vocabSize}, hidden_size: {hiddenSize}, num_hidden_layers: {numLayers}");

            this.embed_tokens = Embedding(vocabSize, hiddenSize);
            TorchService.LogVerbose("[QwenModelInternal] Embedding layer created successfully");
            this.register_module("embed_tokens", this.embed_tokens);
            TorchService.LogVerbose("[QwenModelInternal] <1/3> register_module(embed_tokens) SUCCESS");

            var blocks = new List<TransformerBlock>();
            TorchService.LogVerbose($"[QwenModelInternal] Creating Transformer blocks ({numLayers})...");
            for (int i = 0; i < numLayers; i++)
            {
                blocks.Add(new TransformerBlock(config));
                TorchService.LogVerbose($"    [QwenModelInternal] Transformer block {i} created successfully");
            }
            this.layers = new ModuleList<TransformerBlock>(blocks.ToArray());
            TorchService.LogVerbose($"[QwenModelInternal] All Transformer blocks created successfully ({this.layers.Count})");
            this.register_module("layers", this.layers);
            TorchService.LogVerbose("[QwenModelInternal] <2/3> register_module(layers) SUCCESS");

            double eps = config.TryGetProperty("rms_norm_eps", out var e) ? e.GetDouble() : 1e-6;
            TorchService.LogVerbose($"[QwenModelInternal] RMSNorm epsilon: {eps}");
            this.norm = new RMSNorm(new long[] { hiddenSize }, eps);
            TorchService.LogVerbose("[QwenModelInternal] RMSNorm layer created successfully");
            this.register_module("norm", this.norm);
            TorchService.LogVerbose("[QwenModelInternal] <3/3> register_module(norm) SUCCESS");
        }

        public override Tensor forward(Tensor inputIds)
        {
            using var x = this.embed_tokens.forward(inputIds);
            long seqLen = inputIds.shape[1];

            // 1. Causal Mask erstellen (Verhindert den Blick in die Zukunft)
            // Wir nutzen -10000.0 statt float.MinValue, da Float16 bei extremen Werten manchmal zu NaN crasht.
            using var infMask = torch.full(new long[] { seqLen, seqLen }, -10000.0f, dtype: x.dtype, device: x.device);
            TorchService.LogVerbose($"[QwenModelInternal] Causal mask (infMask) created with shape: {infMask.shape} and value: {infMask[0, 0].item<float>()}");

            // torch.triu mit diagonal:1 setzt alles auf und unter der Diagonale auf 0.
            // Oben rechts bleibt -10000. Das ist die perfekte Maske für LLMs!
            using var causalMask = torch.triu(infMask, diagonal: 1).unsqueeze(0).unsqueeze(0);
            TorchService.LogVerbose($"[QwenModelInternal] Causal mask (causalMask) created with shape: {causalMask.shape}");
            var current = x;

            foreach (var layer in this.layers)
            {
                // HIER: causalMask übergeben statt null!
                var next = layer.forward(current, causalMask);

                if (!ReferenceEquals(current, x))
                {
                    current.Dispose();
                }

                current = next;
                TorchService.LogVerbose($"[QwenModelInternal] Forward pass through layer completed. Current output shape: {current.shape}");
            }

            var finalNormed = this.norm.forward(current);
            if (!ReferenceEquals(current, x))
            {
                current.Dispose();
            }

            TorchService.LogVerbose($"[QwenModelInternal] Final RMSNorm applied. Output shape: {finalNormed.shape}");
            return finalNormed;
        }
    }
}