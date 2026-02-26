using SharpLlmTensors.Runtime.Modules;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpLlmTensors.Runtime.Models
{
    public class GemmaModel : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> embed_tokens;
        private readonly ModuleList<TransformerBlock> layers;
        private readonly RMSNorm norm;
        private readonly Module<Tensor, Tensor> lm_head;
        private readonly double _hiddenSize;

        public GemmaModel(JsonElement config) : base("GemmaModel")
        {
            TorchService.LogVerbose("[GemmaModel] Start Initializing...");

            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            this._hiddenSize = config.GetProperty("hidden_size").GetDouble();
            long numLayers = config.GetProperty("num_hidden_layers").GetInt64();

            double eps = 1e-6;
            if (config.TryGetProperty("rms_norm_eps", out var eProp))
            {
                eps = eProp.GetDouble();
            }

            // 1. Embeddings
            this.embed_tokens = Embedding(vocabSize, (long) this._hiddenSize);
            this.register_module("embed_tokens", this.embed_tokens);
            TorchService.LogVerbose($"[GemmaModel] Registered 'embed_tokens' (Vocab: {vocabSize}, Hidden: {this._hiddenSize})");

            // 2. Transformer Layers via ModuleList (WICHTIG für Safetensors-Struktur)
            var blocks = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
            {
                var block = new TransformerBlock(config, i);
                blocks.Add(block);
                TorchService.LogVerbose($"[GemmaModel] Created TransformerBlock for Layer {i}");
            }
            this.layers = new ModuleList<TransformerBlock>(blocks.ToArray());
            this.register_module("layers", this.layers);
            TorchService.LogVerbose("[GemmaModel] Registered 'layers' (ModuleList)");

            // 3. Final Norm
            this.norm = new RMSNorm(new long[] { (long) this._hiddenSize }, eps);
            this.register_module("norm", this.norm);
            TorchService.LogVerbose($"[GemmaModel] Registered 'norm' with eps: {eps}");

            // 4. Output Head
            this.lm_head = Linear((long) this._hiddenSize, vocabSize, hasBias: false);
            this.register_module("lm_head", this.lm_head);
            TorchService.LogVerbose("[GemmaModel] Registered 'lm_head'");
        }

        public override Tensor forward(Tensor inputIds)
        {
            TorchService.LogVerbose($"[GemmaModel] Forward started. Input shape: {string.Join(',', inputIds.shape)}");

            using var embedded = this.embed_tokens.forward(inputIds);
            using var x = embedded * Math.Sqrt(this._hiddenSize);

            long seqLen = inputIds.shape[1];
            using var infMask = torch.full(new long[] { seqLen, seqLen }, -10000.0f, dtype: x.dtype, device: x.device);
            using var causalMask = torch.triu(infMask, diagonal: 1).unsqueeze(0).unsqueeze(0);

            var current = x;
            int i = 0;
            foreach (var layer in this.layers)
            {
                var next = layer.forward(current, causalMask);
                if (!ReferenceEquals(current, x))
                {
                    current.Dispose();
                }

                current = next;
                i++;
            }

            using var finalNorm = this.norm.forward(current);
            if (!ReferenceEquals(current, x))
            {
                current.Dispose();
            }

            var logits = this.lm_head.forward(finalNorm);
            TorchService.LogVerbose($"[GemmaModel] Forward finished. Logits shape: {string.Join(',', logits.shape)}");
            return logits;
        }
    }
}