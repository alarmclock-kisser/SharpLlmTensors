using SharpLlmTensors.Runtime.Modules;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace SharpLlmTensors.Runtime.Models
{
    public class LlamaModel : Module<Tensor, Tensor>
    {
        private readonly LlamaModelInternal model;
        private readonly Module<Tensor, Tensor> lm_head;

        public LlamaModel(JsonElement config) : base("LlamaModel")
        {
            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();

            TorchService.LogVerbose($"[LlamaModel] Initializing Llama 3.2...");

            // 1. Die innere "model"-Hierarchie erzeugen
            this.model = new LlamaModelInternal(config);
            this.register_module("model", this.model);
            TorchService.LogVerbose("[LlamaModel] Registered module: model");

            // 2. Der "lm_head" liegt ganz außen
            this.lm_head = Linear(hiddenSize, vocabSize, hasBias: false);
            this.register_module("lm_head", this.lm_head);
            TorchService.LogVerbose("[LlamaModel] Registered module: lm_head");
        }

        public override Tensor forward(Tensor inputIds)
        {
            using var x = this.model.forward(inputIds);
            return this.lm_head.forward(x);
        }
    }

    // Die interne Klasse baut die "model.*" Hierarchie auf, OHNE Punkte im Namen zu verwenden
    internal class LlamaModelInternal : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> embed_tokens;
        private readonly ModuleList<TransformerBlock> layers;
        private readonly RMSNorm norm;

        public LlamaModelInternal(JsonElement config) : base("LlamaModelInternal")
        {
            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            long numLayers = config.GetProperty("num_hidden_layers").GetInt64();
            double eps = config.TryGetProperty("rms_norm_eps", out var e) ? e.GetDouble() : 1e-5;

            this.embed_tokens = Embedding(vocabSize, hiddenSize);
            this.norm = new RMSNorm([hiddenSize], eps);

            // Registrierung OHNE Punkte (da wir bereits im "model"-Modul sind)
            this.register_module("embed_tokens", this.embed_tokens);
            TorchService.LogVerbose("[LlamaModelInternal] Registered module: embed_tokens");

            this.register_module("norm", this.norm);
            TorchService.LogVerbose("[LlamaModelInternal] Registered module: norm");

            // Array erstellen und der ModuleList übergeben 
            // (ModuleList kümmert sich intern automatisch um die Registrierung von "layers.0", "layers.1" etc.)
            var layersArray = new TransformerBlock[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                layersArray[i] = new TransformerBlock(config, i);
                TorchService.LogVerbose($"[LlamaModelInternal] Created TransformerBlock for layer {i}");
            }

            this.layers = new ModuleList<TransformerBlock>(layersArray);
            this.register_module("layers", this.layers);
            TorchService.LogVerbose("[LlamaModelInternal] Registered module: layers (ModuleList)");
        }

        public override Tensor forward(Tensor inputIds)
        {
            using var x = this.embed_tokens.forward(inputIds);
            long seqLen = inputIds.shape[1];

            // Causal Mask
            using var infMask = torch.full(new long[] { seqLen, seqLen }, -10000.0f, dtype: x.dtype, device: x.device);
            using var causalMask = torch.triu(infMask, diagonal: 1).unsqueeze(0).unsqueeze(0);

            var current = x;

            foreach (var layer in this.layers)
            {
                var next = layer.forward(current, causalMask);
                if (!ReferenceEquals(current, x))
                {
                    current.Dispose();
                }

                current = next;
            }

            var finalNormed = this.norm.forward(current);
            if (!ReferenceEquals(current, x))
            {
                current.Dispose();
            }

            return finalNormed;
        }
    }
}