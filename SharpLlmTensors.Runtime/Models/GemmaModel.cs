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
        private readonly TorchSharp.Modules.Embedding embed_tokens;
        private readonly ModuleList<TransformerBlock> layers;
        private readonly RMSNorm norm;
        private readonly Module<Tensor, Tensor>? lm_head;

        private readonly double _hiddenSize;
        private readonly bool _tieWordEmbeddings;
        private readonly double _finalLogitSoftcapping;

        public GemmaModel(JsonElement config) : base("GemmaModel")
        {
            TorchService.LogVerbose("[GemmaModel] Start Initializing...");

            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            this._hiddenSize = config.GetProperty("hidden_size").GetDouble();
            long numLayers = config.GetProperty("num_hidden_layers").GetInt64();

            // FIX: Prüfen ob das Element existiert UND nicht null ist!
            double eps = 1e-6;
            if (config.TryGetProperty("rms_norm_eps", out var eProp) && eProp.ValueKind != JsonValueKind.Null)
            {
                eps = eProp.GetDouble();
            }

            // FIX: Tie Embeddings sicher auslesen
            this._tieWordEmbeddings = true; // Gemma Standard
            if (config.TryGetProperty("tie_word_embeddings", out var tieProp) && tieProp.ValueKind != JsonValueKind.Null)
            {
                this._tieWordEmbeddings = tieProp.GetBoolean();
            }

            // FIX: Softcapping sicher auslesen (ist in der JSON oft explizit "null")
            this._finalLogitSoftcapping = 0.0;
            if (config.TryGetProperty("final_logit_softcapping", out var capProp) && capProp.ValueKind != JsonValueKind.Null)
            {
                this._finalLogitSoftcapping = capProp.GetDouble();
            }

            TorchService.LogVerbose($"[GemmaModel] Config loaded - Tie Embeddings: {this._tieWordEmbeddings}, Softcap: {this._finalLogitSoftcapping}");

            // 1. Embeddings
            this.embed_tokens = torch.nn.Embedding(vocabSize, (long) this._hiddenSize);
            this.register_module("embed_tokens", this.embed_tokens);

            // 2. Layers
            var blocks = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
            {
                blocks.Add(new TransformerBlock(config, i));
            }
            this.layers = new ModuleList<TransformerBlock>(blocks.ToArray());
            this.register_module("layers", this.layers);

            // 3. Final Norm (mit addUnitOffset: true für Gemma!)
            this.norm = new RMSNorm(new long[] { (long) this._hiddenSize }, eps, addUnitOffset: true);
            this.register_module("norm", this.norm);

            // 4. Output Head (Nur erstellen, wenn das Modell NICHT tied ist)
            if (!this._tieWordEmbeddings)
            {
                this.lm_head = Linear((long) this._hiddenSize, vocabSize, hasBias: false);
                this.register_module("lm_head", this.lm_head);
                TorchService.LogVerbose("[GemmaModel] Registered standalone 'lm_head'");
            }
        }

        public override Tensor forward(Tensor inputIds)
        {
            using var embedded = this.embed_tokens.forward(inputIds);

            // WICHTIG: Gemma skaliert das Input-Embedding mit der Wurzel der Hidden Size
            using var x = embedded * Math.Sqrt(this._hiddenSize);

            long seqLen = inputIds.shape[1];
            using var infMask = torch.full(new long[] { seqLen, seqLen }, -10000.0f, dtype: x.dtype, device: x.device);
            using var causalMask = torch.triu(infMask, diagonal: 1).unsqueeze(0).unsqueeze(0);

            var current = x;
            foreach (var layer in this.layers)
            {
                var next = layer.forward(current, causalMask);
                if (!ReferenceEquals(current, x)) current.Dispose();
                current = next;
            }

            using var finalNorm = this.norm.forward(current);
            if (!ReferenceEquals(current, x)) current.Dispose();

            Tensor logits;

            // Tied Embeddings: Wir recyclen das Gewicht des Eingangs-Layers!
            if (this._tieWordEmbeddings)
            {
                using var embedWeightT = this.embed_tokens.weight?.transpose(0, 1);
                logits = matmul(finalNorm, embedWeightT ?? throw new InvalidOperationException("Embedding weight is null"));
            }
            else
            {
                logits = this.lm_head!.forward(finalNorm);
            }

            // Logit Softcapping
            if (this._finalLogitSoftcapping > 0)
            {
                using var softcapped = logits / this._finalLogitSoftcapping;
                using var tanh = softcapped.tanh();
                var finalLogits = tanh * this._finalLogitSoftcapping;

                logits.Dispose(); // Alten Tensor wegräumen
                logits = finalLogits;
            }

            return logits;
        }
    }
}