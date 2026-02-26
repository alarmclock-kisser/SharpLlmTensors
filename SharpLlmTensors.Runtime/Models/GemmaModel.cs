using SharpLlmTensors.Runtime.Modules;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System;
using System.Collections.Generic;

namespace SharpLlmTensors.Runtime.Models
{
    public class GemmaModel : Module<Tensor, Tensor>
    {
        private readonly GemmaModelInternal model;
        private readonly Module<Tensor, Tensor>? lm_head;

        private readonly double _hiddenSize;
        private readonly bool _tieWordEmbeddings;
        private readonly double _finalLogitSoftcapping;

        public GemmaModel(JsonElement config) : base("GemmaModel")
        {
            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            this._hiddenSize = config.GetProperty("hidden_size").GetDouble();

            this._tieWordEmbeddings = true;
            if (config.TryGetProperty("tie_word_embeddings", out var tieProp) && tieProp.ValueKind != JsonValueKind.Null)
            {
                this._tieWordEmbeddings = tieProp.GetBoolean();
            }

            this._finalLogitSoftcapping = 0.0;
            if (config.TryGetProperty("final_logit_softcapping", out var capProp) && capProp.ValueKind != JsonValueKind.Null)
            {
                this._finalLogitSoftcapping = capProp.GetDouble();
            }

            this.model = new GemmaModelInternal(config, this._hiddenSize);
            this.register_module("model", this.model);

            if (!this._tieWordEmbeddings)
            {
                this.lm_head = Linear((long) this._hiddenSize, vocabSize, hasBias: false);
                this.register_module("lm_head", this.lm_head);
            }
        }

        public override Tensor forward(Tensor inputIds)
        {
            using var finalNorm = this.model.forward(inputIds);
            Tensor logits;

            // ANTI-NAN FIX: Wir kalkulieren die finalen Logits komplett in Float32!
            using var finalNorm_f32 = finalNorm.to(ScalarType.Float32);

            if (this._tieWordEmbeddings)
            {
                using var embedWeight = this.model.embed_tokens.weight ?? throw new InvalidOperationException("Embedding weight is null");
                using var embedWeightT = embedWeight.transpose(0, 1);
                using var embedWeightT_f32 = embedWeightT.to(ScalarType.Float32);
                logits = matmul(finalNorm_f32, embedWeightT_f32);
            }
            else
            {
                using var w = this.lm_head!.get_parameter("weight");
                using var w_f32 = w.to(ScalarType.Float32);
                using var w_t_f32 = w_f32.transpose(0, 1);
                logits = matmul(finalNorm_f32, w_t_f32);
            }

            if (this._finalLogitSoftcapping > 0)
            {
                using var softcapped = logits / this._finalLogitSoftcapping;
                using var tanh = softcapped.tanh();
                var finalLogits = tanh * this._finalLogitSoftcapping;

                logits.Dispose();
                logits = finalLogits;
            }

            // Wir geben explizit Float32-Logits an die Inference-Schleife zurück,
            // Argmax und Softmax arbeiten damit wunderbar weiter.
            return logits;
        }
    }

    internal class GemmaModelInternal : Module<Tensor, Tensor>
    {
        public readonly TorchSharp.Modules.Embedding embed_tokens;
        private readonly ModuleList<TransformerBlock> layers;
        private readonly RMSNorm norm;
        private readonly double _hiddenSize;

        public GemmaModelInternal(JsonElement config, double hiddenSize) : base("GemmaModelInternal")
        {
            this._hiddenSize = hiddenSize;
            long vocabSize = config.GetProperty("vocab_size").GetInt64();
            long numLayers = config.GetProperty("num_hidden_layers").GetInt64();

            double eps = 1e-6;
            if (config.TryGetProperty("rms_norm_eps", out var eProp) && eProp.ValueKind != JsonValueKind.Null)
            {
                eps = eProp.GetDouble();
            }

            this.embed_tokens = Embedding(vocabSize, (long) this._hiddenSize);
            this.register_module("embed_tokens", this.embed_tokens);

            var blocks = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
            {
                blocks.Add(new TransformerBlock(config, i));
            }
            this.layers = new ModuleList<TransformerBlock>(blocks.ToArray());
            this.register_module("layers", this.layers);

            this.norm = new RMSNorm(new long[] { (long) this._hiddenSize }, eps, addUnitOffset: true);
            this.register_module("norm", this.norm);
        }

        public override Tensor forward(Tensor inputIds)
        {
            using var embedded = this.embed_tokens.forward(inputIds);
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

            var finalNorm = this.norm.forward(current);
            if (!ReferenceEquals(current, x)) current.Dispose();

            return finalNorm;
        }
    }
}