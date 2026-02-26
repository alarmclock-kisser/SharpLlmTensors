using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Text.Json;

namespace SharpLlmTensors.Runtime.Modules
{
    public class SelfAttention : Module<Tensor, Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> q_proj;
        private readonly Module<Tensor, Tensor> k_proj;
        private readonly Module<Tensor, Tensor> v_proj;
        private readonly Module<Tensor, Tensor> o_proj;

        private readonly long num_heads;
        private readonly long num_kv_heads;
        private readonly long head_dim;

        public SelfAttention(JsonElement config) : base("SelfAttention")
        {
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            this.num_heads = config.GetProperty("num_attention_heads").GetInt64();
            this.head_dim = hiddenSize / this.num_heads;

            TorchService.LogVerbose($"[SelfAttention] Initializing. hiddenSize: {hiddenSize}, num_heads: {this.num_heads}, head_dim: {this.head_dim}");

            // FIX: Lese num_key_value_heads für GQA (Grouped-Query Attention) aus
            if (config.TryGetProperty("num_key_value_heads", out var kvProp))
            {
                this.num_kv_heads = kvProp.GetInt64();
            }
            else
            {
                this.num_kv_heads = this.num_heads; // Fallback für alte Modelle
            }

            long kv_dim = this.num_kv_heads * this.head_dim;

            // Q bleibt voll, K und V werden kleiner!
            this.q_proj = Linear(hiddenSize, hiddenSize, hasBias: true);
            this.k_proj = Linear(hiddenSize, kv_dim, hasBias: true);
            this.v_proj = Linear(hiddenSize, kv_dim, hasBias: true);
            this.o_proj = Linear(hiddenSize, hiddenSize, hasBias: false);

            this.register_module("q_proj", this.q_proj);
            this.register_module("k_proj", this.k_proj);
            this.register_module("v_proj", this.v_proj);
            this.register_module("o_proj", this.o_proj);
            TorchService.LogVerbose("[SelfAttention] Projection modules registered");
        }

        public override Tensor forward(Tensor x, Tensor mask)
        {
            TorchService.LogVerbose($"[SelfAttention] Forward called. input shape: {string.Join(',', x.shape)}");
            long batch = x.shape[0];
            long seqLen = x.shape[1];

            using var q_raw = this.q_proj.forward(x).view(batch, seqLen, this.num_heads, this.head_dim).transpose(1, 2);
            TorchService.LogVerbose($"[SelfAttention] q_proj produced q_raw shape: {string.Join(',', q_raw.shape)}");
            using var k_raw = this.k_proj.forward(x).view(batch, seqLen, this.num_kv_heads, this.head_dim).transpose(1, 2);
            TorchService.LogVerbose($"[SelfAttention] k_proj produced k_raw shape: {string.Join(',', k_raw.shape)}");
            using var v_base = this.v_proj.forward(x).view(batch, seqLen, this.num_kv_heads, this.head_dim).transpose(1, 2);
            TorchService.LogVerbose($"[SelfAttention] v_proj produced v_base shape: {string.Join(',', v_base.shape)}");

            // HIER WIRD ROPE ANGEWANDT:
            using var q = this.ApplyRoPE(q_raw);
            using var k_base = this.ApplyRoPE(k_raw);

            long num_kv_groups = this.num_heads / this.num_kv_heads;

            Tensor k = num_kv_groups > 1 ? k_base.repeat_interleave(num_kv_groups, dim: 1) : k_base;
            Tensor v = num_kv_groups > 1 ? v_base.repeat_interleave(num_kv_groups, dim: 1) : v_base;

            // FIX 2: Berechne die Scores in Float32, um NaN zu vermeiden, und skaliere mit der tatsächlichen head_dim
            var d_k = (float) this.head_dim; // Sicherstellen, dass es Float ist
            var scores = matmul(q, k.transpose(-2, -1)) / (float) Math.Sqrt(d_k);

            TorchService.LogVerbose($"[SelfAttention] Scores computed. shape: {string.Join(',', scores.shape)}");

            if (!ReferenceEquals(mask, null))
            {
                var maskedScores = scores + mask;
                scores.Dispose();
                scores = maskedScores;
            }

            // WICHTIG: Softmax immer in Float32 berechnen, um NaN zu vermeiden!
            using var scores_f32 = scores.to(ScalarType.Float32);
            using var weights_f32 = functional.softmax(scores_f32, dim: -1);

            // Zurück zu Float16 für die Matrixmultiplikation mit V
            using var weights = weights_f32.to(x.dtype);
            scores.Dispose();

            using var context = matmul(weights, v).transpose(1, 2).reshape(batch, seqLen, -1);

            TorchService.LogVerbose($"[SelfAttention] Context computed. shape: {string.Join(',', context.shape)}");

            if (num_kv_groups > 1)
            {
                k.Dispose();
                v.Dispose();
            }

            return this.o_proj.forward(context);
        }

        private Tensor ApplyRoPE(Tensor x)
        {
            TorchService.LogVerbose($"[SelfAttention] ApplyRoPE called. input shape: {string.Join(',', x.shape)}");
            long seq_len = x.shape[2];
            long head_dim = x.shape[3];

            using var pos = torch.arange(seq_len, dtype: ScalarType.Float32, device: x.device);
            using var dim = torch.arange(0, head_dim, 2, dtype: ScalarType.Float32, device: x.device);

            // Qwen2.5 nutzt Base 1.000.000 für die Frequenzen
            using var inv_freq = 1.0f / torch.pow(1000000.0f, dim / head_dim);

            using var freqs = torch.outer(pos, inv_freq);
            using var emb = torch.cat(new[] { freqs, freqs }, dim: -1).to(x.dtype);

            using var cos = emb.cos().unsqueeze(0).unsqueeze(0);
            using var sin = emb.sin().unsqueeze(0).unsqueeze(0);

            // Tensoren aufteilen und rotieren [-x2, x1]
            var half = (int) (head_dim / 2);
            using var x1 = x[.., .., .., ..half];
            using var x2 = x[.., .., .., half..];
            using var neg_x2 = -x2;
            using var x_rotated = torch.cat(new[] { neg_x2, x1 }, dim: -1);

            return (x * cos) + (x_rotated * sin);
        }
    }
}