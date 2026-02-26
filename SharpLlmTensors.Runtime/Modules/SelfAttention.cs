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

        private readonly RMSNorm? q_norm;
        private readonly RMSNorm? k_norm;

        private readonly long num_heads;
        private readonly long num_kv_heads;
        private readonly long head_dim;
        private readonly double rope_theta;
        private readonly double attn_logit_softcapping;

        public SelfAttention(JsonElement config) : base("SelfAttention")
        {
            long hiddenSize = config.GetProperty("hidden_size").GetInt64();
            this.num_heads = config.GetProperty("num_attention_heads").GetInt64();

            if (config.TryGetProperty("head_dim", out var hdProp) && hdProp.ValueKind != JsonValueKind.Null)
                this.head_dim = hdProp.GetInt64();
            else
                this.head_dim = hiddenSize / this.num_heads;

            if (config.TryGetProperty("num_key_value_heads", out var kvProp) && kvProp.ValueKind != JsonValueKind.Null)
                this.num_kv_heads = kvProp.GetInt64();
            else
                this.num_kv_heads = this.num_heads;

            if (config.TryGetProperty("rope_theta", out var rtProp) && rtProp.ValueKind != JsonValueKind.Null)
                this.rope_theta = rtProp.GetDouble();
            else
                this.rope_theta = 10000.0;

            if (config.TryGetProperty("attn_logit_softcapping", out var ascProp) && ascProp.ValueKind != JsonValueKind.Null)
                this.attn_logit_softcapping = ascProp.GetDouble();
            else
                this.attn_logit_softcapping = 0.0;

            bool attention_bias = false;
            if (config.TryGetProperty("attention_bias", out var abProp) && abProp.ValueKind != JsonValueKind.Null)
                attention_bias = abProp.GetBoolean();

            this.q_proj = Linear(hiddenSize, this.num_heads * this.head_dim, hasBias: attention_bias);
            this.k_proj = Linear(hiddenSize, this.num_kv_heads * this.head_dim, hasBias: attention_bias);
            this.v_proj = Linear(hiddenSize, this.num_kv_heads * this.head_dim, hasBias: attention_bias);
            this.o_proj = Linear(this.num_heads * this.head_dim, hiddenSize, hasBias: attention_bias);

            this.register_module("q_proj", this.q_proj);
            this.register_module("k_proj", this.k_proj);
            this.register_module("v_proj", this.v_proj);
            this.register_module("o_proj", this.o_proj);

            string modelType = config.TryGetProperty("model_type", out var mt) ? mt.GetString()?.ToLower() ?? "" : "";
            if (modelType.Contains("gemma2") || modelType.Contains("gemma3") || modelType.Contains("gemma3_text"))
            {
                double eps = config.TryGetProperty("rms_norm_eps", out var e) ? e.GetDouble() : 1e-6;
                this.q_norm = new RMSNorm([this.head_dim], eps, addUnitOffset: true);
                this.k_norm = new RMSNorm([this.head_dim], eps, addUnitOffset: true);

                this.register_module("q_norm", this.q_norm);
                this.register_module("k_norm", this.k_norm);
            }
        }

        public override Tensor forward(Tensor x, Tensor attentionMask)
        {
            long batch = x.shape[0];
            long seqLen = x.shape[1];

            using var q_raw = this.q_proj.forward(x);
            using var k_raw = this.k_proj.forward(x);
            using var v_base = this.v_proj.forward(x);

            using var q_reshaped = q_raw.reshape(batch, seqLen, this.num_heads, this.head_dim).transpose(1, 2);
            using var k_reshaped = k_raw.reshape(batch, seqLen, this.num_kv_heads, this.head_dim).transpose(1, 2);
            using var v = v_base.reshape(batch, seqLen, this.num_kv_heads, this.head_dim).transpose(1, 2);

            Tensor q_normed = q_reshaped;
            Tensor k_normed = k_reshaped;

            if (this.q_norm != null && this.k_norm != null)
            {
                q_normed = this.q_norm.forward(q_reshaped);
                k_normed = this.k_norm.forward(k_reshaped);
            }

            using var q = ApplyRoPE(q_normed);
            using var k = ApplyRoPE(k_normed);

            if (this.q_norm != null) { q_normed.Dispose(); k_normed.Dispose(); }

            Tensor k_rep = k;
            Tensor v_rep = v;
            long num_kv_groups = this.num_heads / this.num_kv_heads;
            if (num_kv_groups > 1)
            {
                k_rep = k.repeat_interleave(num_kv_groups, dim: 1);
                v_rep = v.repeat_interleave(num_kv_groups, dim: 1);
            }

            using var k_t = k_rep.transpose(-2, -1);

            // ANTI-NAN FIX: Matmul in Float32 um Float16 Overflow zu vermeiden!
            using var q_f32 = q.to(ScalarType.Float32);
            using var k_t_f32 = k_t.to(ScalarType.Float32);
            using var qk_f32 = torch.matmul(q_f32, k_t_f32);
            using var scaled_qk_f32 = qk_f32 / Math.Sqrt(this.head_dim);

            Tensor final_scores_f32;
            if (this.attn_logit_softcapping > 0)
            {
                using var softcapped = scaled_qk_f32 / this.attn_logit_softcapping;
                using var tanh = softcapped.tanh();
                final_scores_f32 = tanh * this.attn_logit_softcapping;
            }
            else
            {
                final_scores_f32 = scaled_qk_f32.alias();
            }

            using var attentionMask_f32 = attentionMask.to(ScalarType.Float32);
            using var scores_masked_f32 = final_scores_f32 + attentionMask_f32;
            final_scores_f32.Dispose();

            using var probs_f32 = torch.nn.functional.softmax(scores_masked_f32, dim: -1);

            // Zurück nach Float16 für die v_rep Multiplikation (Da Probs zwischen 0 und 1 liegen, ist Overflow hier unmöglich)
            using var probs = probs_f32.to(x.dtype);

            using var context_raw = torch.matmul(probs, v_rep);
            using var context_transposed = context_raw.transpose(1, 2);
            using var context = context_transposed.reshape(batch, seqLen, -1);

            if (num_kv_groups > 1)
            {
                k_rep.Dispose();
                v_rep.Dispose();
            }

            return this.o_proj.forward(context);
        }

        private Tensor ApplyRoPE(Tensor x)
        {
            long seq_len = x.shape[2];
            long head_dim = x.shape[3];

            using var pos = torch.arange(seq_len, dtype: ScalarType.Float32, device: x.device);
            using var dim = torch.arange(0, head_dim, 2, dtype: ScalarType.Float32, device: x.device);

            using var dim_scaled = dim / head_dim;
            using var inv_freq = 1.0f / torch.pow(this.rope_theta, dim_scaled);

            using var freqs = torch.outer(pos, inv_freq);
            using var emb = torch.cat(new[] { freqs, freqs }, dim: -1).to(x.dtype);

            using var cos = emb.cos().unsqueeze(0).unsqueeze(0);
            using var sin = emb.sin().unsqueeze(0).unsqueeze(0);

            var d = (int) (head_dim / 2);
            using var x1 = x.narrow(-1, 0, d);
            using var x2 = x.narrow(-1, d, d);
            using var neg_x2 = -x2;
            using var x_half = torch.cat(new[] { neg_x2, x1 }, dim: -1);

            using var x_cos = x * cos;
            using var x_half_sin = x_half * sin;

            return x_cos + x_half_sin;
        }
    }
}