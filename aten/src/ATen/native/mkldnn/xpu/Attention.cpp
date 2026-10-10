#include <ATen/Context.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/xpu/sdp_utils.h>
#include <torch/library.h> // NOLINT(misc-header-include-cycle)
#include <array>
#include <utility>

namespace {
bool check_head_dim_size_xpu(sdp::sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (!TORCH_GUARD_OR_FALSE(query_size_last.sym_eq(key_size_last))) {
    if (debug) {
      TORCH_WARN(
          "OneDNN attention requires q,k to have the same last dimension.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          " instead.");
    }
    return false;
  }

  constexpr int MAX_HEAD_DIM = 576;
  const auto max_size_last = query_size_last.max(value_size_last);
  if (!TORCH_GUARD_OR_FALSE(max_size_last.sym_le(MAX_HEAD_DIM))) {
    if (debug) {
      TORCH_WARN(
          "OneDNN attention requires q,k,v to have head dimension less than ",
          MAX_HEAD_DIM,
          ". Got ",
          max_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

bool check_no_grad(sdp::sdp_params const& params, bool debug) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  if (debug && any_inputs_require_grad && gradmode_enabled) {
    TORCH_WARN("Backward or grad to be supported.");
  }
  return !any_inputs_require_grad || !gradmode_enabled;
}

bool can_use_overrideable_attention(sdp::sdp_params const& params, bool debug) {
  constexpr auto supported_dtypes = std::to_array<at::ScalarType>(
      {at::kFloat, at::kBFloat16, at::kHalf}); // double is not supported

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints =
      std::to_array<bool (*)(sdp::sdp_params const&, bool)>(
          {sdp::check_nested_tensor,
           sdp::check_for_dropout,
           sdp::check_tensor_shapes,
           sdp::check_batch_size_and_num_heads_dense<
               true /*supports GQA*/,
               true /*requires_same_num_heads*/,
               true /*supports_mqa*/>,
           sdp::check_attn_mask_shape,
           sdp::check_nonzero_sequence_lengths_dense,
           sdp::check_last_dim_stride_equals_1_dense<
               false /*ignore_singleton_dim*/>,
           check_head_dim_size_xpu,
           check_no_grad});
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return sdp::check_tensor_dtype(params, supported_dtypes, debug);
}

bool can_use_cudnn_attention(sdp::sdp_params const& params, bool debug) {
  if (debug) {
    TORCH_WARN("XPU don't support SDPA cudnn attention backend.");
  }
  return false;
}

int64_t minimum_gemm_alignment(sdp::sdp_params const& params) {
  bool is_half = (params.query.dtype() == at::kHalf) ||
      (params.query.dtype() == at::kBFloat16);
  int64_t matmul_alignment_mn = 4;
  int64_t bits_per_scalar = is_half ? 16 : 32;
  matmul_alignment_mn = std::max(matmul_alignment_mn, 128 / bits_per_scalar);

  return matmul_alignment_mn;
}

bool check_head_dim_size_mem_efficient(
    sdp::sdp_params const& params,
    bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  const int64_t alignment = minimum_gemm_alignment(params);
  const bool valid_alignment =
      TORCH_GUARD_OR_FALSE(query_size_last.sym_eq(key_size_last)) &&
      TORCH_GUARD_OR_FALSE((query_size_last % alignment).sym_eq(0)) &&
      TORCH_GUARD_OR_FALSE(query_size_last.sym_gt(0)) &&
      TORCH_GUARD_OR_FALSE((value_size_last % alignment).sym_eq(0)) &&
      TORCH_GUARD_OR_FALSE(value_size_last.sym_gt(0));
  if (!valid_alignment) {
    if (debug) {
      TORCH_WARN(
          "Mem efficient attention requires last dimension of inputs to be divisible by ",
          alignment,
          ". ",
          "Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    return false;
  }
  return true;
}

bool can_use_mem_efficient_attention(
    sdp::sdp_params const& params,
    bool debug) {
  // Define gate functions that determine if a mem efficient can be run
  constexpr auto general_constraints =
      std::to_array<bool (*)(sdp::sdp_params const&, bool)>(
          {sdp::check_runtime_disabled_mem_efficient,
           sdp::check_tensor_shapes,
           check_head_dim_size_mem_efficient});
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints =
        std::to_array<bool (*)(sdp::sdp_params const&, bool)>(
            {sdp::check_requires_grad_and_nested,
             sdp::check_batch_size_nested,
             sdp::check_for_seq_len_0_nested_tensor});
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints =
        std::to_array<bool (*)(sdp::sdp_params const&, bool)>(
            {sdp::check_nonzero_sequence_lengths_dense,
             sdp::check_last_dim_stride_equals_1_dense<false>,
             sdp::check_batch_size_and_num_heads_dense<false>});
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  return true;
}

bool priority_order_init = false;

std::array<sdp::SDPBackend, sdp::num_backends> priority_order(
    sdp::sdp_params const& params) {
  if (!priority_order_init) {
    priority_order_init = true;
    const std::vector<int64_t> priority_order = {
        static_cast<int64_t>(at::SDPBackend::overrideable),
        static_cast<int64_t>(at::SDPBackend::flash_attention),
        static_cast<int64_t>(at::SDPBackend::math),
        static_cast<int64_t>(at::SDPBackend::efficient_attention),
        static_cast<int64_t>(at::SDPBackend::cudnn_attention)};
    at::globalContext().setSDPPriorityOrder(priority_order);
  }
  return at::globalContext().sDPPriorityOrder();
}

sdp::SDPBackend select_sdp_backend_xpu(sdp::sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Math fallback
  auto& ctx = at::globalContext();
  // use overridable linked to onednn as overridable implementation
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledOverrideableSDP() &&
      !ctx.userEnabledFlashSDP() && !ctx.userEnabledMemEfficientSDP()) {
    return sdp::SDPBackend::error;
  }

  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case sdp::SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP() &&
            can_use_overrideable_attention(kernel_params, print_debug)) {
          return sdp::SDPBackend::overrideable;
        }
        break;
      case sdp::SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return sdp::SDPBackend::math;
        }
        break;
      case sdp::SDPBackend::flash_attention:
        if (ctx.userEnabledFlashSDP() &&
            sdp::can_use_flash_attention(kernel_params, print_debug)) {
          return sdp::SDPBackend::flash_attention;
        }
        break;
      case sdp::SDPBackend::cudnn_attention:
        if (ctx.userEnabledCuDNNSDP() &&
            can_use_cudnn_attention(kernel_params, print_debug)) {
          TORCH_CHECK(false, "Invalid backend");
        }
        break;
      case sdp::SDPBackend::efficient_attention:
        if (ctx.userEnabledMemEfficientSDP() &&
            can_use_mem_efficient_attention(kernel_params, print_debug)) {
          TORCH_WARN_ONCE(
              "SDPA Memory Efficient Attention backend is not supported on XPU, falling back to math backend.");
          return sdp::SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. can_use_overridable_attention did not satisfy the constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("Flash attention kernel not used because:");
  sdp::can_use_flash_attention(kernel_params, print_debug);
  TORCH_WARN("Overrideable attention kernel not used because:");
  can_use_overrideable_attention(kernel_params, print_debug);
  TORCH_WARN("CuDNN attention kernel not used because:");
  can_use_cudnn_attention(kernel_params, print_debug);
  TORCH_WARN("Memory Efficient attention kernel not used because:");
  can_use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return sdp::SDPBackend::error;
}
} // namespace

namespace at::native {
// Referenced by native_functions.yaml and REGISTER_XPU_DISPATCH below.
// NOLINTNEXTLINE(misc-use-internal-linkage)
int64_t _fused_sdp_choice_xpu(
    const at::Tensor& query_,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto kernel_params = sdp::normalize_unbatched_input({
      .query = query_,
      .key = key,
      .value = value,
      .attn_mask = attn_mask_,
      .dropout = dropout_p,
      .is_causal = is_causal,
      .enable_gqa = enable_gqa,
  });
  auto backend = select_sdp_backend_xpu(kernel_params);

  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the overrideable kernels.");
  }
  return static_cast<int64_t>(backend);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
// Referenced by native_functions.yaml.
// NOLINTNEXTLINE(misc-use-internal-linkage)
_scaled_dot_product_fused_attention_overrideable_xpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_INTERNAL_ASSERT(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "scaled_dot_product_fused_attention_overrideable_xpu: Accept only 4 dims inputs shape of {(B), H, T, K}");
  TORCH_INTERNAL_ASSERT(
      (key.size(0) == value.size(0)) && (key.size(1) == value.size(1)) &&
          (key.size(2) == value.size(2)),
      "scaled_dot_product_fused_attention_overrideable_xpu: K/V should have the same batch / seq / num_head");
  TORCH_INTERNAL_ASSERT(
      query.size(3) == key.size(3),
      "scaled_dot_product_fused_attention_overrideable_xpu: Q/K should have the same head_dim");
  TORCH_INTERNAL_ASSERT(
      query.size(1) % key.size(1) == 0,
      "scaled_dot_product_fused_attention_overrideable_xpu: number of heads in K/V must divide number of heads in Q");
  TORCH_INTERNAL_ASSERT(
      dropout_p == 0.0,
      "scaled_dot_product_fused_attention_overrideable_xpu: Currently do not support dropout > 0");
  TORCH_INTERNAL_ASSERT(
      !(attn_bias.has_value() && is_causal),
      "scaled_dot_product_fused_attention_overrideable_xpu: attn_bias cannot present with is_causal");

  const int64_t batch_size = query.size(0);
  const int64_t num_head_q = query.size(1);
  const int64_t num_head_kv = key.size(1);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);
  const int64_t seq_len_q = query.size(2);
  const int64_t seq_len_kv = key.size(2);

  at::Tensor output;
  std::vector<int64_t> output_shape = {
      batch_size, num_head_q, seq_len_q, head_dim_v};
  alloc_with_matching_layout(query, output, output_shape);
  at::Tensor logsumexp, debug_attn_mask; // not supported
  // rng not used
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  at::native::onednn::sdpa(
      batch_size,
      seq_len_q,
      seq_len_kv,
      num_head_q,
      num_head_kv,
      head_dim_qk,
      head_dim_v,
      query,
      key,
      value,
      /* q_descale */ std::nullopt,
      /* k_descale */ std::nullopt,
      /* v_descale */ std::nullopt,
      attn_bias,
      is_causal,
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(head_dim_qk)),
      output,
      false,
      logsumexp,
      dropout_p,
      philox_seed,
      philox_offset);

  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      seq_len_kv,
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor,
    at::Tensor>
// Referenced by native_functions.yaml.
// NOLINTNEXTLINE(misc-use-internal-linkage)
_scaled_dot_product_flash_attention_xpu_quantized(
    const Tensor& query, // shape :math:`(N, H_q, L, E)` dtype float8_e4m3fn
    const Tensor& key, // shape :math:`(N, H, S, E)` dtype float8_e4m3fn
    const Tensor& value, // shape :math:`(N, H, S, E_v)` dtype float8_e4m3fn
    const std::optional<Tensor>&
        q_descale, // shape :math:`(N, H)` for PER_HEAD, dtype float32
    const std::optional<Tensor>&
        k_descale, // shape :math:`(N, H)` for PER_HEAD, dtype float32
    const std::optional<Tensor>&
        v_descale, // shape :math:`(N, H)` for PER_HEAD, dtype float32
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "_scaled_dot_product_flash_attention_xpu_quantized: Accept only 4 dims inputs shape of {N, H, L, E}");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat8_e4m3fn &&
          key.scalar_type() == at::kFloat8_e4m3fn &&
          value.scalar_type() == at::kFloat8_e4m3fn,
      "_scaled_dot_product_flash_attention_xpu_quantized: Expected query/key/value to have Float8_e4m3fn data type, but got ",
      query.scalar_type(),
      ", ",
      key.scalar_type(),
      ", ",
      value.scalar_type());
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "_scaled_dot_product_flash_attention_xpu_quantized: Expected query/key/value to be on the same device, but got ",
      query.device(),
      ", ",
      key.device(),
      ", ",
      value.device());

  const int64_t batch_size = query.size(0);
  const int64_t num_head_q = query.size(1);
  const int64_t num_head_kv = key.size(1);
  const int64_t seq_len_q = query.size(2);
  const int64_t seq_len_kv = key.size(2);
  const int64_t head_dim_qk = query.size(3);
  const int64_t head_dim_v = value.size(3);

  TORCH_CHECK(
      query.size(0) == key.size(0),
      "_scaled_dot_product_flash_attention_xpu_quantized: Q/K should have the same batch size");
  TORCH_CHECK(
      (key.size(0) == value.size(0)) && (key.size(1) == value.size(1)) &&
          (key.size(2) == value.size(2)),
      "_scaled_dot_product_flash_attention_xpu_quantized: K/V should have the same batch / seq / num_head");
  TORCH_CHECK(
      head_dim_qk == key.size(3),
      "_scaled_dot_product_flash_attention_xpu_quantized: Q/K should have the same head_dim");
  TORCH_CHECK(
      num_head_q == num_head_kv,
      "_scaled_dot_product_flash_attention_xpu_quantized: oneDNN does not support GQA/MQA for FP8 yet, but got ",
      num_head_q,
      " query heads and ",
      num_head_kv,
      " key/value heads");
  TORCH_CHECK(
      seq_len_q > 0 && key.size(2) > 0,
      "_scaled_dot_product_flash_attention_xpu_quantized: Q/K sequence lengths must be non-zero");
  TORCH_CHECK(
      query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1,
      "_scaled_dot_product_flash_attention_xpu_quantized: Q/K/V must have contiguous last dimension");

  TORCH_CHECK(
      !is_causal,
      "_scaled_dot_product_flash_attention_xpu_quantized: oneDNN does not support is_causal for FP8 yet");
  TORCH_CHECK(
      dropout_p == 0.0,
      "_scaled_dot_product_flash_attention_xpu_quantized: Currently do not support dropout > 0");
  TORCH_CHECK(
      !return_debug_mask,
      "_scaled_dot_product_flash_attention_xpu_quantized: Currently do not support return_debug_mask");

  // Descaling is all-or-nothing: mixing descaled and raw FP8 operands would
  // silently produce a wrongly scaled result.
  const bool has_q_descale = q_descale.has_value();
  const bool has_k_descale = k_descale.has_value();
  const bool has_v_descale = v_descale.has_value();
  TORCH_CHECK(
      has_q_descale && has_q_descale == has_k_descale &&
          has_q_descale == has_v_descale,
      "_scaled_dot_product_flash_attention_xpu_quantized: q_descale, k_descale and v_descale must all be provided, but got ",
      has_q_descale,
      ", ",
      has_k_descale,
      ", ",
      has_v_descale);

  // All descale tensors are indexed by num_head_kv. For GQA, q_descale is
  // broadcast from (N, H_kv) to the query heads internally.
  const auto check_descale = [&](const std::optional<Tensor>& descale,
                                 const char* name) {
    if (!descale.has_value()) {
      return;
    }
    TORCH_CHECK(
        descale->scalar_type() == at::kFloat,
        "_scaled_dot_product_flash_attention_xpu_quantized: ",
        name,
        "_descale must have Float data type, but got ",
        descale->scalar_type());
    TORCH_CHECK(
        descale->device() == query.device(),
        "_scaled_dot_product_flash_attention_xpu_quantized: ",
        name,
        "_descale must be on the same device as query, but got ",
        descale->device(),
        " and ",
        query.device());
    TORCH_CHECK(
        descale->dim() == 2 && descale->size(0) == batch_size &&
            descale->size(1) == num_head_kv,
        "_scaled_dot_product_flash_attention_xpu_quantized: ",
        name,
        "_descale must have shape (",
        batch_size,
        ", ",
        num_head_kv,
        ") for PER_HEAD descaling, but got ",
        descale->sizes());
  };
  check_descale(q_descale, "q");
  check_descale(k_descale, "k");
  check_descale(v_descale, "v");

  const double softmax_scale =
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(head_dim_qk));

  // Attention output; shape :math:`(N, H_q, L, E_v)` dtype bfloat16.
  // FP8 attention accumulates in higher precision and returns BF16, matching
  // the CUDA FA3 quantized op.
  const std::vector<int64_t> output_shape = {
      batch_size, num_head_q, seq_len_q, head_dim_v};
  at::Tensor output =
      at::empty(output_shape, query.options().dtype(at::kBFloat16));

  // logsumexp is only needed by backward, which FP8 does not support.
  at::Tensor logsumexp, debug_attn_mask;
  // dropout is rejected above.
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  at::native::onednn::sdpa(
      batch_size,
      seq_len_q,
      seq_len_kv,
      num_head_q,
      num_head_kv,
      head_dim_qk,
      head_dim_v,
      query,
      key,
      value,
      q_descale,
      k_descale,
      v_descale,
      std::nullopt,
      is_causal,
      softmax_scale,
      output,
      false,
      logsumexp,
      dropout_p,
      philox_seed,
      philox_offset);

  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      key.size(2),
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
}

REGISTER_XPU_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_xpu);
} // namespace at::native
