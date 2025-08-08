#include <ATen/native/mkldnn/xpu/Attention.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/Array.h>
#include <torch/library.h>

namespace sdp {
namespace {
template<bool kv_has_same_headdim=false>
bool check_head_dim_size_xpu(sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (query_size_last != key_size_last) {
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

  if (kv_has_same_headdim && key_size_last != value_size_last) {
    if (debug) {
      TORCH_WARN(
          "Flash attention requires k,v to have the same last dimension.",
          " Got Key.size(-1): ",
          key_size_last,
          ", Value.size(-1): ",
          value_size_last,
          " instead.");
    }
    return false;
  }

  constexpr int MAX_HEAD_DIM = 576;
  const auto max_size_last = query_size_last.max(value_size_last);
  if (max_size_last > MAX_HEAD_DIM) {
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

bool check_no_grad(sdp_params const& params, bool debug) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  if (debug && any_inputs_require_grad && gradmode_enabled) {
    TORCH_WARN("Backward or grad to be supported.");
  }
  return !any_inputs_require_grad || !gradmode_enabled;
}

bool check_all_tensors_on_device(sdp_params const& params, bool debug) {
  // Check that all tensors are on the Intel GPU device
  // This should be handled by the stub dispatch, but we call can_use_*_attention
  // directly from python we need to ensure that the tensors are on xpu
  if (params.query.device().type() != at::DeviceType::XPU) {
    if (debug) {
      TORCH_WARN(
          "All tensors need to be on xpu device. Got query on device: ",
          params.query.device(),
          ", key on device: ",
          params.key.device(),
          ", value on device: ",
          params.value.device());
    }
    return false;
  }
  return true;
}

bool check_flash_causal_non_square_seqlens(sdp_params const& params, bool debug) {
  // FlashAttention's causal mask is now aligned to lower_right.
  if (params.is_causal &&
      !params.query.is_nested() && !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k. ",
          "Got seqlen_q: ", params.query.sym_size(-2), " seqlen_k: ",
          params.key.sym_size(-2), ". If you would like to use causal attention with non-square masks, please see CausalAttnMask.");
    }
    return false;
  }
  return true;
}
} // namespace 

bool use_overrideable_xpu(sdp_params const& params, bool debug) {
  constexpr auto supported_dtypes = c10::array_of<at::ScalarType>(
      at::kFloat, at::kBFloat16, at::kHalf); // double is not supported

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = c10::array_of<bool (*)(
      sdp_params const&, bool)>(
      check_nested_tensor,
      check_for_dropout,
      check_tensor_shapes,
      check_batch_size_and_num_heads_dense<true /*supports GQA*/>,
      check_attn_mask_shape,
      check_nonzero_sequence_lengths_dense,
      check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>,
      check_head_dim_size_xpu,
      check_no_grad);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return check_tensor_dtype(params, supported_dtypes, debug);
}

bool can_use_flash_attention_xpu(sdp_params const& params, bool debug){
  constexpr auto supported_dtypes = c10::array_of<at::ScalarType>(
      at::kFloat, at::kBFloat16, at::kHalf); // double is not supported

  // Define gate functions that determine if a flash kernel can be run
  constexpr auto constraints = c10::array_of<bool (*)(
      sdp_params const&, bool)>(
      check_all_tensors_on_device,
      check_nested_tensor,
      check_for_dropout,
      check_tensor_shapes,
      check_for_attn_mask,
      check_batch_size_and_num_heads_dense<true /*supports GQA*/>,
      check_attn_mask_shape,
      check_nonzero_sequence_lengths_dense,
      check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim*/>,
      check_flash_causal_non_square_seqlens,
      check_head_dim_size_xpu<true /*kv_has_same_headdim*/>,
      check_no_grad);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }
  return check_tensor_dtype(params, supported_dtypes, debug);
}

SDPBackend select_sdp_backend_xpu(sdp_params const& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention
  // 2. Math fallback
  auto& ctx = at::globalContext();
  // use overrideable linked to onednn as overrideable implementation
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledOverrideableSDP() &&
      !ctx.userEnabledFlashSDP()) {
    return SDPBackend::error;
  }

  // Get ideal kernel ordering
  const std::array<SDPBackend, 3> priority_order{
      SDPBackend::flash_attention,
      SDPBackend::overrideable,
      SDPBackend::math,
  };

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : priority_order) {
    switch (backend) {
      case SDPBackend::overrideable:
        if (ctx.userEnabledOverrideableSDP() &&
            use_overrideable_xpu(kernel_params, print_debug)) {
          return SDPBackend::overrideable;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      case SDPBackend::flash_attention:
        if (ctx.userEnabledFlashSDP() &&
            can_use_flash_attention_xpu(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_overrideable_xpu did not satisfy the constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected

  print_debug = true;
  TORCH_WARN("OneDNN kernel not used because:");
  use_overrideable_xpu(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel. Aborting execution.")
  return SDPBackend::error;
}
} // namespace sdp

namespace at::native {
int64_t _fused_sdp_choice_xpu(
    const at::Tensor& query_,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  sdp::sdp_params kernel_params{
      query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
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

  at::native::onednn::gpu_float_sdpa(
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
      attn_bias,
      is_causal ? sdp::CustomMaskType::CausalFromTopLeft : sdp::CustomMaskType::NoCustomMask,
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(head_dim_qk)),
      output);

  // rng not used
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));
  return std::make_tuple(
      output,
      logsumexp,
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      seq_len_kv,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<
    Tensor, 
    Tensor, 
    Tensor, 
    Tensor,
    c10::SymInt, 
    c10::SymInt, 
    Tensor, 
    Tensor, 
    Tensor>
_scaled_dot_product_flash_attention_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_INTERNAL_ASSERT(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "scaled_dot_product_flash_attention_xpu: Accept only 4 dims inputs shape of {(B), H, T, K}");
  TORCH_INTERNAL_ASSERT(
      (key.size(0) == value.size(0)) && (key.size(1) == value.size(1)) &&
          (key.size(2) == value.size(2)),
      "scaled_dot_product_flash_attention_xpu: K/V should have the same batch / seq / num_head");
  TORCH_INTERNAL_ASSERT(
      query.size(3) == key.size(3),
      "scaled_dot_product_flash_attention_xpu: Q/K should have the same head_dim");
  TORCH_INTERNAL_ASSERT(
      query.size(1) % key.size(1) == 0,
      "scaled_dot_product_flash_attention_xpu: number of heads in K/V must divide number of heads in Q");
  TORCH_INTERNAL_ASSERT(
      dropout_p == 0.0,
      "scaled_dot_product_flash_attention_xpu: Currently do not support dropout > 0");

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

  auto opts = query.options();
  at::Tensor logsumexp =
      at::empty({batch_size, num_head_q, seq_len_q}, opts.dtype(at::kFloat));

  at::native::onednn::gpu_float_sdpa(
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
      std::nullopt,
      is_causal ? sdp::CustomMaskType::CausalFromBottomRight : sdp::CustomMaskType::NoCustomMask,
      scale.has_value() ? scale.value() : (1.0 / std::sqrt(head_dim_qk)),
      output);

  // rng not used
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));
  return std::make_tuple(
      output,
      logsumexp,
      /* cum_seq_q */ at::Tensor(),
      /* cum_seq_k */ at::Tensor(),
      seq_len_q,
      seq_len_kv,
      philox_seed,
      philox_offset,
      /*debug_attn_mask */ at::Tensor());
}

REGISTER_XPU_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_xpu);
} // namespace at::native
