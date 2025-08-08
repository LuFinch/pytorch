#pragma once

#include <ATen/Context.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/macros/Macros.h>
#include <c10/macros/Export.h>

namespace sdp {
C10_EXPORT bool can_use_flash_attention_xpu(sdp_params const& params, bool debug);

} // namespace sdp
