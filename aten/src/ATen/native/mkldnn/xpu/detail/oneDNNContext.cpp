#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <c10/core/CPUAllocator.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>

/* *
 * Do NOT put any kernels or call any device binaries here!
 * Only maintain oneDNN runtime states in this file.
 * */
namespace at::native::onednn {

using namespace dnnl;

static inline void* dnnl_cpu_alloc(size_t size, size_t alignment) {
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  return c10_allocator->raw_allocate(size);
}

static inline void dnnl_cpu_delete(void* ptr) {
  static c10::Allocator* c10_allocator = c10::GetCPUAllocator();
  c10_allocator->raw_deallocate(ptr);
}

static inline void* dnnl_alloc(
    size_t size,
    size_t /*alignment*/,
    const void* /*dev*/,
    const void* /*context*/) {
  return c10::xpu::XPUCachingAllocator::raw_alloc(size);
}

static inline void dnnl_delete(
    void* buf,
    const void* /*dev*/,
    const void* /*context*/,
    void* /*event*/) {
  return c10::xpu::XPUCachingAllocator::raw_delete(buf);
}

CpuEngineManager::CpuEngineManager() {
  dnnl::graph::allocator host_alloc(dnnl_cpu_alloc, dnnl_cpu_delete);
  host_eng = dnnl::graph::make_engine_with_allocator(
      dnnl::engine::kind::cpu, 0, host_alloc);
}

CpuEngineManager& CpuEngineManager::Instance() {
  static CpuEngineManager myInstance;
  return myInstance;
}

GpuEngineManager::GpuEngineManager() {
  c10::DeviceIndex device_count = c10::xpu::device_count_ensure_non_zero();
  for (const auto i : c10::irange(device_count)) {
    static dnnl::graph::allocator alloc =
        dnnl::graph::sycl_interop::make_allocator(dnnl_alloc, dnnl_delete);
    engine_pool.push_back(std::make_shared<dnnl::engine>(
        dnnl::graph::sycl_interop::make_engine_with_allocator(
            c10::xpu::get_raw_device(i),
            c10::xpu::get_device_context(),
            alloc)));
  }
}

GpuEngineManager& GpuEngineManager::Instance() {
  static GpuEngineManager myInstance;
  return myInstance;
}

GpuStreamManager& GpuStreamManager::Instance() {
  static thread_local GpuStreamManager myInstance;
  return myInstance;
}

bool set_onednn_verbose(int level) {
  dnnl::status rs = dnnl::set_verbose(level);
  return rs == dnnl::status::success;
}

} // namespace at::native::onednn
