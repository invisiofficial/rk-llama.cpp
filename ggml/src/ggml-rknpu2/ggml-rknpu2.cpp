#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <omp.h>

#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <random>
#include <limits>
#include <sys/mman.h>
#include <sstream>
#include <cstdio>
#include <unordered_set>
#include <atomic>
#include <cstdint>

#define UNUSED(x) (void)(x)

// F8.0 diagnostic: when RKNPU_BATCHED_MM_DIAGNOSTIC=1, log shapes that the
// supports_op rank>2 guard rejects today. Used to scope the F8 work without
// changing behavior. Each unique shape is logged once to stderr.
static bool rknpu_batched_diag_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("RKNPU_BATCHED_MM_DIAGNOSTIC");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// F8.3 gate: when RKNPU_BATCHED_MM_ENABLE=1, supports_op accepts mul_mat ops
// with src1->ne[2] > 1 (broadcast n_head_q over a rank-2 src0). Default off
// keeps the F8.1 baseline shape coverage intact for safe rollout.
static bool rknpu_batched_mm_enabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("RKNPU_BATCHED_MM_ENABLE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// F8.5 route counters: per-process atomic stats so any bench / PPL run can
// answer "where did the batched mul_mat ops actually go?" Used by the codex
// review criteria. Printed by the backend's free hook (one summary per
// backend lifetime); a diagnostic env flag also dumps them into stderr.
struct RknpuRouteStats {
    std::atomic<uint64_t> mul_mat_total{0};
    std::atomic<uint64_t> rejected_rank{0};       // ne[3]>1, etc.
    std::atomic<uint64_t> rejected_batched_off{0}; // RKNPU_BATCHED_MM_ENABLE=0
    std::atomic<uint64_t> rejected_min_m{0};
    std::atomic<uint64_t> rejected_cbuf{0};
    std::atomic<uint64_t> rejected_other{0};      // pipeline/align/contig/...
    std::atomic<uint64_t> accepted_rank2{0};
    std::atomic<uint64_t> accepted_batched{0};
    std::atomic<uint64_t> dispatched_static{0};
    std::atomic<uint64_t> dispatched_dynamic{0};
    std::atomic<uint64_t> set_tensor_calls{0};
    std::atomic<uint64_t> set_tensor_first_pipelined{0};   // first set_tensor on a pipelined alloc
    std::atomic<uint64_t> set_tensor_repeat_pipelined{0};  // subsequent calls (demote path)
    std::atomic<uint64_t> set_tensor_no_pipeline{0};       // memcpy fallback
};

static RknpuRouteStats& rknpu_route_stats() {
    static RknpuRouteStats s;
    return s;
}

static void rknpu_print_route_stats(const char* label) {
    auto& s = rknpu_route_stats();
    std::fprintf(stderr,
        "RKNPU_ROUTE_STATS %s: mul_mat_total=%lu "
        "rejected_rank=%lu rejected_batched_off=%lu rejected_min_m=%lu rejected_cbuf=%lu rejected_other=%lu "
        "accepted_rank2=%lu accepted_batched=%lu "
        "dispatched_static=%lu dispatched_dynamic=%lu "
        "set_tensor_calls=%lu set_tensor_first_pipelined=%lu set_tensor_repeat_pipelined=%lu set_tensor_no_pipeline=%lu\n",
        label,
        (unsigned long)s.mul_mat_total.load(),
        (unsigned long)s.rejected_rank.load(),
        (unsigned long)s.rejected_batched_off.load(),
        (unsigned long)s.rejected_min_m.load(),
        (unsigned long)s.rejected_cbuf.load(),
        (unsigned long)s.rejected_other.load(),
        (unsigned long)s.accepted_rank2.load(),
        (unsigned long)s.accepted_batched.load(),
        (unsigned long)s.dispatched_static.load(),
        (unsigned long)s.dispatched_dynamic.load(),
        (unsigned long)s.set_tensor_calls.load(),
        (unsigned long)s.set_tensor_first_pipelined.load(),
        (unsigned long)s.set_tensor_repeat_pipelined.load(),
        (unsigned long)s.set_tensor_no_pipeline.load());
}

// TST.3 / debug override: when RKNPU_FORCE_DYNAMIC_B=1, graph_compute
// pretends src0 is never pre-quantized, forcing the dynamic-B path
// (prepare_b_dynamic_segment) even for tensors that went through
// set_tensor's pipeline path. Used to exercise dynamic-B from the
// existing test_mul_mat infrastructure (which always allocates src0 in
// the test backend's buffer, so set_tensor marks it pre_quantized=true
// and the static path would otherwise win). Production-safe default off.
static bool rknpu_force_dynamic_b() {
    static const bool enabled = []() {
        const char* v = std::getenv("RKNPU_FORCE_DYNAMIC_B");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// F8.6b stability cap: max M_total (= M * n_heads) safe for one matmul
// submission, derived from RK3588 NPU CBUF limits. CBUF has 12 banks ×
// 32 KB; the librknnrt 2.3.2 driver appears to need at least 2 banks
// reserved for weight, leaving (12-2)*32768 bytes for data. A row of
// A occupies align_up(K, 32) * sizeof(npu_type_a) bytes, so
//   max_M = 10 * 32768 / (align_up(K, 32) * type_a_bytes).
// Crossing this segfaults inside librknnrt for the affected shape
// (observed at M_total=2048, K=512, FP16). Reference: allbilly/npu
// matmul_fix_424 branch, which solves the same limit with explicit
// M-tiling. We currently only gate; F8.6c can replace the gate with
// a real M-tile loop.
static int rknpu_max_m_for_cbuf(const rknpu2_configuration::Rknpu2HardwarePipeline* pipeline, int K_seg) {
    constexpr int CBUF_BANK_BYTES = 32768;
    constexpr int CBUF_BANKS = 12;
    constexpr int WEIGHT_BANKS_RESERVED = 2;
    const int data_banks = CBUF_BANKS - WEIGHT_BANKS_RESERVED;
    const int align_in = ((K_seg + 31) / 32) * 32;

    size_t row_bytes;
    switch (pipeline->npu_type_a) {
        case rknpu2_configuration::NPU_TYPE_INT4:
            row_bytes = (size_t)align_in / 2; // 4 bits per element, packed
            break;
        case rknpu2_configuration::NPU_TYPE_INT8:
            row_bytes = (size_t)align_in;
            break;
        default: // FP16 / FP32
            row_bytes = (size_t)align_in * 2;
            break;
    }
    if (row_bytes == 0) return std::numeric_limits<int>::max();

    const size_t cbuf_data_bytes = (size_t)data_banks * (size_t)CBUF_BANK_BYTES;
    int max_m = (int)(cbuf_data_bytes / row_bytes);
    if (max_m < 1) max_m = 1;
    return max_m;
}

// F8.6a gate: minimum M (src1->ne[1] = q_len) required to route a batched
// rank-3 mul_mat to the NPU. Decode (M=1) is dominated by per-call B prep
// overhead and runs faster on CPU; prefill / chunk passes (M >= 16) are
// where the NPU has a shot. Default 16; override with RKNPU_BATCHED_MM_MIN_M.
// Setting to 0 disables the gate entirely.
static int rknpu_batched_mm_min_m() {
    static const int value = []() {
        const char* v = std::getenv("RKNPU_BATCHED_MM_MIN_M");
        if (v == nullptr || v[0] == '\0') return 16;
        const int parsed = std::atoi(v);
        return parsed >= 0 ? parsed : 16;
    }();
    return value;
}

static void rknpu_log_batched_shape_once(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * op) {
    static std::mutex m;
    static std::unordered_set<std::string> seen;

    char key[768];
    std::snprintf(key, sizeof(key),
        "name=%s op_name=%s "
        "src0=%s[%lld,%lld,%lld,%lld]@nb[%zu,%zu,%zu,%zu] "
        "src1=%s[%lld,%lld,%lld,%lld]@nb[%zu,%zu,%zu,%zu] "
        "dst=%s[%lld,%lld,%lld,%lld]",
        op->name[0]   ? op->name   : "?",
        src0->name[0] ? src0->name : "?",
        ggml_type_name(src0->type),
        (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2], (long long)src0->ne[3],
        (size_t)src0->nb[0], (size_t)src0->nb[1], (size_t)src0->nb[2], (size_t)src0->nb[3],
        ggml_type_name(src1->type),
        (long long)src1->ne[0], (long long)src1->ne[1], (long long)src1->ne[2], (long long)src1->ne[3],
        (size_t)src1->nb[0], (size_t)src1->nb[1], (size_t)src1->nb[2], (size_t)src1->nb[3],
        ggml_type_name(op->type),
        (long long)op->ne[0], (long long)op->ne[1], (long long)op->ne[2], (long long)op->ne[3]);

    std::lock_guard<std::mutex> lock(m);
    if (seen.insert(key).second) {
        std::fprintf(stderr, "RKNPU_BATCHED_DIAG %s\n", key);
    }
}

// --- IOMMU Domain Manager ---

// Helper function for parsing complex integer lists
static std::vector<int32_t> parse_domain_list(const std::string& str) {
    std::vector<int32_t> result;
    if (str.empty()) return result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        auto dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            int start = std::strtol(token.substr(0, dash_pos).c_str(), nullptr, 10);
            int end = std::strtol(token.substr(dash_pos + 1).c_str(), nullptr, 10);
            for (int i = start; i <= end; ++i) result.push_back(i);
        } else {
            result.push_back(std::strtol(token.c_str(), nullptr, 10));
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct IOMMUDomainManager {
    std::mutex mutex;

    // Max domain size for assigning
    const size_t max_domain_size = ((size_t) std::numeric_limits<int32_t>::max() - 65536);

    // Storage for domains and their sizes
    std::unordered_map<int32_t, size_t> domain_sizes;
    std::unordered_map<int32_t, rknn_matmul_ctx> allocator_contexts;

    // Allowed domain IDs defined by the user
    std::vector<int32_t> allowed_domains;

    IOMMUDomainManager() {
        // Read restricted domains from ENV variable
        const char* env_domains = std::getenv("RKNPU_DOMAINS");
        if (env_domains != nullptr) {
            allowed_domains = parse_domain_list(env_domains);

            if (!allowed_domains.empty()) {
                fprintf(stderr, "\n"
                    "RKNPU WARNING: Custom IOMMU domains detected via RKNPU_DOMAINS.\n"
                    "Due to Rockchip library limitations, concurrent execution of\n"
                    "multiple processes accessing the NPU simultaneously WILL LEAD\n"
                    "to a SYSTEM KERNEL PANIC and WILL FREEZE YOUR OPERATING SYSTEM.\n"
                    "Execute models SEQUENTIALLY if using multiple independent processes.\n");
            }
        }
    }

    // Function for assigning the domain for the tensor of given size
    int32_t assign_domain_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Allocate strictly within the allowed domains
        if (!allowed_domains.empty()) {
            for (int32_t d : allowed_domains) {
                if (domain_sizes[d] + size <= max_domain_size) {
                    domain_sizes[d] += size;
                    ensure_allocator_context(d);
                    return d;
                }
            }

            fprintf(stderr, "RKNPU ERROR: Out of memory in allowed IOMMU domains!\n");
            assert(false);
            return -1;
        // Allocate dynamically
        } else {
            for (int32_t i = 0; i <= 15; ++i) {
                if (domain_sizes[i] + size <= max_domain_size) {
                    domain_sizes[i] += size;
                    ensure_allocator_context(i);
                    return i;
                }
            }
            fprintf(stderr, "RKNPU ERROR: Out of memory in all IOMMU domains!\n");
            assert(false);
            return -1;
        }
    }

    // Function for releasing the given size of the domain memory
    void release_domain_memory(int32_t domain_id, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = domain_sizes.find(domain_id);
        if (it != domain_sizes.end()) {
            if (it->second >= size) {
                it->second -= size;
            } else {
                it->second = 0;
            }
        }
    }

    // Function for getting a new dummy context in the required domain
    rknn_matmul_ctx get_allocator_context(int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);
        ensure_allocator_context(domain_id);
        return allocator_contexts[domain_id];
    }

private:
    // Function for ensuring a dummy context existence in the required domain
    void ensure_allocator_context(int32_t domain_id) {
        if (allocator_contexts.find(domain_id) == allocator_contexts.end()) {
            rknn_matmul_info info;
            memset(&info, 0, sizeof(info));
            info.M = 32; info.K = 32; info.N = 32;
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
            info.iommu_domain_id = domain_id;

            rknn_matmul_io_attr io_attr;
            rknn_matmul_ctx ctx = 0;
            rknn_matmul_create(&ctx, &info, &io_attr);
            allocator_contexts[domain_id] = ctx;
        }
    }
};
static IOMMUDomainManager g_domain_manager;

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,         \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

// Matrix segment information for N dimension
struct MatrixSegmentN {
    int offset_n;
    int size_n;
    int core_id;
};

// Matrix segment information for K dimension
struct MatrixSegmentK {
    int offset_k;
    int size_k;
};

// Split B-matrix into N-segments for cores
static std::vector<MatrixSegmentN> compute_n_segments(int N, const std::vector<int>& active_cores, int alignment) {
    std::vector<MatrixSegmentN> segments;
    int num_cores = active_cores.size();

    if (num_cores == 0) return segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);

    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegmentN seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = active_cores[i];

        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }

        offset += seg.size_n;
        segments.push_back(seg);
    }
    return segments;
}

// Split B-matrix into K-segments for hardware limit
static std::vector<MatrixSegmentK> compute_k_segments(int K_op, int k_limit, int alignment) {
    std::vector<MatrixSegmentK> segments;

    if (k_limit <= 0 || K_op <= k_limit) {
        segments.push_back({0, K_op});
        return segments;
    }

    int k_limit_aligned = (k_limit / alignment) * alignment;
    int offset = 0;
    while (offset < K_op) {
        int size = std::min(k_limit_aligned, K_op - offset);
        segments.push_back({offset, size});
        offset += size;
    }
    return segments;
}

// --- Structs ---

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    void* virtual_base;
    size_t total_size;
    std::string name;

    // RKNN buffers allocations for each tensor
    struct TensorAllocation {
        rknn_tensor_mem* mem = nullptr;
        size_t size = 0;
        int32_t iommu_domain_id = 0;
        // F8.3: init_tensor creates an alloc for any pipelined tensor; only
        // set_tensor's pipeline path actually populates the DMA buffer.
        // graph_compute uses pre_quantized to gate the static fast path.
        bool pre_quantized = false;
        // F8.3: count of set_tensor invocations on this alloc. Real weights
        // see exactly one (load-time). Scheduler-managed copy tensors (e.g.
        // F16 KV cache views the sched dups into the RKNPU buffer for batched
        // attention) see one per inference step. We pre-quantize on the first
        // call and demote to memcpy on any subsequent call.
        int set_count = 0;
    };
    std::unordered_map<size_t, TensorAllocation> tensor_allocs;

    // Per-block scaling factors for quantized weights
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    std::mutex mutex;

    // Function for the allocation of a RKNN buffer for the individual tensor
    TensorAllocation get_tensor_allocation(size_t tensor_offset, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Trying to find an existing buffer
        auto it = tensor_allocs.find(tensor_offset);
        if (it != tensor_allocs.end()) {
            if (it->second.size < size) {
                rknn_matmul_ctx old_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                rknn_destroy_mem(old_ctx, it->second.mem);
                g_domain_manager.release_domain_memory(it->second.iommu_domain_id, it->second.size);

                it->second.iommu_domain_id = g_domain_manager.assign_domain_memory(size);
                rknn_matmul_ctx new_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                it->second.mem = rknn_create_mem(new_ctx, size);
                it->second.size = size;
            }
            return it->second;
        }

        // Acquiring a domain for allocation
        int32_t domain_id = g_domain_manager.assign_domain_memory(size);
        rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(domain_id);

        // Allocating a new buffer for the tensor
        TensorAllocation alloc;
        alloc.mem = rknn_create_mem(alloc_ctx, size);
        alloc.size = size;
        alloc.iommu_domain_id = domain_id;

        GGML_ASSERT(alloc.mem != nullptr && "Failed to allocate tensor memory via RKNN API");
        tensor_allocs[tensor_offset] = alloc;

        return alloc;
    }
};


// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    bool b_bound = false;
    std::shared_ptr<rknn_tensor_mem> mem_B;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type, int32_t domain_id) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        info.iommu_domain_id = domain_id;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) ctx = 0;
    }

    ~rknpu_matmul_context() {
        mem_B.reset();

        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};

// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache (tensor_fd, offset, M, K, N, core_id, type, domain_id)
    std::unordered_map<std::tuple<uintptr_t, size_t, int, int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // A-matrices cache (M, K, npu_type_a, domain_id)
    std::unordered_map<std::tuple<int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;

    // C-matrices cache (M, N, core_id, npu_type_c, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    // F8.2 dynamic-B path: matmul ctx cache keyed only by shape (no tensor_id/offset),
    // since for dynamic B we re-bind the B memory on every call.
    std::unordered_map<std::tuple<int, int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> dynamic_matmul_ctx_cache;

    // F8.2 dynamic-B path: per-shape DMA buffer cache for B (re-quantized in-place per call).
    std::unordered_map<std::tuple<int, int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> b_dynamic_buffer_cache;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(uintptr_t tensor_id, size_t offset, int M, int K, int N, int core_id, rknn_matmul_type type, int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(tensor_id, offset, M, K, N, core_id, (int)type, (int)domain_id);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }

        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        matmul_ctx_cache[key] = ctx;
        return ctx;
    }

    // F8.2 dynamic-B variant of get_matmul_ctx. Same RKNN ctx setup as the
    // static path, but cached purely by shape so different src0 tensors that
    // share the same (M, K, N, core, type, domain) reuse the same context.
    std::shared_ptr<rknpu_matmul_context> get_dynamic_matmul_ctx(int M, int K, int N, int core_id, rknn_matmul_type type, int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(M, K, N, core_id, (int)type, (int)domain_id);
        auto it = dynamic_matmul_ctx_cache.find(key);
        if (it != dynamic_matmul_ctx_cache.end()) {
            return it->second;
        }

        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch (core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        dynamic_matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};


// F8.2 forward declarations: helpers for the dynamic-B path live further down
// the file (next to the matching set_tensor helpers); graph_compute needs them
// before they are defined.
static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft);
static void dequantize_row(
    const struct ggml_tensor * tensor, const void * raw_data,
    int n, int K, float * row_out);
static void pack_native(
    uint8_t* dst, const uint8_t* src,
    int K_total, int k_offset, int k_segment, int k_align,
    int N_total, int n_offset, int n_segment, int n_align,
    int element_bits);
static float prepare_b_dynamic_segment(
    const struct ggml_tensor * src0, const void * src0_raw,
    int K, int N, int K_op,
    const struct MatrixSegmentK & k_seg,
    const struct MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline,
    const float * s_vec_data,
    uint8_t * dst_dma_ptr);

// F8.2: identify whether a tensor is backed by the RKNPU buffer type. We use
// the static get_name function pointer as the discriminator — comparing to a
// local-static buffer_type instance would require exposing it across functions.
static bool is_rknpu_buffer(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer || !tensor->buffer->buft) return false;
    return tensor->buffer->buft->iface.get_name == ggml_backend_rknpu_buffer_type_get_name;
}

//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    // F8.5 route counter dump: print one summary per backend lifetime so
    // benches/PPL runs can verify what actually ran on the NPU vs CPU.
    rknpu_print_route_stats("backend_free");
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;

    // Cleanup ordering fix: a/c/b buffer caches all hold shared_ptr<rknn_tensor_mem>
    // whose deleters captured the rknn_matmul_ctx handle that allocated each mem.
    // The handles must still be valid when the deleters fire. With both static
    // and dynamic paths populating c_buffer_cache and a_buffer_cache, those
    // caches contain mems whose captured handle came from EITHER matmul_ctx_cache
    // OR dynamic_matmul_ctx_cache. Destroying ctx caches first leaves stale
    // handles in the buffer caches, which segfaults inside rknn_destroy_mem.
    // Solution: clear ALL buffer caches before ANY ctx cache.
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->b_dynamic_buffer_cache.clear();
        ctx->c_buffer_cache.clear();
        ctx->a_buffer_cache.clear();
        ctx->dynamic_matmul_ctx_cache.clear();
        ctx->matmul_ctx_cache.clear();
    }

    delete ctx;
    delete backend;
}

// F8.5.STAT — debug helper to inspect per-buffer set_count distribution at
// buffer free time. Shows how many tensors went through set_tensor>=2
// times (the demote path), and the max observed count. Helps locate why
// dispatched_dynamic stays 0 in production despite accepted_batched > 0.
static void rknpu_log_buffer_set_count_histogram(const ggml_backend_rknpu_buffer_context* ctx, const char* label) {
    if (!ctx) return;
    int total_allocs = 0;
    int set_count_zero = 0;
    int set_count_one = 0;
    int set_count_ge_two = 0;
    int max_set_count = 0;
    int pre_quantized = 0;
    for (const auto& pair : ctx->tensor_allocs) {
        ++total_allocs;
        const int sc = pair.second.set_count;
        if (sc == 0) ++set_count_zero;
        else if (sc == 1) ++set_count_one;
        else ++set_count_ge_two;
        if (sc > max_set_count) max_set_count = sc;
        if (pair.second.pre_quantized) ++pre_quantized;
    }
    std::fprintf(stderr,
        "RKNPU_BUFFER_FREE %s: name=%s allocs=%d set_count{0=%d, 1=%d, >=2=%d} max=%d pre_quantized_now=%d\n",
        label,
        ctx->name.c_str(),
        total_allocs, set_count_zero, set_count_one, set_count_ge_two, max_set_count, pre_quantized);
}

// Function for acquiring a pointer for tensor data
static void* get_tensor_real_ptr(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nullptr;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        auto* ctx = (ggml_backend_rknpu_buffer_context*)tensor->buffer->context;
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->tensor_allocs.find(offset);
        if (it != ctx->tensor_allocs.end()) {
            return it->second.mem->virt_addr;
        }
    }

    return tensor->data;
}

// Function for getting buffer from cache or creating new one
template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_tensor_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->size >= size) {
            return it->second;
        }
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx, size);
    if (!mem) { return nullptr; }

    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m != 0) {
            rknn_destroy_mem(matmul_ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (node->op != GGML_OP_MUL_MAT) continue;

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        // Skipping zero-dimension matmuls
        if (M == 0 || K == 0 || N == 0) {
            continue;
        }

        // F8.3: dst (and src1) may carry a head dim ne[2] that broadcasts a
        // single rank-2 src0 across n_heads sub-matmuls. n_heads=1 collapses
        // to the original single-matmul path. Detect it early so M_op below
        // already reflects the flattened head count (F8.6b).
        const int n_heads = (int)dst->ne[2];
        GGML_ASSERT(n_heads >= 1 && "dst->ne[2] must be >= 1");

        // F8.6b: flatten heads. Run a single NPU matmul per (k_seg, n_seg)
        // with M_total = M * n_heads rows of A, instead of one matmul per
        // head. The NPU sees a bigger A and produces a bigger C in one
        // submission; we scatter rows back to dst[h] in the collect step.
        // For n_heads=1 (static path / non-batched dynamic) M_total == M and
        // the routine is unchanged.
        const int M_total = M * n_heads;
        int M_op = M_total;
        if (M_total > 1) {
            M_op = rknpu2_calibration::next_power_of_two(M_total);
        }

        const auto* pipeline = config.resolve_op_support(src0);
        if (!pipeline) continue;

        // Initializing Hadamard Transform Logic
        const bool is_hadamard = (pipeline->use_hadamard);
        const int K_op = is_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        const rknn_matmul_type matmul_type = pipeline->mm_type;
        const int alignment = pipeline->n_align;

        // Computing specific hardware segments
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto all_k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto all_n_segments = compute_n_segments(N, config.active_cores, alignment);

        std::vector<MatrixSegmentN> active_n_segments;
        for (const auto& seg : all_n_segments) {
            if (seg.size_n > 0) active_n_segments.push_back(seg);
        }

        if (active_n_segments.empty()) continue;

        // Initializing variables
        const size_t num_active_segments = active_n_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        // F8.2/F8.3 path discriminator: src0 takes the static (FD-mapped, pre-
        // quantized) path only when an RKNPU alloc record exists for it.
        // Otherwise (CPU-resident src0, or RKNPU-resident but not pre-quantized
        // — e.g. K/V copied in by the scheduler for batched attention), we use
        // the dynamic path that quantizes/packs src0 inline.
        ggml_backend_rknpu_buffer_context* src0_buf_ctx = nullptr;
        size_t tensor_offset_in_virtual = 0;
        bool src0_is_pre_quantized = false;
        int32_t b_domain_id = 0;
        int tensor_fd = -1;
        void* tensor_virt_addr = nullptr;
        if (is_rknpu_buffer(src0)) {
            src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0->buffer->context;
            tensor_offset_in_virtual = (uintptr_t)src0->data - (uintptr_t)src0_buf_ctx->virtual_base;

            std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
            auto it = src0_buf_ctx->tensor_allocs.find(tensor_offset_in_virtual);
            if (it != src0_buf_ctx->tensor_allocs.end()) {
                b_domain_id = it->second.iommu_domain_id;
                src0_is_pre_quantized = it->second.pre_quantized;
                tensor_fd = it->second.mem->fd;
                tensor_virt_addr = it->second.mem->virt_addr;
            }
        }
        const bool use_static_path = src0_is_pre_quantized && !rknpu_force_dynamic_b();

        // F8.5 route counter: which path actually ran the matmul on the NPU.
        if (use_static_path) {
            rknpu_route_stats().dispatched_static.fetch_add(1, std::memory_order_relaxed);
        } else {
            rknpu_route_stats().dispatched_dynamic.fetch_add(1, std::memory_order_relaxed);
        }

        const size_t src1_head_stride = (n_heads > 1) ? (src1->nb[2] / sizeof(float)) : 0;
        const size_t dst_head_stride  = (n_heads > 1) ? (dst->nb[2]  / sizeof(float)) : 0;

        // Cleaning the C-matrix buffer (per head, since dst may be non-contig
        // along ne[2] for permuted attention outputs).
        float* dst_data = (float*)get_tensor_real_ptr(dst);
        for (int h = 0; h < n_heads; ++h) {
            float* dst_h = dst_data + (size_t)h * dst_head_stride;
            memset(dst_h, 0, (size_t)M * N * sizeof(float));
        }

        // Acquiring the Hadamard vector
        std::vector<float> s_vec;
        if (is_hadamard) {
            if (use_static_path) {
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                auto it = src0_buf_ctx->hadamard_s_vectors.find(src0);
                GGML_ASSERT(it != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
                s_vec = it->second;
            } else {
                // Dynamic path: regenerate deterministically from src0 ptr,
                // matching the seed scheme used in buffer_set_tensor.
                s_vec.assign(K_op, 1.0f);
                std::mt19937 gen(reinterpret_cast<uintptr_t>(src0));
                std::uniform_int_distribution<int> distrib(0, 1);
                for (int k = 0; k < K_op; ++k) {
                    s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
                }
            }
        }

        // Calculating the B-matrix scale. Static path looks the grid up from
        // the buffer ctx; dynamic path fills it inline below as we quantize
        // each (k_seg, n_seg) block.
        std::vector<float> scales_B_grid;
        if (use_static_path) {
            if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                auto it = src0_buf_ctx->quantized_tensor_scales.find(src0);
                GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scales grid not found");
                scales_B_grid = it->second;
            }
        } else {
            scales_B_grid.assign(all_k_segments.size() * num_active_segments, 1.0f);
        }

        // Calculating tensor packed size (only used by the static FD-mapped layout)
        size_t type_size_packed = 0;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
        else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;

        // Computing K dimensions segments
        size_t current_offset_in_tensor = 0;
        for (size_t k_idx = 0; k_idx < all_k_segments.size(); ++k_idx) {
            const auto& k_seg = all_k_segments[k_idx];
            const int K_seg_op = k_seg.size_k;

            // ===========================================
            // ========== 1. Preparing Contexts ==========
            // ===========================================
            for (const auto& n_seg : all_n_segments) {
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    if (active_n_segments[idx].offset_n != n_seg.offset_n) continue;

                    if (use_static_path) {
                        // Static path: B was pre-quantized + packed at set_tensor
                        // time and lives at a known offset inside the FD-mapped
                        // tensor blob; bind once via create_mem_from_fd.
                        size_t offset_in_dma = current_offset_in_tensor;

                        matmul_ctxs[idx] = backend_ctx->get_matmul_ctx(
                            (uintptr_t)tensor_virt_addr, offset_in_dma, M_op, K_seg_op, n_seg.size_n,
                            n_seg.core_id, matmul_type, b_domain_id
                        );
                        if (!matmul_ctxs[idx] || matmul_ctxs[idx]->ctx == 0) return GGML_STATUS_FAILED;

                        auto& matmul_ctx = matmul_ctxs[idx];

                        if (!matmul_ctx->b_bound) {
                            size_t segment_size_bytes = matmul_ctx->io_attr.B.size;

                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(
                                matmul_ctx->ctx,
                                tensor_fd,
                                tensor_virt_addr,
                                segment_size_bytes,
                                offset_in_dma
                            );
                            if (!mem) return GGML_STATUS_FAILED;

                            auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                            matmul_ctx->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);

                            RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->mem_B.get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");

                            matmul_ctx->b_bound = true;
                        }
                    } else {
                        // F8.2 dynamic-B path: src0 is a runtime tensor (e.g.
                        // K/V from CPU backend in attention K·Q^T), not a
                        // pre-quantized weight. Reuse a shape-keyed matmul ctx
                        // and a shape-keyed DMA buffer, but re-fill the B
                        // contents and re-bind every call.
                        const int32_t domain_id = b_domain_id;

                        matmul_ctxs[idx] = backend_ctx->get_dynamic_matmul_ctx(
                            M_op, K_seg_op, n_seg.size_n,
                            n_seg.core_id, matmul_type, domain_id
                        );
                        if (!matmul_ctxs[idx] || matmul_ctxs[idx]->ctx == 0) return GGML_STATUS_FAILED;

                        auto& matmul_ctx = matmul_ctxs[idx];

                        auto b_key = std::make_tuple(
                            M_op, K_seg_op, n_seg.size_n, n_seg.core_id,
                            (int)pipeline->npu_type_b, (int)domain_id);
                        auto mem_B = get_tensor_buffer(
                            backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.B.size,
                            b_key, backend_ctx->b_dynamic_buffer_cache);
                        if (!mem_B) return GGML_STATUS_FAILED;
                        matmul_ctx->mem_B = mem_B;

                        const float * s_vec_ptr = is_hadamard ? s_vec.data() : nullptr;
                        const float block_scale = prepare_b_dynamic_segment(
                            src0, src0->data,
                            K, N, K_op,
                            k_seg, n_seg, pipeline,
                            s_vec_ptr,
                            (uint8_t*)mem_B->virt_addr);
                        scales_B_grid[k_idx * num_active_segments + idx] = block_scale;

                        RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B.get(), &matmul_ctx->io_attr.B), "set_io_mem B dynamic");
                        RKNN_CHECK(rknn_mem_sync(matmul_ctx->ctx, mem_B.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync B dynamic TO_DEVICE");
                    }
                    break;
                }

                if (use_static_path && n_seg.size_n > 0) {
                    current_offset_in_tensor += type_size_packed > 0 ? (size_t)n_seg.size_n * K_seg_op * type_size_packed : (size_t)n_seg.size_n * K_seg_op / 2;
                }
            }

            // F8.6b: heads are flattened into M_total = M * n_heads rows of A.
            // One NPU matmul per (k_seg, n_seg) handles all heads at once;
            // collect scatters row (h*M + m) of C back to dst[h][m]. For
            // n_heads=1 this is identical to the original single-matmul path.
            if (rknpu_batched_diag_enabled() && n_heads > 1 && k_idx == 0) {
                static std::mutex m;
                static std::unordered_set<std::string> seen;
                char key[512];
                const char* buf_name = (src0->buffer && src0->buffer->buft && src0->buffer->buft->iface.get_name)
                    ? src0->buffer->buft->iface.get_name(src0->buffer->buft) : "?";
                std::snprintf(key, sizeof(key),
                    "ROUTED static=%d in_rknpu=%d pre_q=%d M=%d K=%d N=%d n_heads=%d M_total=%d src0=%s/%s/buft=%s/op=%d/view_src=%p/flags=0x%x dst=%s op=%s",
                    use_static_path ? 1 : 0, is_rknpu_buffer(src0) ? 1 : 0, src0_is_pre_quantized ? 1 : 0,
                    M, K, N, n_heads, M_total,
                    ggml_type_name(src0->type),
                    src0->name[0] ? src0->name : "?",
                    buf_name,
                    (int)src0->op,
                    (void*)src0->view_src,
                    (unsigned)src0->flags,
                    ggml_type_name(dst->type),
                    node->name[0] ? node->name : "?");
                std::lock_guard<std::mutex> lock(m);
                if (seen.insert(key).second) {
                    std::fprintf(stderr, "RKNPU_BATCHED_ROUTED %s\n", key);
                }
            }
            const float* x_base = (const float*)get_tensor_real_ptr(src1);

            // ===========================================
            // ========== 2. Preparing A-matrix ==========
            // ===========================================
            std::vector<float> scales_A(M_total, 1.0f);
            {
                auto cache_key = std::make_tuple(M_op, K_seg_op, (int)pipeline->npu_type_a, b_domain_id);
                auto& matmul_ctx_0 = matmul_ctxs[0];

                // Getting A-buffer from cache (sized M_op × K_seg_op via matmul ctx)
                mem_A_shared = get_tensor_buffer(backend_ctx, matmul_ctx_0->ctx, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
                if (!mem_A_shared) return GGML_STATUS_FAILED;

                const int row_stride = (int)(src1->nb[1] / sizeof(float));
                void* dst_base = mem_A_shared->virt_addr;

                #pragma omp parallel for
                for (int row_idx = 0; row_idx < M_total; ++row_idx) {
                    const int h = (n_heads > 1) ? (row_idx / M) : 0;
                    const int m = (n_heads > 1) ? (row_idx - h * M) : row_idx;
                    const float* src_row = x_base
                        + (size_t)h * src1_head_stride
                        + (size_t)m * row_stride;
                    std::vector<float> ready_row(K_seg_op);

                    // Applying Hadamard Transform
                    if (is_hadamard) {
                        std::vector<float> signed_row(K);
                        std::vector<float> full_hadamard_row(K_op);
                        for(int k=0; k<K; ++k) signed_row[k] = src_row[k] * s_vec[k];
                        rknpu2_calibration::hadamard_transform(full_hadamard_row.data(), signed_row.data(), K, K_op);

                        memcpy(ready_row.data(), full_hadamard_row.data() + k_seg.offset_k, K_seg_op * sizeof(float));
                    } else {
                        memcpy(ready_row.data(), src_row + k_seg.offset_k, K_seg_op * sizeof(float));
                    }

                    // Handling types and quantizations (write to A row `row_idx`)
                    if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                        uint16_t* dst_ptr = (uint16_t*)dst_base;
                        uint16_t* dst_row = dst_ptr + (size_t)row_idx * K_seg_op;
                        rknpu2_quantization::convert_fp32_to_fp16(ready_row.data(), dst_row, K_seg_op);
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                        float amax_m = 0.0f;
                        for (int k = 0; k < K_seg_op; ++k) amax_m = std::max(amax_m, std::abs(ready_row[k]));
                        scales_A[row_idx] = amax_m / 127.0f;

                        int8_t* dst_ptr = (int8_t*)dst_base;
                        int8_t* dst_row = dst_ptr + (size_t)row_idx * K_seg_op;
                        rknpu2_quantization::quantize_fp32_to_int8(ready_row.data(), dst_row, K_seg_op, scales_A[row_idx]);
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                        float amax_m = 0.0f;
                        for (int k = 0; k < K_seg_op; ++k) amax_m = std::max(amax_m, std::abs(ready_row[k]));
                        scales_A[row_idx] = amax_m / 7.0f;

                        uint8_t* dst_ptr = (uint8_t*)dst_base;
                        uint8_t* dst_row = dst_ptr + (size_t)row_idx * (K_seg_op / 2);
                        rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row.data(), dst_row, K_seg_op, scales_A[row_idx]);
                    }
                }

                // Assigning A-matrix to all contexts for the parallel execution
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[idx]->ctx, mem_A_shared.get(), &matmul_ctxs[idx]->io_attr.A), "set_io_mem A for core");
                }

                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[0]->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");
            }

            // ===========================================
            // ========== 3. Preparing C-matrix ==========
            // ===========================================
            {
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    auto& matmul_ctx = matmul_ctxs[idx];
                    auto cache_key = std::make_tuple(M_op, active_n_segments[idx].size_n, active_n_segments[idx].core_id, (int)pipeline->npu_type_c, b_domain_id);

                    // Getting C-buffer from cache
                    mem_C_segments[idx] = get_tensor_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                    if (!mem_C_segments[idx]) return GGML_STATUS_FAILED;

                    // Assigning C-matrix to current context for the parallel execution
                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[idx].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
                }
            }

            // ==========================================
            // ========== 4. Running operation ==========
            // ==========================================
            {
                #pragma omp parallel for num_threads(num_active_segments)
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    int ret = rknn_matmul_run(matmul_ctxs[idx]->ctx);
                    if (ret != RKNN_SUCC) {
                        // Handle error
                    }
                }
            }

            // ===========================================
            // ========== 5. Collecting results ==========
            // ===========================================
            {
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_C_segments[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
                }

                const float hadamard_divisor = pipeline->use_hadamard ? (float)K_op : 1.0f;

                #pragma omp parallel for
                for (int row_idx = 0; row_idx < M_total; row_idx++) {
                    const int h = (n_heads > 1) ? (row_idx / M) : 0;
                    const int m = (n_heads > 1) ? (row_idx - h * M) : row_idx;
                    float* dst_data_h = dst_data + (size_t)h * dst_head_stride;

                    // Handling types and quantizations
                    switch (pipeline->npu_type_c) {
                        case rknpu2_configuration::NPU_TYPE_FP32: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                float scale_B = scales_B_grid.empty() ? 1.0f : scales_B_grid[k_idx * num_active_segments + idx];
                                float dequant_scale = (scales_A[row_idx] * scale_B) / hadamard_divisor;

                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                float* src_segment_base = (float*)mem_C_segments[idx]->virt_addr;
                                float* dst_ptr = dst_data_h + (size_t)m * N + N_offset;
                                float* src_ptr = src_segment_base + (size_t)row_idx * N_segment;

                                for(int n=0; n<N_segment; ++n) {
                                    dst_ptr[n] += src_ptr[n] * dequant_scale;
                                }
                            }
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT32: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                float scale_B = scales_B_grid.empty() ? 1.0f : scales_B_grid[k_idx * num_active_segments + idx];
                                float dequant_scale = (scales_A[row_idx] * scale_B) / hadamard_divisor;

                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                float* dst_ptr = dst_data_h + (size_t)m * N + N_offset;
                                int32_t* src_ptr = (int32_t*)mem_C_segments[idx]->virt_addr + (size_t)row_idx * N_segment;

                                for(int n=0; n<N_segment; ++n) {
                                    dst_ptr[n] += (float)src_ptr[n] * dequant_scale;
                                }
                            }
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT16: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                float scale_B = scales_B_grid.empty() ? 1.0f : scales_B_grid[k_idx * num_active_segments + idx];
                                float dequant_scale = (scales_A[row_idx] * scale_B) / hadamard_divisor;

                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                float* dst_ptr = dst_data_h + (size_t)m * N + N_offset;
                                int16_t* src_ptr = (int16_t*)mem_C_segments[idx]->virt_addr + (size_t)row_idx * N_segment;

                                for(int n=0; n<N_segment; ++n) {
                                    dst_ptr[n] += (float)src_ptr[n] * dequant_scale;
                                }
                            }
                            break;
                        }

                        default:
                            // This should not be reached if config is correct
                            break;
                    }
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

// Function for calculating a real tensor size for the NPU
static size_t get_tensor_packed_size(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        size_t total_size = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& seg : n_segments) {
                if (seg.size_n > 0) {
                    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                        total_size += (size_t)seg.size_n * k_seg.size_k / 2;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
                        total_size += (size_t)seg.size_n * k_seg.size_k;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
                        total_size += (size_t)seg.size_n * k_seg.size_k * 2;
                    }
                }
            }
        }
        return total_size;
    }
    return ggml_nbytes(tensor);
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    if (rknpu_batched_diag_enabled()) {
        rknpu_log_buffer_set_count_histogram(ctx, "free_buffer");
    }

    // Freeing an every individual RKNN buffer using the allocator context
    for (auto& pair : ctx->tensor_allocs) {
        if (pair.second.mem) {
            rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(pair.second.iommu_domain_id);
            rknn_destroy_mem(alloc_ctx, pair.second.mem);
            g_domain_manager.release_domain_memory(pair.second.iommu_domain_id, pair.second.size);
        }
    }

    // Freeing the virtual memory block
    munmap(ctx->virtual_base, ctx->total_size);

    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->virtual_base;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    // Initialize tensor only if it is supported by the pipeline. The DMA
    // buffer is owned by the alloc; whether it actually holds pre-quantized
    // bytes is decided in set_tensor (first call only).
    if (pipeline) {
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;
        size_t size = get_tensor_packed_size(tensor);
        ctx->get_tensor_allocation(offset, size);
    }

    return GGML_STATUS_SUCCESS;
}

// Function for dequantizing a single row from GGUF format to FP32
static void dequantize_row(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int n, int K,
    float * row_out)
{
    if (tensor->type == GGML_TYPE_F32) {
        const float* src = (const float*)raw_data;
        memcpy(row_out, src + (size_t)n * K, K * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)raw_data;
        const ggml_fp16_t* src_row = src + (size_t)n * K;
        for (int k = 0; k < K; ++k) row_out[k] = ggml_fp16_to_fp32(src_row[k]);
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        const block_q8_0* src = (const block_q8_0*)raw_data;
        dequantize_row_q8_0(src + (size_t)n * (K / QK8_0), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        const block_q6_K* src = (const block_q6_K*)raw_data;
        dequantize_row_q6_K(src + (size_t)n * (K / QK_K), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        const block_q4_0* src = (const block_q4_0*)raw_data;
        dequantize_row_q4_0(src + (size_t)n * (K / QK4_0), row_out, K);
    } else {
        GGML_ASSERT(false && "Unsupported weight type for NPU pipeline");
    }
}

// Function for extracting a specific tensor segment and converting it to FP32
static void dequantize_tensor_segment(
    std::vector<float>& out_segment,
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const void * raw_data,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    bool use_hadamard)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
    out_segment.resize(seg_elements);

    std::vector<float> s_vec;
    if (use_hadamard) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        s_vec = ctx->hadamard_s_vectors[tensor];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_seg.size_n; ++i) {
        int global_n = n_seg.offset_n + i;

        if (global_n < N) {
            std::vector<float> row_raw(K);
            std::vector<float> row_processed(K_op, 0.0f);

            dequantize_row(tensor, raw_data, global_n, K, row_raw.data());

            if (use_hadamard) {
                std::vector<float> signed_row(K);
                for (int k = 0; k < K; ++k) signed_row[k] = row_raw[k] * s_vec[k];
                rknpu2_calibration::hadamard_transform(row_processed.data(), signed_row.data(), K, K_op);
            } else {
                memcpy(row_processed.data(), row_raw.data(), K * sizeof(float));
            }

            memcpy(&out_segment[i * k_seg.size_k], &row_processed[k_seg.offset_k], k_seg.size_k * sizeof(float));
        } else {
            memset(&out_segment[i * k_seg.size_k], 0, k_seg.size_k * sizeof(float));
        }
    }
}

// Function for quantizing the FP32 segment to the target NPU format
static void quantize_tensor_segment(
    const std::vector<float>& fp32_segment,
    std::vector<uint8_t>& out_quantized,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    float scale,
    rknpu2_configuration::Rknpu2NpuType npu_type)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;

    if (npu_type == rknpu2_configuration::NPU_TYPE_FP16) {
        out_quantized.resize(seg_elements * 2);
        rknpu2_quantization::convert_fp32_to_fp16(
            fp32_segment.data(),
            (uint16_t*)out_quantized.data(),
            seg_elements);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT8) {
        out_quantized.resize(seg_elements);
        rknpu2_quantization::quantize_fp32_to_int8(
            fp32_segment.data(),
            (int8_t*)out_quantized.data(),
            seg_elements,
            scale);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT4) {
        out_quantized.resize(seg_elements / 2);
        rknpu2_quantization::quantize_fp32_to_int4_packed(
            fp32_segment.data(),
            out_quantized.data(),
            seg_elements,
            scale);
    }
}

// Function for packing
static void pack_native(
    uint8_t* dst, const uint8_t* src,
    int K_total, int k_offset, int k_segment, int k_align,
    int N_total, int n_offset, int n_segment, int n_align,
    int element_bits)
{
    UNUSED(N_total);

    GGML_ASSERT(k_segment % k_align == 0 && "k_segment must be aligned to k_align");
    GGML_ASSERT(n_segment % n_align == 0 && "n_segment must be aligned to n_align");

    const size_t k_sub_bytes     = (size_t)k_align * element_bits / 8;
    const size_t src_row_bytes  = (size_t)K_total * element_bits / 8;
    const size_t n_blocks       = n_segment / n_align;
    const size_t k_blocks       = k_segment / k_align;
    const size_t kblock_stride  = (size_t)n_align * k_sub_bytes;
    const size_t nblock_stride  = k_blocks * kblock_stride;

    for (size_t ni = 0; ni < n_blocks; ++ni) {
        for (size_t ki = 0; ki < k_blocks; ++ki) {
            uint8_t* dst_tile = dst + ni * nblock_stride + ki * kblock_stride;

            for (int nn = 0; nn < n_align; ++nn) {
                const size_t n_global = (size_t)n_offset + ni * n_align + nn;
                const size_t k_start  = (size_t)k_offset + ki * k_align;

                const uint8_t* src_ptr = src + n_global * src_row_bytes
                                             + k_start * element_bits / 8;
                uint8_t* dst_ptr = dst_tile + nn * k_sub_bytes;

                size_t off = 0;
                for (; off + 16 <= k_sub_bytes; off += 16) {
                    vst1q_u8(dst_ptr + off, vld1q_u8(src_ptr + off));
                }
                for (; off < k_sub_bytes; ++off) {
                    dst_ptr[off] = src_ptr[off];
                }
            }
        }
    }
}

// Function for packing the quantized segment into the native NPU layout and writing to DMA
static size_t pack_tensor_segment(
    const std::vector<uint8_t>& quantized_segment,
    uint8_t * dst_dma_ptr,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline)
{
    int element_bits = 0;
    size_t segment_packed_size = 0;

    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
        element_bits = 16;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k * 2;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
        element_bits = 8;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
        element_bits = 4;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k / 2;
    }

    pack_native(dst_dma_ptr, quantized_segment.data(),
                k_seg.size_k, 0, k_seg.size_k, pipeline->k_align,
                n_seg.size_n, 0, n_seg.size_n, pipeline->n_align,
                element_bits);

    return segment_packed_size;
}

// F8.2 dynamic-B helper: dequantize one (k_seg, n_seg) tile from src0 directly
// (no buffer-context lookup), apply Hadamard if requested, compute the block
// scale, quantize and pack into the supplied DMA buffer. Returns the block
// scale so the caller can record it in scales_B_grid for dequantization in
// the result-collection stage. Mirrors the math in buffer_set_tensor so the
// dynamic and static paths stay numerically equivalent.
static float prepare_b_dynamic_segment(
    const struct ggml_tensor * src0, const void * src0_raw,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline,
    const float * s_vec_data,
    uint8_t * dst_dma_ptr)
{
    const bool use_hadamard = (s_vec_data != nullptr);
    const size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;

    std::vector<float> seg_fp32(seg_elements, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < n_seg.size_n; ++i) {
        const int global_n = n_seg.offset_n + i;
        if (global_n >= N) {
            std::memset(&seg_fp32[(size_t)i * k_seg.size_k], 0, k_seg.size_k * sizeof(float));
            continue;
        }

        std::vector<float> row_raw(K);
        std::vector<float> row_processed(K_op, 0.0f);

        dequantize_row(src0, src0_raw, global_n, K, row_raw.data());

        if (use_hadamard) {
            std::vector<float> signed_row(K);
            for (int k = 0; k < K; ++k) signed_row[k] = row_raw[k] * s_vec_data[k];
            rknpu2_calibration::hadamard_transform(row_processed.data(), signed_row.data(), K, K_op);
        } else {
            std::memcpy(row_processed.data(), row_raw.data(), K * sizeof(float));
        }

        std::memcpy(&seg_fp32[(size_t)i * k_seg.size_k],
                    &row_processed[k_seg.offset_k],
                    k_seg.size_k * sizeof(float));
    }

    float block_scale = 1.0f;
    if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
        float amax = 0.0f;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
            amax = rknpu2_calibration::calculate_entropy_amax(seg_fp32.data(), seg_fp32.size());
        } else {
            for (float v : seg_fp32) amax = std::max(amax, std::abs(v));
        }
        const float quant_divisor = (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) ? 7.0f : 127.0f;
        block_scale = (amax == 0.0f) ? 1.0f : amax / quant_divisor;
    }

    std::vector<uint8_t> seg_npu;
    int element_bits = 0;
    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
        element_bits = 16;
        seg_npu.resize(seg_elements * 2);
        rknpu2_quantization::convert_fp32_to_fp16(seg_fp32.data(), (uint16_t*)seg_npu.data(), seg_elements);
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
        element_bits = 8;
        seg_npu.resize(seg_elements);
        rknpu2_quantization::quantize_fp32_to_int8(seg_fp32.data(), (int8_t*)seg_npu.data(), seg_elements, block_scale);
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
        element_bits = 4;
        seg_npu.resize(seg_elements / 2);
        rknpu2_quantization::quantize_fp32_to_int4_packed(seg_fp32.data(), seg_npu.data(), seg_elements, block_scale);
    }

    pack_native(dst_dma_ptr, seg_npu.data(),
                k_seg.size_k, 0, k_seg.size_k, pipeline->k_align,
                n_seg.size_n, 0, n_seg.size_n, pipeline->n_align,
                element_bits);

    return block_scale;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    rknpu_route_stats().set_tensor_calls.fetch_add(1, std::memory_order_relaxed);

    // F8.3: pre-quantize only the first full set_tensor call against an alloc.
    // Real weights see exactly one such call at model load. Scheduler-managed
    // copy tensors (the F16 KV-cache slices ggml_backend_sched dups into the
    // RKNPU buffer for batched attention) see a fresh set_tensor every step —
    // demote those to memcpy so graph_compute reads fresh values via the
    // dynamic path. Partial writes are also memcpy-only: pre-quantizing the
    // whole tensor from a partial source buffer would read beyond the source.
    // INPUT/OUTPUT flags are unreliable (the sched only sets them when
    // n_copies > 1; default n_copies is 1), so we count the calls.
    bool is_first_call = false;
    const bool is_full_tensor_set = (offset == 0 && size == ggml_nbytes(tensor));
    if (pipeline) {
        size_t required_size = get_tensor_packed_size(tensor);
        ctx->get_tensor_allocation(tensor_offset_in_virtual, required_size);
        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->tensor_allocs.find(tensor_offset_in_virtual);
        GGML_ASSERT(it != ctx->tensor_allocs.end());
        const int prev_count = it->second.set_count++;
        if (prev_count == 0 && is_full_tensor_set) {
            is_first_call = true;
            rknpu_route_stats().set_tensor_first_pipelined.fetch_add(1, std::memory_order_relaxed);
        } else {
            it->second.pre_quantized = false;
            rknpu_route_stats().set_tensor_repeat_pipelined.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        rknpu_route_stats().set_tensor_no_pipeline.fetch_add(1, std::memory_order_relaxed);
    }

    if (pipeline && is_first_call) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];
        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        // Initializing Hadamard Transform Logic
        if (pipeline->use_hadamard) {
            std::vector<float> s_vec(K_op, 1.0f);
            std::mt19937 gen(reinterpret_cast<uintptr_t>(tensor));
            std::uniform_int_distribution<int> distrib(0, 1);

            for(int k = 0; k < K_op; ++k) {
                s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
            }

            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->hadamard_s_vectors[tensor] = s_vec;
        }

        // Computing global scale
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        // Allocating a new buffer for a tensor
        size_t required_size = get_tensor_packed_size(tensor);
        auto alloc = ctx->get_tensor_allocation(tensor_offset_in_virtual, required_size);
        uint8_t* tensor_dma_ptr = (uint8_t*)alloc.mem->virt_addr;

        // Computing specific hardware segments
        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        std::vector<float> seg_fp32;
        std::vector<uint8_t> seg_npu;
        uint8_t* current_write_ptr = tensor_dma_ptr + offset;

        std::vector<float> tensor_block_scales;

        // Processing individual segments block-by-block
        for (const auto& k_seg : k_segments) {
            for (const auto& n_seg : n_segments) {
                if (n_seg.size_n == 0) continue;

                // Dequantizing the block
                dequantize_tensor_segment(seg_fp32, tensor, ctx, data, K, N, K_op, k_seg, n_seg, pipeline->use_hadamard);

                // Calculating local scale of the block
                float block_scale = 1.0f;
                if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
                    float amax = 0.0f;
                    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                        amax = rknpu2_calibration::calculate_entropy_amax(seg_fp32.data(), seg_fp32.size());
                    } else {
                        for (float val : seg_fp32) {
                            amax = std::max(amax, std::abs(val));
                        }
                    }
                    float quant_divisor = (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) ? 7.0f : 127.0f;
                    block_scale = (amax == 0.0f) ? 1.0f : amax / quant_divisor;
                }
                tensor_block_scales.push_back(block_scale);

                // Quantizing
                quantize_tensor_segment(seg_fp32, seg_npu, k_seg, n_seg, block_scale, pipeline->npu_type_b);

                // Packing into chip native layout
                size_t bytes_written = pack_tensor_segment(seg_npu, current_write_ptr, k_seg, n_seg, pipeline);

                current_write_ptr += bytes_written;
            }
        }

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->quantized_tensor_scales[tensor] = tensor_block_scales;
            // F8.3: mark this alloc as actually pre-quantized so graph_compute
            // can distinguish it from init_tensor-allocated runtime tensors
            // (e.g. F16 KV cache slabs) that share the alloc map but have
            // never had quantized bytes written to their DMA buffer.
            auto it = ctx->tensor_allocs.find(tensor_offset_in_virtual);
            if (it != ctx->tensor_allocs.end()) {
                it->second.pre_quantized = true;
            }
        }

        rknn_matmul_ctx sync_ctx = g_domain_manager.get_allocator_context(alloc.iommu_domain_id);
        RKNN_CHECK(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE");

        // TST.2: also keep the raw GGUF bytes in tensor->data so cross-backend
        // copies (ggml_backend_tensor_copy via get_tensor) and any other
        // external read-back returns the original format instead of the
        // NPU-packed bytes that live in alloc.mem. tensor->data already lives
        // inside the mmap'd virtual_base, so this is a one-time extra memcpy
        // at model load with no additional allocation.
        memcpy((uint8_t*)tensor->data + offset, data, size);
    } else {
        memcpy((uint8_t*)tensor->data + offset, data, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context*)buffer->context;
    UNUSED(ctx);

    // TST.2: tensor->data is now the source of truth for the original GGUF
    // bytes (set_tensor's pipeline path memcpy's there too). Reading from
    // alloc.mem would give NPU-packed bytes, which CPU consumers cannot
    // interpret as Q8_0 / F16 / etc — see TST.1 findings.
    memcpy(data, (uint8_t*)tensor->data + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);

    // TST.2 follow-up: tensor->data lives in virtual_base and is the source
    // of truth for get_tensor and the dynamic-B path. Clearing only alloc.mem
    // would leave stale GGUF bytes visible on read-back. Clear the whole
    // virtual_base so semantics match the CPU backend's clear.
    if (ctx->virtual_base) {
        memset(ctx->virtual_base, value, ctx->total_size);
    }

    for (auto& pair : ctx->tensor_allocs) {
        memset((uint8_t*)pair.second.mem->virt_addr, value, pair.second.size);
    }
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    // Reserving virtual memory block
    void* virtual_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (virtual_base == MAP_FAILED) {
        return NULL;
    }

    // Initializing buffer context
    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context();
    ctx->virtual_base = virtual_base;
    ctx->total_size = size;
    ctx->name = "rknpu_virtual_buffer";

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    return get_tensor_packed_size(tensor);
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights / runtime B
            const struct ggml_tensor * src1 = op->src[1]; // Activations
            auto& stats = rknpu_route_stats();
            stats.mul_mat_total.fetch_add(1, std::memory_order_relaxed);

            // src0 is always rank-2 here. src1 / op may be rank-3 (broadcast
            // over n_head_q) when F8.3 is enabled. ne[3] must always be 1.
            const bool batched_ok = rknpu_batched_mm_enabled();
            const bool batched_shape =
                (src1->ne[2] > 1) || (op->ne[2] > 1);

            if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
                src1->ne[3] != 1 || op->ne[3] != 1) {
                if (rknpu_batched_diag_enabled()) {
                    rknpu_log_batched_shape_once(src0, src1, op);
                }
                stats.rejected_rank.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            if (batched_shape) {
                if (!batched_ok) {
                    if (rknpu_batched_diag_enabled()) {
                        rknpu_log_batched_shape_once(src0, src1, op);
                    }
                    stats.rejected_batched_off.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                // dst's head dim must match src1's; we slice both by nb[2].
                if (op->ne[2] != src1->ne[2]) {
                    stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                // F8.6a: gate small-M (decode) shapes off the NPU. Per-call
                // B-prep on F16 K/V cache slices is more expensive than the
                // CPU matmul for M=1 and very small M.
                const int min_m = rknpu_batched_mm_min_m();
                if (min_m > 0 && (int)src1->ne[1] < min_m) {
                    if (rknpu_batched_diag_enabled()) {
                        rknpu_log_batched_shape_once(src0, src1, op);
                    }
                    stats.rejected_min_m.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }

                // F8.6b CBUF cap experiment outcome: with the F8.5.CLEAN
                // teardown fix, the original "crashing" shape
                // (M_total=2048, K=512, FP16) runs cleanly on librknnrt
                // 2.3.2. The crash was the destructor bug, NOT a hardware
                // capacity wall. allbilly/npu's CBUF formula applies to
                // their pure-C driver, not to librknnrt.
                // Cap is now OPT-IN via RKNPU_BATCHED_MM_ENABLE_CBUF_CAP=1
                // to keep the diagnostic available, but default OFF — let
                // librknnrt decide. F8.6e (M-tile) becomes a perf knob
                // instead of a stability requirement.
                static const bool enable_cbuf_cap = []() {
                    const char* v = std::getenv("RKNPU_BATCHED_MM_ENABLE_CBUF_CAP");
                    return v != nullptr && v[0] != '\0' && v[0] != '0';
                }();
                if (enable_cbuf_cap) {
                    const auto* pipe_for_cap = config.resolve_op_support(src0);
                    if (pipe_for_cap) {
                        const int K_seg_worst = (int)src0->ne[0];
                        const int max_m = rknpu_max_m_for_cbuf(pipe_for_cap, K_seg_worst);
                        const int64_t M_total = src1->ne[1] * src1->ne[2];
                        if (M_total > max_m) {
                            if (rknpu_batched_diag_enabled()) {
                                rknpu_log_batched_shape_once(src0, src1, op);
                            }
                            stats.rejected_cbuf.fetch_add(1, std::memory_order_relaxed);
                            return false;
                        }
                    }
                }
            }

            // Searching for available hardware pipeline for this tensor
            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) {
                stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            // Rejecting zero-dimension ops
            if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
                src1->ne[0] == 0 || src1->ne[1] == 0) {
                stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != GGML_TYPE_F32) {
                stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % pipeline->k_align != 0) {
                stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % pipeline->n_align != 0) {
                stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            // src0 must be contiguous (we read it row-major in dequantize_row).
            // For batched src1 we only require contiguous rows so permuted
            // attention Q (post permute(0,2,1,3)) is acceptable.
            if (!ggml_is_contiguous(src0)) {
                stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (batched_shape) {
                if (!ggml_is_contiguous_rows(src1)) {
                    stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
            } else {
                if (!ggml_is_contiguous(src1)) {
                    stats.rejected_other.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
            }

            if (batched_shape) {
                stats.accepted_batched.fetch_add(1, std::memory_order_relaxed);
            } else {
                stats.accepted_rank2.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
        default:
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // Fetch device from environment variable, default to RK3588 if not set
    const char* env_device = std::getenv("RKNPU_DEVICE");
    std::string target_device = env_device ? env_device : "RK3588";
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device(target_device)) return NULL;

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();

    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }

    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .is_host        = */ NULL,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type; },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif