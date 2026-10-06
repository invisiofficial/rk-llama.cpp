#include <arm_neon.h>
#include <cmath>
#include <algorithm>

#include "rknpu2-quantization.h"

namespace rknpu2_quantization {

// --- Conversion from FP32 ---

void convert_fp32_to_fp16(const float * src, uint16_t * dst, size_t n_elements) {
    size_t i = 0;
#ifdef __ARM_NEON
    for (; i + 7 < n_elements; i += 8) {
        float32x4_t f32_vec_0 = vld1q_f32(src + i);
        float32x4_t f32_vec_1 = vld1q_f32(src + i + 4);
        float16x8_t f16_vec = vcombine_f16(vcvt_f16_f32(f32_vec_0), vcvt_f16_f32(f32_vec_1));
        vst1q_u16(dst + i, (uint16x8_t)f16_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = GGML_FP32_TO_FP16(src[i]);
    }
}

void quantize_fp32_to_int8(const float * src, int8_t * dst, size_t n_elements, const float * scales, size_t num_scales) {
    if (num_scales == 1) {
        const float scale = scales[0];
        const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
        for (size_t i = 0; i < n_elements; ++i) {
            dst[i] = (int8_t)roundf(src[i] * iscale);
        }
    } else {
        for (size_t i = 0; i < n_elements; ++i) {
            float scale = scales[i];
            float iscale = (scale == 0.0f || std::abs(scale) < 1e-9f) ? 0.0f : 1.0f / scale;
            int32_t q = (int32_t)std::roundf(src[i] * iscale);
            q = std::max(-127, std::min(127, q));
            dst[i] = (int8_t)q;
        }
    }
}

void quantize_fp32_to_int4_packed(const float * src, uint8_t * dst, size_t n_elements, const float * scales, size_t num_scales) {
    if (num_scales == 1) {
        const float scale = scales[0];
        const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
        for (size_t i = 0; i < n_elements / 2; ++i) {
            float v0_f = src[i * 2 + 0] * iscale;
            float v1_f = src[i * 2 + 1] * iscale;

            int8_t v0_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(v0_f)));
            int8_t v1_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(v1_f)));

            dst[i] = ((uint8_t)v0_i & 0x0F) | (((uint8_t)v1_i & 0x0F) << 4);
        }
    } else {
        for (size_t i = 0; i < n_elements / 2; ++i) {
            float s0 = scales[i * 2 + 0];
            float s1 = scales[i * 2 + 1];
            float is0 = (s0 == 0.0f || std::abs(s0) < 1e-9f) ? 0.0f : 1.0f / s0;
            float is1 = (s1 == 0.0f || std::abs(s1) < 1e-9f) ? 0.0f : 1.0f / s1;

            int32_t v0_i = (int32_t)std::roundf(src[i * 2 + 0] * is0);
            int32_t v1_i = (int32_t)std::roundf(src[i * 2 + 1] * is1);
            v0_i = std::max(-7, std::min(7, v0_i));
            v1_i = std::max(-7, std::min(7, v1_i));

            dst[i] = ((uint8_t)v0_i & 0x0F) | (((uint8_t)v1_i & 0x0F) << 4);
        }
    }
}


// --- Dequantization to FP32 ---

void dequantize_int16_to_fp32(const int16_t * src, float * dst, size_t n_elements, float scale) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 3 < n_elements; i += 4) {
        int16x4_t i16_vec = vld1_s16(src + i);
        int32x4_t i32_vec = vmovl_s16(i16_vec);
        float32x4_t f32_vec = vcvtq_f32_s32(i32_vec);
        f32_vec = vmulq_f32(f32_vec, scale_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (float)src[i] * scale;
    }
}

void dequantize_int32_to_fp32(const int32_t * src, float * dst, size_t n_elements, float scale) {
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 3 < n_elements; i += 4) {
        int32x4_t i32_vec = vld1q_s32(src + i);
        float32x4_t f32_vec = vcvtq_f32_s32(i32_vec);
        f32_vec = vmulq_f32(f32_vec, scale_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (float)src[i] * scale;
    }
}

} // namespace rknpu2_quantization