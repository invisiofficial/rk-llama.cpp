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

void quantize_fp32_to_int8(const float * src, int8_t * dst, size_t n_elements, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    size_t i = 0;
#ifdef __ARM_NEON
    const float32x4_t iscale_vec = vdupq_n_f32(iscale);
    for (; i + 15 < n_elements; i += 16) {
        float32x4_t f0 = vld1q_f32(src + i);
        float32x4_t f1 = vld1q_f32(src + i + 4);
        float32x4_t f2 = vld1q_f32(src + i + 8);
        float32x4_t f3 = vld1q_f32(src + i + 12);
        f0 = vmulq_f32(f0, iscale_vec);
        f1 = vmulq_f32(f1, iscale_vec);
        f2 = vmulq_f32(f2, iscale_vec);
        f3 = vmulq_f32(f3, iscale_vec);
        int32x4_t i0 = vcvtaq_s32_f32(f0);
        int32x4_t i1 = vcvtaq_s32_f32(f1);
        int32x4_t i2 = vcvtaq_s32_f32(f2);
        int32x4_t i3 = vcvtaq_s32_f32(f3);
        int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        int8x16_t result = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
        vst1q_s8(dst + i, result);
    }
#endif
    for (; i < n_elements; ++i) {
        dst[i] = (int8_t)roundf(src[i] * iscale);
    }
}

void quantize_fp32_to_int4_packed(const float * src, uint8_t * dst, size_t n_elements, float scale) {
    const float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
    size_t i_src = 0;
    size_t i_dst = 0;
#ifdef __ARM_NEON
    const float32x4_t iscale_vec = vdupq_n_f32(iscale);
    const int32x4_t lo7 = vdupq_n_s32(-7);
    const int32x4_t hi7 = vdupq_n_s32(7);
    for (; i_src + 15 < n_elements; i_src += 16, i_dst += 8) {
        float32x4_t f0 = vld1q_f32(src + i_src);
        float32x4_t f1 = vld1q_f32(src + i_src + 4);
        float32x4_t f2 = vld1q_f32(src + i_src + 8);
        float32x4_t f3 = vld1q_f32(src + i_src + 12);
        f0 = vmulq_f32(f0, iscale_vec);
        f1 = vmulq_f32(f1, iscale_vec);
        f2 = vmulq_f32(f2, iscale_vec);
        f3 = vmulq_f32(f3, iscale_vec);
        int32x4_t i0 = vminq_s32(hi7, vmaxq_s32(lo7, vcvtaq_s32_f32(f0)));
        int32x4_t i1 = vminq_s32(hi7, vmaxq_s32(lo7, vcvtaq_s32_f32(f1)));
        int32x4_t i2 = vminq_s32(hi7, vmaxq_s32(lo7, vcvtaq_s32_f32(f2)));
        int32x4_t i3 = vminq_s32(hi7, vmaxq_s32(lo7, vcvtaq_s32_f32(f3)));
        int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        int8x16_t b8 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
        uint8x16_t u8 = vreinterpretq_u8_s8(b8);
        uint8x16x2_t pairs = vuzpq_u8(u8, u8);
        uint8x8_t evens = vget_low_u8(pairs.val[0]);
        uint8x8_t odds  = vget_low_u8(pairs.val[1]);
        uint8x8_t evens_lo = vand_u8(evens, vdup_n_u8(0x0F));
        uint8x8_t odds_lo  = vshl_n_u8(vand_u8(odds, vdup_n_u8(0x0F)), 4);
        uint8x8_t packed  = vorr_u8(evens_lo, odds_lo);
        vst1_u8(dst + i_dst, packed);
    }
#endif
    for (; i_src < n_elements; i_src += 2) {
        float v0 = src[i_src] * iscale;
        float v1 = (i_src + 1 < n_elements) ? src[i_src + 1] * iscale : 0.0f;
        int8_t v0_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(v0)));
        int8_t v1_i = std::max((int8_t)-7, std::min((int8_t)7, (int8_t)roundf(v1)));
        dst[i_dst++] = ((uint8_t)v0_i & 0x0F) | (((uint8_t)v1_i & 0x0F) << 4);
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
