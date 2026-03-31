#pragma once

#include "../macros.hpp"
#include "../traits.hpp"

#include <cstdint>

#include <immintrin.h>

namespace simd {
    namespace isa {
        template<>
        struct prior_isa<avx2> {
            using type = sse42;
        };

        template<>
        struct default_abi<avx2> {
            using type = abi::m256;
        };
    }

    template<>
    struct simd_traits<std::int32_t, abi::m256, isa::avx2> {
        using type = __m256i;

        [[nodiscard]] SIMD_INLINE static auto set1(const std::int32_t x) noexcept -> type {
            return _mm256_set1_epi32(x);
        }

        [[nodiscard]] SIMD_INLINE static auto set(
            const std::int32_t x0,
            const std::int32_t x1,
            const std::int32_t x2,
            const std::int32_t x3,
            const std::int32_t x4,
            const std::int32_t x5,
            const std::int32_t x6,
            const std::int32_t x7
        ) noexcept -> type {
            return _mm256_set_epi32(x0, x1, x2, x3, x4, x5, x6, x7);
        }

        [[nodiscard]] SIMD_INLINE static auto setr(
            const std::int32_t x0,
            const std::int32_t x1,
            const std::int32_t x2,
            const std::int32_t x3,
            const std::int32_t x4,
            const std::int32_t x5,
            const std::int32_t x6,
            const std::int32_t x7
        ) noexcept -> type {
            return _mm256_setr_epi32(x0, x1, x2, x3, x4, x5, x6, x7);
        }

        [[nodiscard]] SIMD_INLINE static auto load(const std::int32_t* p) noexcept -> type {
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
        }

        [[nodiscard]] SIMD_INLINE static auto loadu(const std::int32_t* p) noexcept -> type {
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        }

        SIMD_INLINE static auto store(const type a, std::int32_t* p) noexcept -> void {
            _mm256_store_si256(reinterpret_cast<__m256i*>(p), a);
        }

        SIMD_INLINE static auto storeu(const type a, std::int32_t* p) noexcept -> void {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a);
        }

        [[nodiscard]] SIMD_INLINE static auto neg(const type x) noexcept -> type {
            return _mm256_sub_epi32(_mm256_setzero_si256(), x);
        }

        [[nodiscard]] SIMD_INLINE static auto add(const type a, const type b) noexcept -> type {
            return _mm256_add_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto sub(const type a, const type b) noexcept -> type {
            return _mm256_sub_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto mul(const type a, const type b) noexcept -> type {
            return _mm256_mullo_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto fmadd(const type a, const type b, const type c) noexcept -> type {
            return add(mul(a, b), c);
        }

        [[nodiscard]] SIMD_INLINE static auto fnmadd(const type a, const type b, const type c) noexcept -> type {
            return sub(c, mul(a, b));
        }

        [[nodiscard]] SIMD_INLINE static auto fmsub(const type a, const type b, const type c) noexcept -> type {
            return sub(mul(a, b), c);
        }

        [[nodiscard]] SIMD_INLINE static auto fnmsub(const type a, const type b, const type c) noexcept -> type {
            return neg(add(mul(a, b), c));
        }

        [[nodiscard]] SIMD_INLINE static auto inv(const type x) noexcept -> type {
            const __m256i all_ones = _mm256_cmpeq_epi32(x, x);
            return _mm256_xor_si256(all_ones, x);
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return _mm256_and_si256(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto conjinv(const type a, const type b) noexcept -> type {
            return _mm256_andnot_si256(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto disj(const type a, const type b) noexcept -> type {
            return _mm256_or_si256(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto exor(const type a, const type b) noexcept -> type {
            return _mm256_xor_si256(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto lshift(const type a, const int count) noexcept -> type {
            return _mm256_slli_epi32(a, count);
        }

        [[nodiscard]] SIMD_INLINE static auto rshift(const type a, const int count) noexcept -> type {
            return _mm256_srai_epi32(a, count);
        }

        [[nodiscard]] SIMD_INLINE static auto rshift2(const type a, const int count) noexcept -> type {
            return _mm256_srli_epi32(a, count);
        }

        [[nodiscard]] SIMD_INLINE static auto eq(const type a, const type b) noexcept -> type {
            return _mm256_cmpeq_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto neq(const type a, const type b) noexcept -> type {
            return inv(_mm256_cmpeq_epi32(a, b));
        }

        [[nodiscard]] SIMD_INLINE static auto gt(const type a, const type b) noexcept -> type {
            return _mm256_cmpgt_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto ge(const type a, const type b) noexcept -> type {
            return inv(_mm256_cmpgt_epi32(b, a)); // NOLINT
        }

        [[nodiscard]] SIMD_INLINE static auto lt(const type a, const type b) noexcept -> type {
            return _mm256_cmpgt_epi32(b, a); // NOLINT
        }

        [[nodiscard]] SIMD_INLINE static auto le(const type a, const type b) noexcept -> type {
            return inv(_mm256_cmpgt_epi32(a, b));
        }

        [[nodiscard]] SIMD_INLINE static auto abs(const type x) noexcept -> type {
            return _mm256_abs_epi32(x);
        }

        [[nodiscard]] SIMD_INLINE static auto min(const type a, const type b) noexcept -> type {
            return _mm256_min_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto max(const type a, const type b) noexcept -> type {
            return _mm256_max_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto cast(const type x) noexcept -> __m256 {
            return _mm256_castsi256_ps(x);
        }

        [[nodiscard]] SIMD_INLINE static auto cvt(const type x) noexcept -> __m256 {
            return _mm256_cvtepi32_ps(x);
        }
    };

    template<>
    struct simd_traits<float, abi::m256, isa::avx2> {
        using type = __m256;

        [[nodiscard]] SIMD_INLINE static auto set1(const float x) noexcept -> type {
            return _mm256_set1_ps(x);
        }

        [[nodiscard]] SIMD_INLINE static auto set(
            const float x0,
            const float x1,
            const float x2,
            const float x3,
            const float x4,
            const float x5,
            const float x6,
            const float x7
        ) noexcept -> type {
            return _mm256_set_ps(x0, x1, x2, x3, x4, x5, x6, x7);
        }

        [[nodiscard]] SIMD_INLINE static auto setr(
            const float x0,
            const float x1,
            const float x2,
            const float x3,
            const float x4,
            const float x5,
            const float x6,
            const float x7
        ) noexcept -> type {
            return _mm256_setr_ps(x0, x1, x2, x3, x4, x5, x6, x7);
        }

        [[nodiscard]] SIMD_INLINE static auto load(const float* p) noexcept -> type {
            return _mm256_load_ps(p);
        }

        [[nodiscard]] SIMD_INLINE static auto loadu(const float* p) noexcept -> type {
            return _mm256_loadu_ps(p);
        }

        SIMD_INLINE static auto store(const type a, float* p) noexcept -> void {
            _mm256_store_ps(p, a);
        }

        SIMD_INLINE static auto storeu(const type a, float* p) noexcept -> void {
            _mm256_storeu_ps(p, a);
        }

        [[nodiscard]] SIMD_INLINE static auto neg(const type x) noexcept -> type {
            return _mm256_xor_ps(x, _mm256_set1_ps(-0.0F));
        }

        [[nodiscard]] SIMD_INLINE static auto add(const type a, const type b) noexcept -> type {
            return _mm256_add_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto sub(const type a, const type b) noexcept -> type {
            return _mm256_sub_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto mul(const type a, const type b) noexcept -> type {
            return _mm256_mul_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto fmadd(const type a, const type b, const type c) noexcept -> type {
#if defined(__FMA__)
            return _mm256_fmadd_ps(a, b, c);
#else
            return add(mul(a, b), c);
#endif
        }

        [[nodiscard]] SIMD_INLINE static auto fnmadd(const type a, const type b, const type c) noexcept -> type {
#if defined(__FMA__)
            return _mm256_fnmadd_ps(a, b, c);
#else
            return sub(c, mul(a, b));
#endif
        }

        [[nodiscard]] SIMD_INLINE static auto fmsub(const type a, const type b, const type c) noexcept -> type {
#if defined(__FMA__)
            return _mm256_fmsub_ps(a, b, c);
#else
            return sub(mul(a, b), c);
#endif
        }

        [[nodiscard]] SIMD_INLINE static auto fnmsub(const type a, const type b, const type c) noexcept -> type {
#if defined(__FMA__)
            return _mm256_fnmsub_ps(a, b, c);
#else
            return neg(add(mul(a, b), c));
#endif
        }

        [[nodiscard]] SIMD_INLINE static auto inv(const type a) noexcept -> type {
            const __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
            return _mm256_xor_ps(_mm256_castsi256_ps(all_ones), a);
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return _mm256_and_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto conjinv(const type a, const type b) noexcept -> type {
            return _mm256_andnot_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto disj(const type a, const type b) noexcept -> type {
            return _mm256_or_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto exor(const type a, const type b) noexcept -> type {
            return _mm256_xor_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto eq(const type a, const type b) noexcept -> type {
            return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
        }

        [[nodiscard]] SIMD_INLINE static auto neq(const type a, const type b) noexcept -> type {
            return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
        }

        [[nodiscard]] SIMD_INLINE static auto gt(const type a, const type b) noexcept -> type {
            return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
        }

        [[nodiscard]] SIMD_INLINE static auto ge(const type a, const type b) noexcept -> type {
            return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
        }

        [[nodiscard]] SIMD_INLINE static auto lt(const type a, const type b) noexcept -> type {
            return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
        }

        [[nodiscard]] SIMD_INLINE static auto le(const type a, const type b) noexcept -> type {
            return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
        }

        [[nodiscard]] SIMD_INLINE static auto abs(const type x) noexcept -> type {
            const __m256 sign_bit = _mm256_set1_ps(-0.0F);
            return _mm256_andnot_ps(sign_bit, x);
        }

        [[nodiscard]] SIMD_INLINE static auto trunc(const type x) noexcept -> type {
            return _mm256_round_ps(x, _MM_FROUND_TO_ZERO);
        }

        [[nodiscard]] SIMD_INLINE static auto round(const type x) noexcept -> type {
            return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT);
        }

        [[nodiscard]] SIMD_INLINE static auto floor(const type x) noexcept -> type {
            return _mm256_floor_ps(x);
        }

        [[nodiscard]] SIMD_INLINE static auto ceil(const type x) noexcept -> type {
            return _mm256_ceil_ps(x);
        }

        [[nodiscard]] SIMD_INLINE static auto min(const type a, const type b) noexcept -> type {
            return _mm256_min_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto max(const type a, const type b) noexcept -> type {
            return _mm256_max_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto sqrt(const type x) noexcept -> type {
            return _mm256_sqrt_ps(x);
        }

        [[nodiscard]] SIMD_INLINE static auto rsqrt(const type x) noexcept -> type {
            return _mm256_rsqrt_ps(x);
        }

        [[nodiscard]] SIMD_INLINE static auto cast(const type x) noexcept -> __m256i {
            return _mm256_castps_si256(x);
        }

        [[nodiscard]] SIMD_INLINE static auto cvt(const type x) noexcept -> __m256i {
            return _mm256_cvtps_epi32(x);
        }
    };
}
