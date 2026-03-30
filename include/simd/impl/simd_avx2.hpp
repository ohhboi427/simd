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

        [[nodiscard]] SIMD_INLINE static auto neg(const type a) noexcept -> type {
            return _mm256_sub_epi32(_mm256_setzero_si256(), a);
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

        [[nodiscard]] SIMD_INLINE static auto inv(const type a) noexcept -> type {
            const __m256i all_ones = _mm256_cmpeq_epi32(a, a);
            return _mm256_xor_si256(all_ones, a);
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return _mm256_and_si256(a, b);
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

        [[nodiscard]] SIMD_INLINE static auto neg(const type a) noexcept -> type {
            return _mm256_xor_ps(a, _mm256_set1_ps(-0.0F));
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

        [[nodiscard]] SIMD_INLINE static auto inv(const type a) noexcept -> type {
            const __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
            return _mm256_xor_ps(_mm256_castsi256_ps(all_ones), a);
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return _mm256_and_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto disj(const type a, const type b) noexcept -> type {
            return _mm256_or_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto exor(const type a, const type b) noexcept -> type {
            return _mm256_xor_ps(a, b);
        }
    };
}
