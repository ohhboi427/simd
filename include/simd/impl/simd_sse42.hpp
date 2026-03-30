#pragma once

#include "../macros.hpp"
#include "../traits.hpp"

#include <cstdint>

#include <emmintrin.h>
#include <smmintrin.h>

namespace simd {
    namespace isa {
        struct sse42;

        template<>
        struct default_abi<sse42> {
            using type = abi::m128;
        };
    }

    template<>
    struct simd_traits<std::int32_t, abi::m128, isa::sse42> {
        using type = __m128i;

        static constexpr int LANES = 4;

        [[nodiscard]] SIMD_INLINE static auto set1(const std::int32_t x) noexcept -> type {
            return _mm_set1_epi32(x);
        }

        [[nodiscard]] SIMD_INLINE static auto set(
            const std::int32_t x0,
            const std::int32_t x1,
            const std::int32_t x2,
            const std::int32_t x3
        ) noexcept -> type {
            return _mm_set_epi32(x0, x1, x2, x3);
        }

        [[nodiscard]] SIMD_INLINE static auto setr(
            const std::int32_t x0,
            const std::int32_t x1,
            const std::int32_t x2,
            const std::int32_t x3
        ) noexcept -> type {
            return _mm_setr_epi32(x0, x1, x2, x3);
        }

        [[nodiscard]] SIMD_INLINE static auto load(const std::int32_t* p) noexcept -> type {
            return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
        }

        [[nodiscard]] SIMD_INLINE static auto loadu(const std::int32_t* p) noexcept -> type {
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        }

        SIMD_INLINE static auto store(const type a, std::int32_t* p) noexcept -> void {
            _mm_store_si128(reinterpret_cast<__m128i*>(p), a);
        }

        SIMD_INLINE static auto storeu(const type a, std::int32_t* p) noexcept -> void {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(p), a);
        }

        [[nodiscard]] SIMD_INLINE static auto neg(const type a) noexcept -> type {
            return _mm_sub_epi32(_mm_setzero_si128(), a);
        }

        [[nodiscard]] SIMD_INLINE static auto add(const type a, const type b) noexcept -> type {
            return _mm_add_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto sub(const type a, const type b) noexcept -> type {
            return _mm_sub_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto mul(const type a, const type b) noexcept -> type {
            return _mm_mullo_epi32(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto inv(const type a) noexcept -> type {
            const __m128i all_ones = _mm_cmpeq_epi32(a, a);
            return _mm_xor_si128(all_ones, a);
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return _mm_and_si128(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto disj(const type a, const type b) noexcept -> type {
            return _mm_or_si128(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto exor(const type a, const type b) noexcept -> type {
            return _mm_xor_si128(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto lshift(const type a, const int count) noexcept -> type {
            return _mm_slli_epi32(a, count);
        }

        [[nodiscard]] SIMD_INLINE static auto rshift(const type a, const int count) noexcept -> type {
            return _mm_srai_epi32(a, count);
        }

        [[nodiscard]] SIMD_INLINE static auto rshift2(const type a, const int count) noexcept -> type {
            return _mm_srli_epi32(a, count);
        }
    };

    template<>
    struct simd_traits<float, abi::m128, isa::sse42> {
        using type = __m128;

        static constexpr int LANES = 4;

        [[nodiscard]] SIMD_INLINE static auto set1(const float x) noexcept -> type {
            return _mm_set1_ps(x);
        }

        [[nodiscard]] SIMD_INLINE static auto set(
            const float x0,
            const float x1,
            const float x2,
            const float x3
        ) noexcept -> type {
            return _mm_set_ps(x0, x1, x2, x3);
        }

        [[nodiscard]] SIMD_INLINE static auto setr(
            const float x0,
            const float x1,
            const float x2,
            const float x3
        ) noexcept -> type {
            return _mm_setr_ps(x0, x1, x2, x3);
        }

        [[nodiscard]] SIMD_INLINE static auto load(const float* p) noexcept -> type {
            return _mm_load_ps(p);
        }

        [[nodiscard]] SIMD_INLINE static auto loadu(const float* p) noexcept -> type {
            return _mm_loadu_ps(p);
        }

        SIMD_INLINE static auto store(const type a, float* p) noexcept -> void {
            _mm_store_ps(p, a);
        }

        SIMD_INLINE static auto storeu(const type a, float* p) noexcept -> void {
            _mm_storeu_ps(p, a);
        }

        [[nodiscard]] SIMD_INLINE static auto neg(const type a) noexcept -> type {
            return _mm_xor_ps(a, _mm_set1_ps(-0.0F));
        }

        [[nodiscard]] SIMD_INLINE static auto add(const type a, const type b) noexcept -> type {
            return _mm_add_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto sub(const type a, const type b) noexcept -> type {
            return _mm_sub_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto mul(const type a, const type b) noexcept -> type {
            return _mm_mul_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto inv(const type a) noexcept -> type {
            const __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
            return _mm_xor_ps(_mm_castsi128_ps(all_ones), a);
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return _mm_and_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto disj(const type a, const type b) noexcept -> type {
            return _mm_or_ps(a, b);
        }

        [[nodiscard]] SIMD_INLINE static auto exor(const type a, const type b) noexcept -> type {
            return _mm_xor_ps(a, b);
        }
    };
}
