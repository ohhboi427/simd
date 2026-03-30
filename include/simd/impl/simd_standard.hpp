#pragma once

#include "../macros.hpp"
#include "../traits.hpp"

#include <bit>
#include <cstdint>

namespace simd {
    namespace isa {
        struct standard;

        template<>
        struct default_abi<standard> {
            using type = abi::scalar;
        };
    }

    template<>
    struct simd_traits<std::int32_t, abi::scalar, isa::standard> {
        using type = std::int32_t;

        static constexpr int LANES = 1;

        [[nodiscard]] SIMD_INLINE static auto set1(const std::int32_t x) noexcept -> type {
            return x;
        }

        [[nodiscard]] SIMD_INLINE static auto set(const std::int32_t x) noexcept -> type {
            return x;
        }

        [[nodiscard]] SIMD_INLINE static auto setr(const std::int32_t x) noexcept -> type {
            return x;
        }

        [[nodiscard]] SIMD_INLINE static auto load(const std::int32_t* p) noexcept -> type {
            return *p;
        }

        [[nodiscard]] SIMD_INLINE static auto loadu(const std::int32_t* p) noexcept -> type {
            return *p;
        }

        SIMD_INLINE static auto store(const type a, std::int32_t* p) noexcept -> void {
            *p = a;
        }

        SIMD_INLINE static auto storeu(const type a, std::int32_t* p) noexcept -> void {
            *p = a;
        }

        [[nodiscard]] SIMD_INLINE static auto neg(const type a) noexcept -> type {
            return -a;
        }

        [[nodiscard]] SIMD_INLINE static auto add(const type a, const type b) noexcept -> type {
            return a + b;
        }

        [[nodiscard]] SIMD_INLINE static auto sub(const type a, const type b) noexcept -> type {
            return a - b;
        }

        [[nodiscard]] SIMD_INLINE static auto mul(const type a, const type b) noexcept -> type {
            return a * b;
        }

        [[nodiscard]] SIMD_INLINE static auto div(const type a, const type b) noexcept -> type {
            return a / b;
        }

        [[nodiscard]] SIMD_INLINE static auto inv(const type a) noexcept -> type {
            return ~a;
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return a & b;
        }

        [[nodiscard]] SIMD_INLINE static auto disj(const type a, const type b) noexcept -> type {
            return a | b;
        }

        [[nodiscard]] SIMD_INLINE static auto exor(const type a, const type b) noexcept -> type {
            return a ^ b;
        }

        [[nodiscard]] SIMD_INLINE static auto lshift(const type a, const int count) noexcept -> type {
            return a << count;
        }

        [[nodiscard]] SIMD_INLINE static auto rshift(const type a, const int count) noexcept -> type {
            return a >> count;
        }

        [[nodiscard]] SIMD_INLINE static auto rshift2(const type a, const int count) noexcept -> type {
            return std::bit_cast<std::int32_t>(std::bit_cast<std::uint32_t>(a) >> count);
        }
    };

    template<>
    struct simd_traits<float, abi::scalar, isa::standard> {
        using type = float;

        static constexpr int LANES = 1;

        [[nodiscard]] SIMD_INLINE static auto set1(const float x) noexcept -> type {
            return x;
        }

        [[nodiscard]] SIMD_INLINE static auto set(const float x) noexcept -> type {
            return x;
        }

        [[nodiscard]] SIMD_INLINE static auto setr(const float x) noexcept -> type {
            return x;
        }

        [[nodiscard]] SIMD_INLINE static auto load(const float* p) noexcept -> type {
            return *p;
        }

        [[nodiscard]] SIMD_INLINE static auto loadu(const float* p) noexcept -> type {
            return *p;
        }

        SIMD_INLINE static auto store(const type a, float* p) noexcept -> void {
            *p = a;
        }

        SIMD_INLINE static auto storeu(const type a, float* p) noexcept -> void {
            *p = a;
        }

        [[nodiscard]] SIMD_INLINE static auto neg(const type a) noexcept -> type {
            return -a;
        }

        [[nodiscard]] SIMD_INLINE static auto add(const type a, const type b) noexcept -> type {
            return a + b;
        }

        [[nodiscard]] SIMD_INLINE static auto sub(const type a, const type b) noexcept -> type {
            return a - b;
        }

        [[nodiscard]] SIMD_INLINE static auto mul(const type a, const type b) noexcept -> type {
            return a * b;
        }

        [[nodiscard]] SIMD_INLINE static auto div(const type a, const type b) noexcept -> type {
            return a / b;
        }

        [[nodiscard]] SIMD_INLINE static auto conj(const type a, const type b) noexcept -> type {
            return std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) & std::bit_cast<std::uint32_t>(b));
        }

        [[nodiscard]] SIMD_INLINE static auto disj(const type a, const type b) noexcept -> type {
            return std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) | std::bit_cast<std::uint32_t>(b));
        }

        [[nodiscard]] SIMD_INLINE static auto exor(const type a, const type b) noexcept -> type {
            return std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) ^ std::bit_cast<std::uint32_t>(b));
        }
    };
}
