#pragma once

#include "../traits.hpp"

#include <bit>
#include <cstdint>

namespace simd {
    template<>
    struct isa::default_abi<isa::standard> {
        using type = abi::scalar;
    };

    template<>
    struct simd_traits<std::int32_t, abi::scalar, isa::standard> {
        using type = std::int32_t;

        [[nodiscard]] static auto neg(const type a) noexcept -> type {
            return -a;
        }

        [[nodiscard]] static auto add(const type a, const type b) noexcept -> type {
            return a + b;
        }

        [[nodiscard]] static auto sub(const type a, const type b) noexcept -> type {
            return a - b;
        }

        [[nodiscard]] static auto mul(const type a, const type b) noexcept -> type {
            return a * b;
        }

        [[nodiscard]] static auto div(const type a, const type b) noexcept -> type {
            return a / b;
        }

        [[nodiscard]] static auto conj(const type a, const type b) noexcept -> type {
            return a & b;
        }

        [[nodiscard]] static auto disj(const type a, const type b) noexcept -> type {
            return a | b;
        }

        [[nodiscard]] static auto exor(const type a, const type b) noexcept -> type {
            return a ^ b;
        }

        [[nodiscard]] static auto lshift(const type a, const int count) noexcept -> type {
            return a << count;
        }

        [[nodiscard]] static auto rshift(const type a, const int count) noexcept -> type {
            return a >> count;
        }

        [[nodiscard]] static auto rshift2(const type a, const int count) noexcept -> type {
            return std::bit_cast<std::int32_t>(std::bit_cast<std::uint32_t>(a) >> count);
        }
    };

    template<>
    struct simd_traits<float, abi::scalar, isa::standard> {
        using type = float;

        [[nodiscard]] static auto neg(const type a) noexcept -> type {
            return -a;
        }

        [[nodiscard]] static auto add(const type a, const type b) noexcept -> type {
            return a + b;
        }

        [[nodiscard]] static auto sub(const type a, const type b) noexcept -> type {
            return a - b;
        }

        [[nodiscard]] static auto mul(const type a, const type b) noexcept -> type {
            return a * b;
        }

        [[nodiscard]] static auto div(const type a, const type b) noexcept -> type {
            return a / b;
        }

        [[nodiscard]] static auto conj(const type a, const type b) noexcept -> type {
            return std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) & std::bit_cast<std::uint32_t>(b));
        }

        [[nodiscard]] static auto disj(const type a, const type b) noexcept -> type {
            return std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) | std::bit_cast<std::uint32_t>(b));
        }

        [[nodiscard]] static auto exor(const type a, const type b) noexcept -> type {
            return std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) ^ std::bit_cast<std::uint32_t>(b));
        }
    };
}
