#pragma once

#include "../traits.hpp"

namespace simd {
    template<>
    struct isa::default_abi<isa::standard> {
        using type = abi::scalar;
    };

    template<>
    struct simd_traits<int, abi::scalar, isa::standard> {
        using type = int;

        [[nodiscard]] static auto add(const type a, const type b) noexcept -> type {
            return a + b;
        }
    };
}
