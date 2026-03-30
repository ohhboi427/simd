#pragma once

#include "simd_export.h"

#include "traits.hpp"
#include "impl/simd_standard.hpp"

namespace simd {
    SIMD_EXPORT auto hello_world() -> void;

    template<typename T, typename A = abi::scalar, typename I = isa::standard>
    struct alignas(sizeof(typename simd_traits<T, A, I>::type)) simd {
        using traits = simd_traits<T, A, I>;
        using type   = traits::type;

        type data;
    };

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator-(const simd<T, A, I> a) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::neg(a.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator+(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::add(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator-(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::sub(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator*(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::mul(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator/(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::div(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator&(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::conj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator|(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::disj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator^(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::exor(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator<<(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::lshift(a.data, count) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] auto operator>>(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::rshift(a.data, count) };
    }
}
