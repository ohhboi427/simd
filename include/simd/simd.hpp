#pragma once

#include "simd_export.h"

#include "traits.hpp"
#include "impl/simd_standard.hpp"
#include "impl/simd_sse42.hpp"

#include <array>
#include <concepts>
#include <utility>

namespace simd {
    SIMD_EXPORT auto hello_world() -> void;

    template<typename T, typename A = abi::scalar, typename I = isa::standard>
    struct alignas(sizeof(typename simd_traits<T, A, I>::type)) simd {
        using traits = simd_traits<T, A, I>;
        using type   = traits::type;

        type data;

        simd() noexcept
            : data{ traits::set1(T{}) } {}

        simd(const type data) noexcept // NOLINT
            : data{ data } {}

        explicit simd(const T scalar) noexcept
            requires (!std::same_as<T, type>)
            : data{ traits::set1(scalar) } {}

        template<std::convertible_to<T>... Args>
            requires (sizeof...(Args) == traits::LANES)
        explicit simd(Args&&... args) noexcept
            : data{ traits::setr(std::forward<Args>(args)...) } {}

        [[nodiscard]] static auto load_aligned(const T* src) noexcept -> simd {
            return { traits::load(src) };
        }

        [[nodiscard]] static auto load_unaligned(const T* src) noexcept -> simd {
            return { traits::loadu(src) };
        }

        auto store_aligned(T* dest) const noexcept -> void {
            traits::store(data, dest);
        }

        auto store_unaligned(T* dest) const noexcept -> void {
            traits::storeu(data, dest);
        }

        [[nodiscard]] auto get() const noexcept -> std::array<T, traits::LANES> {
            std::array<T, traits::LANES> dest;
            traits::storeu(data, dest.data());

            return dest;
        }
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
