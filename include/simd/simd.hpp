#pragma once

#include "macros.hpp"
#include "traits.hpp"
#include "impl/simd_standard.hpp"
#include "impl/simd_sse42.hpp"
#include "impl/simd_avx2.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

namespace simd {
    template<typename T, typename A = abi::scalar, typename I = isa::standard>
    struct alignas(sizeof(typename simd_traits<T, A, I>::type)) simd {
        using traits = simd_traits<T, A, I>;
        using type   = traits::type;

        static constexpr std::size_t LANES = sizeof(type) / sizeof(T);

        type data;

        simd() noexcept
            : data{ traits::set1(T{}) } {}

        simd(const type data) noexcept // NOLINT
            : data{ data } {}

        explicit simd(const T scalar) noexcept
            requires (!std::same_as<T, type>)
            : data{ traits::set1(scalar) } {}

        template<std::convertible_to<T>... Args>
            requires (sizeof...(Args) == LANES)
        explicit simd(Args&&... args) noexcept
            : data{ traits::setr(std::forward<Args>(args)...) } {}

        [[nodiscard]] static constexpr auto lanes() noexcept -> std::size_t {
            return LANES;
        }

        [[nodiscard]] SIMD_INLINE static auto load_aligned(const T* src) noexcept -> simd {
            return { traits::load(src) };
        }

        [[nodiscard]] SIMD_INLINE static auto load_unaligned(const T* src) noexcept -> simd {
            return { traits::loadu(src) };
        }

        SIMD_INLINE auto store_aligned(T* dest) const noexcept -> void {
            traits::store(data, dest);
        }

        SIMD_INLINE auto store_unaligned(T* dest) const noexcept -> void {
            traits::storeu(data, dest);
        }

        [[nodiscard]] auto get() const noexcept -> std::array<T, LANES> {
            std::array<T, LANES> dest;
            traits::storeu(data, dest.data());

            return dest;
        }
    };

    template<typename U, typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto simd_cast(const simd<T, A, I> x) noexcept -> simd<U, A, I> {
        return { simd_traits<T, A, I>::cast(x.data) };
    }

    template<typename U, typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto simd_cvt(const simd<T, A, I> x) noexcept -> simd<U, A, I> {
        return { simd_traits<T, A, I>::cvt(x.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator-(const simd<T, A, I> a) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::neg(a.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator+(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::add(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator-(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::sub(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator*(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::mul(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator/(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::div(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator~(const simd<T, A, I> a) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::inv(a.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator&(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::conj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator|(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::disj(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator^(const simd<T, A, I> a, const simd<T, A, I> b) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::exor(a.data, b.data) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator<<(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::lshift(a.data, count) };
    }

    template<typename T, typename A, typename I>
    [[nodiscard]] SIMD_INLINE auto operator>>(const simd<T, A, I> a, const int count) noexcept -> simd<T, A, I> {
        return { simd_traits<T, A, I>::rshift(a.data, count) };
    }

#if defined(__AVX2__)
    using native_isa = isa::avx2;
#elif defined(__SSE4_2__)
    using available_isa = isa::sse42;
#else
    using available_isa = isa::standard;
#endif

    template<typename T, typename A = isa::default_abi<native_isa>>
    using native_simd = simd<T, A, native_isa>;
}
