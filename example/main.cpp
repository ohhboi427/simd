#include <simd/simd.hpp>

#include <cstdint>
#include <print>

auto main() -> int {
    const simd::native_simd<std::int32_t, simd::abi::m256> a{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const simd::native_simd<std::int32_t, simd::abi::m256> b{ 2 };

    const auto c = -a * b;

    std::println("{}", c.get());
}
