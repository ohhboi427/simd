#include <simd/simd.hpp>

#include <cstdint>
#include <print>

auto main() -> int {
    using avx2_int = simd::simd<std::int32_t, simd::abi::m256, simd::isa::avx2>;

    const avx2_int a{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const avx2_int b{ 2 };

    const auto c = -a * b;

    std::println("{}", c.get());
}
