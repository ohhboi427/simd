#include <simd/simd.hpp>

auto main() -> int {
    simd::simd<int, simd::abi::m128, simd::isa::sse42> a{ 1, 2, 3, 4 };
    simd::simd<int, simd::abi::m128, simd::isa::sse42> b{ 2 };

    const auto c = -a * b;
}
