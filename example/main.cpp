#include <simd/simd.hpp>

auto main() -> int {
    simd::simd<int> a{ 1 };
    simd::simd<int> b{ 2 };

    const auto c = a + b;
}
