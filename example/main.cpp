#include <simd/simd.hpp>

#include <cstdint>
#include <print>

auto main() -> int {
    using sse42_int = simd::simd<std::int32_t, simd::abi::m128, simd::isa::sse42>;

    const sse42_int a{ 1, 2, 3, 4 };
    const sse42_int b{ 2 };
    const auto      c = -a * b;

    std::println("{}", c.get());
}
