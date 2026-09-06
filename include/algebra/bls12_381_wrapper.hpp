#pragma once

#include <mcl/bls12_381.hpp>

using namespace mcl::bn;

namespace mcl::bn {

inline bool operator<(const Fr& a, const Fr& b)
{
    const std::string a_hex = a.getStr(16);
    const std::string b_hex = b.getStr(16);
    if (a_hex.size() != b_hex.size()) {
        return a_hex.size() < b_hex.size();
    }
    return a_hex < b_hex;
}

} // namespace mcl::bn

/**
 * @brief Thin wrapper over MCL groups (G1/G2) providing
 *        convenient operator overloads.
 *
 * Expected instantiations: Group<G1>, Group<G2>.
 *
 * Supports:
 *  - Group<G>  + Group<G>
 *  - Group<G>  - Group<G>
 *  - -Group<G>
 *  - Fr * Group<G>   (scalar on the left)
 */
template<class G>
class Group {
public:
    G v;

    Group()
    {
        if constexpr (std::is_same_v<G, Fp12>) {
            v.setOne();  // GT is multiplicative: identity is 1
        } else {
            v.clear();   // identity element for G1/G2
        }
    }

    explicit Group(const G& g)
        : v(g)
    {
    }

    // Access underlying value if needed
    const G& value() const { return v; }
    G& value() { return v; }

    // Identity element helper
    static Group& zero();
    static Group& one();

    // Group addition
    friend Group operator+(const Group& a, const Group& b)
    {
        Group r;
        if constexpr (std::is_same_v<G, Fp12>) {
            Fp12::mul(r.v, a.v, b.v);
        } else {
            G::add(r.v, a.v, b.v);
        }
        return r;
    }

    Group& operator+=(const Group& other)
    {
        if constexpr (std::is_same_v<G, Fp12>) {
            Fp12::mul(v, v, other.v);
        } else {
            G::add(v, v, other.v);
        }
        return *this;
    }

    // Group subtraction
    friend Group operator-(const Group& a, const Group& b)
    {
        Group r;
        if constexpr (std::is_same_v<G, Fp12>) {
            // In Fp12 (representing GT), subtraction is interpreted as division (inverse multiplication)
            Fp12 inv_b;
            Fp12::unitaryInv(inv_b, b.v);
            Fp12::mul(r.v, a.v, inv_b);
        } else {
            G::sub(r.v, a.v, b.v);
        }
        return r;
    }

    Group& operator-=(const Group& other)
    {
        if constexpr (std::is_same_v<G, Fp12>) {
            Fp12 inv_other;
            Fp12::unitaryInv(inv_other, other.v);
            Fp12::mul(v, v, inv_other);
        } else {
            G::sub(v, v, other.v);
        }
        return *this;
    }

    // Group negation
    Group operator-() const
    {
        Group r;
        if constexpr (std::is_same_v<G, Fp12>) {
            Fp12::unitaryInv(r.v, v);
        } else {
            G::neg(r.v, v);
        }
        return r;
    }

    friend Group operator*(const Fr& s, const Group& g)
    {
        Group r;
        if constexpr (std::is_same_v<G, Fp12>) {
            Fp12::pow(r.v, g.v, s);
        } else {
            G::mul(r.v, g.v, s);
        }
        return r;
    }

    // Equality / inequality
    friend bool operator==(const Group& a, const Group& b)
    {
        return a.v == b.v;
    }

    friend bool operator!=(const Group& a, const Group& b)
    {
        return !(a == b);
    }

    // Inner product of two vectors: fs[0]*gs[0] + fs[1]*gs[1] + ... + fs[n-1]*gs[n-1]
    // If the vectors are of different lengths, the shorter vector is padded with zeros.
    static Group inner_prod(const std::vector<Fr>& fs, const std::vector<Group>& gs) 
    {
        size_t n = std::min(fs.size(), gs.size());
        if (n == 0) {
            return zero();  // identity element for empty inner product
        }
        Group r;
        for (size_t i = 0; i < n; i++) {
            if (fs[i].isZero()) continue;
            r += fs[i] * gs[i];
        }
        return r;
    }
};
using Group1 = Group<G1>;
using Group2 = Group<G2>;
using GroupT = Group<Fp12>;

// Primitive 2^32-th root of unity for BLS12-381 Fr (set in init_groups).
extern Fr OMEGA;

// Pairing wrapper: e : G1 x G2 -> GT (Fp12)
GroupT pairing(const Group1& g1, const Group2& g2);

// Product pairing with one Miller-loop batch and one final exponentiation:
//   prod_i e(g1[i], g2[i]).
// This preserves the equation while avoiding one final exponentiation per term.
GroupT pairing_product(const std::vector<Group1>& g1, const std::vector<Group2>& g2);

// Initialize curve and global group constants.
void init_groups();

inline bool isPowerOfTwo(size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

inline size_t nextPow2(size_t n)
{
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// bitN such that 2^bitN == n (n must be power of two).
inline size_t log2N(size_t n)
{
    size_t bitN = 0;
    while ((size_t(1) << bitN) < n)
        bitN++;
    return bitN;
}
