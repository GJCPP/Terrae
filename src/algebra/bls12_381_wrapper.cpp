#include "bls12_381_wrapper.hpp"

Fr OMEGA;

template<>
Group<G1>& Group<G1>::zero()
{
    static Group<G1> z;
    return z;
}

template<>
Group<G1>& Group<G1>::one()
{
    static Group<G1> o;
    return o;
}

template<>
Group<G2>& Group<G2>::zero()
{
    static Group<G2> z;
    return z;
}

template<>
Group<G2>& Group<G2>::one()
{
    static Group<G2> o;
    return o;
}

template<>
Group<Fp12>& Group<Fp12>::zero()
{
    static Group<Fp12> z;
    return z;
}

template<>
Group<Fp12>& Group<Fp12>::one()
{
    static Group<Fp12> o;
    return o;
}


GroupT pairing(const Group1& g1, const Group2& g2)
{
    GroupT ret;
    pairing(ret.value(), g1.value(), g2.value());
    return ret;
}

GroupT pairing_product(const std::vector<Group1>& g1, const std::vector<Group2>& g2)
{
    if (g1.size() != g2.size()) {
        throw std::invalid_argument("pairing_product: input sizes differ");
    }
    if (g1.empty()) {
        return GroupT::zero();
    }

    std::vector<G1> raw_g1;
    std::vector<G2> raw_g2;
    raw_g1.reserve(g1.size());
    raw_g2.reserve(g2.size());
    for (size_t i = 0; i < g1.size(); ++i) {
        raw_g1.push_back(g1[i].value());
        raw_g2.push_back(g2[i].value());
    }

    GroupT ret;
    millerLoopVec(ret.value(), raw_g1.data(), raw_g2.data(), raw_g1.size());
    finalExp(ret.value(), ret.value());
    return ret;
}

void init_groups()
{
    initPairing(mcl::BLS12_381);

    Group1::zero().value().clear();
    Group2::zero().value().clear();
    GroupT::zero().value().setOne();  // GT is multiplicative: identity is 1

    hashAndMapToG1(Group1::one().value(), "one", 3);
    hashAndMapToG2(Group2::one().value(), "one", 3);
    GroupT::one() = pairing(Group1::one(), Group2::one());

    bool b;
    OMEGA.setStr(&b, "212d79e5b416b6f0fd56dc8d168d6c0c4024ff270b3e0941b788f500b912f1f", 16);
    assert(b);
}
