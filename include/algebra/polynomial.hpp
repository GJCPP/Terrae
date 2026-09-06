#pragma once

/**
 * @file polynomial.hpp
 * @brief Polynomial utilities over the BLS12-381 scalar field (Fr).
 *
 * The polynomial class adapts between dense and sparse storage. Dense storage
 * is used by default, while genuinely sparse high-span polynomials are kept as
 * terms (exponent, coefficient).
 */

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mcl/bls12_381.hpp>

using namespace mcl::bn;

bool ntt(Fr* arr, size_t n, const Fr& omega);
bool intt(Fr* arr, size_t n, const Fr& omega);
bool get_root_of_unity(Fr& omega, size_t n);
Fr get_root_of_unity(size_t n);

class poly {
public:
    poly() = default;
    ~poly() = default;

    void init_zero();

    void init_with_coeff(const Fr coeff[], size_t deg_plus_1);
    void init_with_coeff(const std::vector<Fr>& coeff);

    void init_with_evals(const Fr evals[], size_t n);
    void init_with_evals(const std::vector<Fr>& evals);

    void init_as_vanishing(const Fr x[], size_t n);
    void init_as_vanishing(const std::vector<Fr>& x);

    void init_with_interpolation(const Fr x[], const Fr y[], size_t n);
    void init_with_interpolation(const std::vector<Fr>& x, const std::vector<Fr>& y);

    void init_with_terms(const std::vector<std::pair<size_t, Fr>>& terms);
    void add_term(size_t exp, const Fr& c);

    Fr eval(const Fr& x) const;
    bool is_zero() const;
    bool is_sparse() const { return storage_ == Storage::SPARSE; }

    poly operator+(const poly& p) const;
    poly operator-(const poly& p) const;
    poly operator*(const poly& p) const;
    poly operator/(const poly& p) const;
    poly operator+(const Fr& c) const;
    poly operator-(const Fr& c) const;
    poly operator*(const Fr& c) const;
    poly operator/(const Fr& c) const;

    const std::vector<Fr>& coefficients() const;
    const std::vector<Fr>& evaluations() const;
    std::vector<Fr> evaluations_on_domain(size_t n) const;
    const std::vector<std::pair<size_t, Fr>>& terms() const;
    size_t degree() const;
    poly scale_argument(const Fr& scale) const;
    poly column_replicated(size_t n, size_t m) const;

private:
    enum class Storage {
        DENSE,
        SPARSE,
    };

    static void trim(std::vector<Fr>& v);
    static size_t count_non_zero(const std::vector<Fr>& v);
    static bool should_keep_sparse(size_t span, size_t non_zero_terms);

    void ensure_coeff() const;
    void ensure_evals() const;
    void ensure_sparse_terms() const;
    void set_dense_coefficients_(std::vector<Fr>&& dense);
    void set_sparse_terms_(std::vector<std::pair<size_t, Fr>>&& sparse_terms);
    void clear_evals();

    friend std::pair<poly, poly> div(const poly& a, const poly& b);

    mutable std::vector<Fr> coeff;
    mutable std::vector<Fr> evals;
    mutable std::vector<std::pair<size_t, Fr>> sparse_terms_;
    Storage storage_ = Storage::DENSE;
};

inline poly poly_from_coeff(Fr coeffs[], size_t deg_plus_1)
{
    poly p;
    p.init_with_coeff(coeffs, deg_plus_1);
    return p;
}

inline poly poly_from_coeff(const std::vector<Fr>& coeffs)
{
    poly p;
    p.init_with_coeff(coeffs);
    return p;
}

inline poly poly_from_evals(const Fr evals[], size_t n)
{
    poly p;
    p.init_with_evals(evals, n);
    return p;
}

inline poly poly_from_evals(const std::vector<Fr>& evals)
{
    poly p;
    p.init_with_evals(evals);
    return p;
}

inline poly poly_from_interpolation(const Fr x[], const Fr y[], size_t n)
{
    poly p;
    p.init_with_interpolation(x, y, n);
    return p;
}

inline poly poly_from_interpolation(const std::vector<Fr>& x, const std::vector<Fr>& y)
{
    if (x.size() != y.size()) {
        throw std::runtime_error("poly_from_interpolation: x and y must have the same size");
    }
    poly p;
    p.init_with_interpolation(x, y);
    return p;
}

inline poly poly_as_vanishing(const Fr x[], size_t n)
{
    poly p;
    p.init_as_vanishing(x, n);
    return p;
}

inline poly poly_as_vanishing(const std::vector<Fr>& x)
{
    poly p;
    p.init_as_vanishing(x);
    return p;
}

inline poly poly_from_terms(const std::vector<std::pair<size_t, Fr>>& terms)
{
    poly p;
    p.init_with_terms(terms);
    return p;
}

inline poly operator+(const Fr& c, const poly& p) { return p + c; }

inline poly operator-(const Fr& c, const poly& p)
{
    std::vector<Fr> out = p.coefficients();
    if (out.empty()) {
        out.push_back(c);
        return poly_from_coeff(out);
    }
    Fr::neg(out[0], out[0]);
    Fr::add(out[0], out[0], c);
    for (size_t i = 1; i < out.size(); ++i) {
        Fr::neg(out[i], out[i]);
    }
    return poly_from_coeff(out);
}

inline poly operator*(const Fr& c, const poly& p) { return p * c; }

std::pair<poly, poly> div(const poly& a, const poly& b);

inline poly vanishing_poly(size_t n)
{
    if (!isPowerOfTwo(n)) {
        throw std::runtime_error("n must be a power of two");
    }
    poly ret;
    ret.add_term(0, -Fr(1));
    ret.add_term(n, Fr(1));
    return ret;
}

inline poly identity_poly()
{
    static poly ret = [] {
        poly p;
        p.add_term(1, Fr(1));
        return p;
    }();
    return ret;
}
