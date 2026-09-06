#include "algebra_utils.hpp"
#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <mcl/ntt.hpp>
#include <stdexcept>
#include <vector>

namespace {

size_t bitRev(size_t i, size_t bitN)
{
    size_t r = 0;
    for (size_t b = 0; b < bitN; b++) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

void bitRevArray(Fr* xs, size_t n, size_t bitN)
{
    for (size_t i = 0; i < n; i++) {
        size_t j = bitRev(i, bitN);
        if (i < j) {
            Fr t = xs[i];
            xs[i] = xs[j];
            xs[j] = t;
        }
    }
}

void nttWithWs(Fr* xs, size_t n, const Fr* ws, bool inv)
{
    size_t bitN = log2N(n);
    bitRevArray(xs, n, bitN);

    size_t h = n;
    for (size_t L = 1; L < n; L *= 2) {
        h >>= 1;
        for (size_t i = 0; i < n; i += L * 2) {
            size_t idx = 0;
            for (size_t j = i; j < L + i; j++) {
                Fr tmp;
                Fr::mul(tmp, xs[j + L], ws[inv ? n - idx : idx]);
                Fr::sub(xs[j + L], xs[j], tmp);
                Fr::add(xs[j], xs[j], tmp);
                idx += h;
            }
        }
    }

    if (inv) {
        Fr invN;
        Fr::inv(invN, n);
        for (size_t i = 0; i < n; i++) {
            Fr::mul(xs[i], xs[i], invN);
        }
    }
}

using SparseTerms = std::vector<std::pair<size_t, Fr>>;

std::vector<Fr> dense_naive_mul(const std::vector<Fr>& a, const std::vector<Fr>& b)
{
    std::vector<Fr> out(a.size() + b.size() - 1, Fr(0));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            Fr prod;
            Fr::mul(prod, a[i], b[j]);
            Fr::add(out[i + j], out[i + j], prod);
        }
    }
    return out;
}

std::vector<Fr> dense_fft_mul(const std::vector<Fr>& a, const std::vector<Fr>& b)
{
    const size_t result_size = a.size() + b.size() - 1;
    const size_t n = nextPow2(result_size);

    std::vector<Fr> fa(n, Fr(0));
    std::vector<Fr> fb(n, Fr(0));
    for (size_t i = 0; i < a.size(); ++i) fa[i] = a[i];
    for (size_t i = 0; i < b.size(); ++i) fb[i] = b[i];

    const Fr omega = get_root_of_unity(n);
    ntt(fa.data(), n, omega);
    ntt(fb.data(), n, omega);
    for (size_t i = 0; i < n; ++i) {
        fa[i] *= fb[i];
    }
    intt(fa.data(), n, omega);

    return std::vector<Fr>(fa.begin(), fa.begin() + result_size);
}

bool should_use_fft_mul(size_t lhs_size, size_t rhs_size)
{
    if (lhs_size == 0 || rhs_size == 0) {
        return false;
    }
    const size_t naive_cost = lhs_size * rhs_size;
    const size_t fft_size = nextPow2(lhs_size + rhs_size - 1);
    const size_t fft_cost = fft_size * log2N(fft_size) * 12;
    return fft_cost < naive_cost;
}

bool should_keep_sparse_product(const SparseTerms& left, const SparseTerms& right)
{
    if (left.empty() || right.empty()) {
        return false;
    }
    const size_t sparse_cost = left.size() * right.size();
    const size_t span = left.back().first + right.back().first + 1;
    return sparse_cost <= 128 || sparse_cost * sparse_cost <= span * 8;
}

std::vector<Fr> sparse_dense_mul(const SparseTerms& sparse, const std::vector<Fr>& dense)
{
    std::vector<Fr> out(sparse.back().first + dense.size(), Fr(0));
    for (const auto& [exp, c] : sparse) {
        for (size_t i = 0; i < dense.size(); ++i) {
            Fr prod;
            Fr::mul(prod, c, dense[i]);
            Fr::add(out[exp + i], out[exp + i], prod);
        }
    }
    return out;
}

SparseTerms build_sparse_terms(const std::vector<Fr>& dense)
{
    SparseTerms sparse_terms;
    sparse_terms.reserve(dense.size());
    for (size_t i = 0; i < dense.size(); ++i) {
        if (!dense[i].isZero()) {
            sparse_terms.emplace_back(i, dense[i]);
        }
    }
    return sparse_terms;
}

void accumulate_sparse_term(SparseTerms& terms, size_t exp, const Fr& delta)
{
    if (delta.isZero()) {
        return;
    }
    auto it = std::lower_bound(
        terms.begin(), terms.end(), exp,
        [](const auto& term, size_t target_exp) { return term.first < target_exp; });
    if (it == terms.end() || it->first != exp) {
        terms.insert(it, {exp, delta});
        return;
    }

    Fr sum;
    Fr::add(sum, it->second, delta);
    if (sum.isZero()) {
        terms.erase(it);
    } else {
        it->second = sum;
    }
}

void add_sparse_into_dense(std::vector<Fr>& out,
                           const SparseTerms& sparse_terms,
                           bool negate)
{
    if (sparse_terms.empty()) {
        return;
    }
    out.resize(std::max(out.size(), sparse_terms.back().first + 1), Fr(0));
    for (const auto& [exp, c] : sparse_terms) {
        if (negate) {
            Fr::sub(out[exp], out[exp], c);
        } else {
            Fr::add(out[exp], out[exp], c);
        }
    }
}

SparseTerms sparse_add_sub(const SparseTerms& left,
                          const SparseTerms& right,
                          bool subtract)
{
    SparseTerms out;
    out.reserve(left.size() + right.size());
    size_t i = 0;
    size_t j = 0;
    while (i < left.size() || j < right.size()) {
        if (j == right.size() || (i < left.size() && left[i].first < right[j].first)) {
            out.push_back(left[i++]);
            continue;
        }
        if (i == left.size() || right[j].first < left[i].first) {
            Fr value = right[j].second;
            if (subtract) {
                Fr::neg(value, value);
            }
            if (!value.isZero()) {
                out.emplace_back(right[j].first, value);
            }
            ++j;
            continue;
        }

        Fr sum = left[i].second;
        if (subtract) {
            Fr::sub(sum, sum, right[j].second);
        } else {
            Fr::add(sum, sum, right[j].second);
        }
        if (!sum.isZero()) {
            out.emplace_back(left[i].first, sum);
        }
        ++i;
        ++j;
    }
    return out;
}

SparseTerms sparse_mul(const SparseTerms& left, const SparseTerms& right)
{
    SparseTerms out;
    for (const auto& [exp_a, c_a] : left) {
        for (const auto& [exp_b, c_b] : right) {
            Fr prod;
            Fr::mul(prod, c_a, c_b);
            accumulate_sparse_term(out, exp_a + exp_b, prod);
        }
    }
    return out;
}

void trim_dense(std::vector<Fr>& v)
{
    while (!v.empty() && v.back().isZero()) {
        v.pop_back();
    }
}

std::pair<poly, poly> div_dense_by_dense(const std::vector<Fr>& dividend, const std::vector<Fr>& divisor)
{
    poly result;
    const std::vector<Fr>& a = dividend;
    const std::vector<Fr>& b = divisor;
    if (b.empty() || std::all_of(b.begin(), b.end(), [](const Fr& f) { return f.isZero(); })) {
        return {result, result};
    }
    if (a.empty()) {
        result.init_zero();
        return {result, result};
    }

    size_t deg_a = a.size() - 1;
    size_t deg_b = b.size() - 1;
    if (deg_a < deg_b) {
        poly zero;
        zero.init_zero();
        return {zero, poly_from_coeff(a)};
    }

    std::vector<Fr> q(deg_a - deg_b + 1, Fr(0));
    std::vector<Fr> r = a;

    Fr inv_lead_b;
    Fr::inv(inv_lead_b, b[deg_b]);

    for (size_t k = deg_a + 1; k-- > deg_b;) {
        if (r[k].isZero()) {
            continue;
        }
        Fr factor;
        Fr::mul(factor, r[k], inv_lead_b);
        q[k - deg_b] = factor;

        for (size_t j = 0; j <= deg_b; ++j) {
            Fr prod;
            Fr::mul(prod, factor, b[j]);
            Fr::sub(r[k - deg_b + j], r[k - deg_b + j], prod);
        }
    }

    poly quotient;
    quotient.init_with_coeff(q);
    poly remainder;
    remainder.init_with_coeff(r);
    return {quotient, remainder};
}

std::pair<poly, poly> div_dense_by_sparse_terms(
    const std::vector<Fr>& dividend,
    const SparseTerms& divisor_terms)
{
    poly result;
    if (divisor_terms.empty()) {
        return {result, result};
    }
    if (dividend.empty()) {
        result.init_zero();
        return {result, result};
    }

    std::vector<Fr> r = dividend;
    trim_dense(r);
    if (r.empty()) {
        poly zero;
        zero.init_zero();
        return {zero, zero};
    }

    const size_t deg_a = r.size() - 1;
    const size_t deg_b = divisor_terms.back().first;
    const Fr& lead_b = divisor_terms.back().second;
    if (deg_a < deg_b) {
        poly zero;
        zero.init_zero();
        return {zero, poly_from_coeff(r)};
    }

    std::vector<Fr> q(deg_a - deg_b + 1, Fr(0));
    Fr inv_lead_b;
    Fr::inv(inv_lead_b, lead_b);

    for (size_t k = deg_a + 1; k-- > deg_b;) {
        if (r[k].isZero()) {
            continue;
        }

        Fr factor;
        Fr::mul(factor, r[k], inv_lead_b);
        q[k - deg_b] = factor;

        const size_t shift = k - deg_b;
        for (const auto& [exp, c] : divisor_terms) {
            Fr prod;
            Fr::mul(prod, factor, c);
            Fr::sub(r[shift + exp], r[shift + exp], prod);
        }
    }

    poly quotient;
    quotient.init_with_coeff(q);
    poly remainder;
    remainder.init_with_coeff(r);
    return {quotient, remainder};
}
mcl::Ntt<Fr>& cached_standard_ntt_domain(size_t n)
{
    thread_local std::map<size_t, std::unique_ptr<mcl::Ntt<Fr>>> cache;
    const auto it = cache.find(n);
    if (it != cache.end()) {
        return *it->second;
    }

    auto domain = std::make_unique<mcl::Ntt<Fr>>();
    if (!domain->init(n)) {
        throw std::runtime_error("failed to initialize cached NTT domain");
    }
    auto [inserted, ok] = cache.emplace(n, std::move(domain));
    (void)ok;
    return *inserted->second;
}

} // namespace

bool ntt(Fr* arr, size_t n, const Fr& omega)
{
    if (n == 0 || !isPowerOfTwo(n)) return false;
    if (n == 1) return true;

    mcl::Ntt<Fr>& domain = cached_standard_ntt_domain(n);
    if (domain.getW() == omega) {
        domain.ntt(arr);
        return true;
    }

    std::vector<Fr> ws(n + 1);
    mcl::local::initPowSeq(ws.data(), omega, n + 1);
    nttWithWs(arr, n, ws.data(), false);
    return true;
}

bool intt(Fr* arr, size_t n, const Fr& omega)
{
    if (n == 0 || !isPowerOfTwo(n)) return false;
    if (n == 1) return true;

    mcl::Ntt<Fr>& domain = cached_standard_ntt_domain(n);
    if (domain.getW() == omega) {
        domain.intt(arr);
        return true;
    }

    std::vector<Fr> ws(n + 1);
    mcl::local::initPowSeq(ws.data(), omega, n + 1);
    nttWithWs(arr, n, ws.data(), true);
    return true;
}

bool get_root_of_unity(Fr& root, size_t n)
{
    if (n == 0 || !isPowerOfTwo(n) || n > (1ull << 32)) return false;
    if (OMEGA.isZero()) {
        throw std::runtime_error("OMEGA is uninitialized");
    }
    static std::map<size_t, Fr> cache;
    const auto it = cache.find(n);
    if (it != cache.end()) {
        root = it->second;
        return true;
    }
    Fr::pow(root, OMEGA, (1ull << 32) / n);
    cache.emplace(n, root);
    return true;
}

Fr get_root_of_unity(size_t n)
{
    Fr ret;
    bool b = get_root_of_unity(ret, n);
    if (!b) ret = 0;
    return ret;
}

void poly::trim(std::vector<Fr>& v)
{
    while (!v.empty() && v.back().isZero()) {
        v.pop_back();
    }
}

size_t poly::count_non_zero(const std::vector<Fr>& v)
{
    size_t count = 0;
    for (const Fr& x : v) {
        if (!x.isZero()) {
            ++count;
        }
    }
    return count;
}

bool poly::should_keep_sparse(size_t span, size_t non_zero_terms)
{
    if (non_zero_terms == 0) return false;
    return non_zero_terms <= 8 || non_zero_terms * non_zero_terms <= span;
}

void poly::clear_evals()
{
    evals.clear();
}

void poly::ensure_coeff() const
{
    if (!coeff.empty()) return;

    if (!evals.empty()) {
        size_t n = nextPow2(evals.size());
        coeff = evals;
        coeff.resize(n);
        intt(coeff.data(), n, get_root_of_unity(n));
        trim(coeff);
        return;
    }

    if (sparse_terms_.empty()) {
        return;
    }

    const size_t max_exp = sparse_terms_.back().first;
    coeff.assign(max_exp + 1, Fr(0));
    for (const auto& [exp, c] : sparse_terms_) {
        coeff[exp] = c;
    }
    trim(coeff);
}

void poly::ensure_evals() const
{
    if (!evals.empty()) return;
    ensure_coeff();
    if (coeff.empty()) return;

    size_t n = nextPow2(coeff.size());
    evals = coeff;
    evals.resize(n);
    ntt(evals.data(), n, get_root_of_unity(n));
}

void poly::ensure_sparse_terms() const
{
    if (!sparse_terms_.empty()) return;
    ensure_coeff();
    sparse_terms_ = build_sparse_terms(coeff);
}

void poly::set_dense_coefficients_(std::vector<Fr>&& dense)
{
    trim(dense);
    const size_t span = dense.size();
    const size_t non_zero = count_non_zero(dense);
    coeff = std::move(dense);
    evals.clear();

    if (should_keep_sparse(span, non_zero)) {
        sparse_terms_ = build_sparse_terms(coeff);
        coeff.clear();
        storage_ = Storage::SPARSE;
        return;
    }

    sparse_terms_.clear();
    storage_ = Storage::DENSE;
}

void poly::set_sparse_terms_(std::vector<std::pair<size_t, Fr>>&& sparse_terms)
{
    coeff.clear();
    evals.clear();

    if (sparse_terms.empty()) {
        sparse_terms_.clear();
        storage_ = Storage::DENSE;
        return;
    }

    const size_t span = sparse_terms.back().first + 1;
    const size_t non_zero = sparse_terms.size();
    if (should_keep_sparse(span, non_zero)) {
        sparse_terms_ = std::move(sparse_terms);
        storage_ = Storage::SPARSE;
        return;
    }

    sparse_terms_ = std::move(sparse_terms);
    ensure_coeff();
    sparse_terms_.clear();
    storage_ = Storage::DENSE;
}

void poly::init_zero()
{
    coeff.clear();
    evals.clear();
    sparse_terms_.clear();
    storage_ = Storage::DENSE;
}

void poly::init_with_coeff(const Fr c[], size_t deg_plus_1)
{
    std::vector<Fr> dense(c, c + deg_plus_1);
    set_dense_coefficients_(std::move(dense));
}

void poly::init_with_coeff(const std::vector<Fr>& c)
{
    if (c.empty()) {
        init_zero();
        return;
    }
    std::vector<Fr> dense = c;
    set_dense_coefficients_(std::move(dense));
}

void poly::init_with_evals(const Fr e[], size_t n)
{
    size_t nextN = nextPow2(n);
    evals.assign(e, e + n);
    evals.resize(nextN);
    coeff.clear();
    sparse_terms_.clear();
    storage_ = Storage::DENSE;
}

void poly::init_with_evals(const std::vector<Fr>& e)
{
    init_with_evals(e.data(), e.size());
}

void poly::init_as_vanishing(const Fr x[], size_t n)
{
    if (n == 0) {
        init_with_coeff(std::vector<Fr>{Fr(1)});
        return;
    }

    std::vector<Fr> z = {Fr(1)};
    for (size_t j = 0; j < n; ++j) {
        std::vector<Fr> new_z(z.size() + 1, Fr(0));
        for (size_t k = 0; k < z.size(); ++k) {
            Fr temp;
            Fr::mul(temp, z[k], -x[j]);
            Fr::add(new_z[k], new_z[k], temp);
            Fr::add(new_z[k + 1], new_z[k + 1], z[k]);
        }
        z.swap(new_z);
    }
    set_dense_coefficients_(std::move(z));
}

void poly::init_as_vanishing(const std::vector<Fr>& x)
{
    init_as_vanishing(x.data(), x.size());
}

void poly::init_with_interpolation(const Fr x[], const Fr y[], size_t n)
{
    if (n == 0) {
        init_zero();
        return;
    }
    if (n == 1) {
        init_with_coeff(std::vector<Fr>{y[0]});
        return;
    }

    poly Z;
    Z.init_as_vanishing(x, n);

    std::vector<Fr> out(n, Fr(0));
    for (size_t i = 0; i < n; ++i) {
        poly divisor;
        std::vector<Fr> div_coeff(2);
        Fr::neg(div_coeff[0], x[i]);
        div_coeff[1] = Fr(1);
        divisor.init_with_coeff(div_coeff);

        poly basis = Z / divisor;

        Fr denom(1);
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            Fr delta;
            Fr::sub(delta, x[i], x[j]);
            Fr::mul(denom, denom, delta);
        }

        Fr inv_denom;
        Fr::inv(inv_denom, denom);
        Fr scale;
        Fr::mul(scale, y[i], inv_denom);

        const std::vector<Fr>& b = basis.coefficients();
        for (size_t k = 0; k < b.size(); ++k) {
            Fr temp;
            Fr::mul(temp, b[k], scale);
            Fr::add(out[k], out[k], temp);
        }
    }
    set_dense_coefficients_(std::move(out));
}

void poly::init_with_interpolation(const std::vector<Fr>& x, const std::vector<Fr>& y)
{
    if (x.size() != y.size()) {
        throw std::runtime_error("poly::init_with_interpolation: x and y must have the same size");
    }
    init_with_interpolation(x.data(), y.data(), x.size());
}

void poly::init_with_terms(const std::vector<std::pair<size_t, Fr>>& terms)
{
    SparseTerms sparse_terms;
    sparse_terms.reserve(terms.size());
    for (const auto& [exp, c] : terms) {
        accumulate_sparse_term(sparse_terms, exp, c);
    }
    set_sparse_terms_(std::move(sparse_terms));
}

void poly::add_term(size_t exp, const Fr& c)
{
    if (c.isZero()) return;
    ensure_sparse_terms();
    accumulate_sparse_term(sparse_terms_, exp, c);

    coeff.clear();
    evals.clear();

    SparseTerms normalized = sparse_terms_;
    set_sparse_terms_(std::move(normalized));
}

Fr poly::eval(const Fr& x) const
{
    if (storage_ == Storage::SPARSE) {
        ensure_sparse_terms();
        Fr result(0);
        for (const auto& [exp, c] : sparse_terms_) {
            Fr x_pow = pow_fr(x, exp);
            Fr term;
            Fr::mul(term, c, x_pow);
            Fr::add(result, result, term);
        }
        return result;
    }

    ensure_coeff();
    if (coeff.empty()) return Fr(0);
    if (coeff.size() == 1) return coeff.front();

    Fr y = coeff.back();
    for (size_t j = coeff.size() - 1; j > 0; --j) {
        size_t i = j - 1;
        Fr::mul(y, y, x);
        Fr::add(y, y, coeff[i]);
    }
    return y;
}

bool poly::is_zero() const
{
    ensure_coeff();
    return coeff.empty();
}

const std::vector<Fr>& poly::coefficients() const
{
    ensure_coeff();
    return coeff;
}

const std::vector<Fr>& poly::evaluations() const
{
    ensure_evals();
    return evals;
}

std::vector<Fr> poly::evaluations_on_domain(size_t n) const
{
    if (!isPowerOfTwo(n) || n == 0) {
        throw std::runtime_error("poly::evaluations_on_domain: n must be a non-zero power of two");
    }
    if (!evals.empty() && evals.size() == n) {
        return evals;
    }

    std::vector<Fr> domain_evals(n, Fr(0));
    if (storage_ == Storage::SPARSE) {
        ensure_sparse_terms();
        for (const auto& [exponent, coefficient] : sparse_terms_) {
            const size_t reduced_exponent = exponent & (n - 1);
            Fr::add(domain_evals[reduced_exponent], domain_evals[reduced_exponent], coefficient);
        }
    } else {
        ensure_coeff();
        for (size_t exponent = 0; exponent < coeff.size(); ++exponent) {
            const size_t reduced_exponent = exponent & (n - 1);
            Fr::add(domain_evals[reduced_exponent], domain_evals[reduced_exponent], coeff[exponent]);
        }
    }
    ntt(domain_evals.data(), n, get_root_of_unity(n));
    return domain_evals;
}

const std::vector<std::pair<size_t, Fr>>& poly::terms() const
{
    ensure_sparse_terms();
    return sparse_terms_;
}

size_t poly::degree() const
{
    if (storage_ == Storage::SPARSE) {
        ensure_sparse_terms();
        if (sparse_terms_.empty()) return 0;
        return sparse_terms_.back().first;
    }
    ensure_coeff();
    if (coeff.empty()) return 0;
    return coeff.size() - 1;
}

poly poly::column_replicated(size_t n, size_t m) const
{
    if (!isPowerOfTwo(n) || !isPowerOfTwo(m) || n == 0 || m == 0) {
        throw std::runtime_error("poly::column_replicated: n and m must be powers of two");
    }

    const std::vector<Fr> hn_evals = evaluations_on_domain(n);
    std::vector<Fr> evals(n * m, Fr(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            evals[i * m + j] = hn_evals[i];
        }
    }

    poly out;
    out.init_with_evals(evals);
    return out;
}

poly poly::scale_argument(const Fr& scale) const
{
    poly out;
    if (is_sparse()) {
        ensure_sparse_terms();
        for (const auto& [exp, coeff_value] : sparse_terms_) {
            out.add_term(exp, coeff_value * pow_fr(scale, exp));
        }
        return out;
    }

    ensure_coeff();
    std::vector<Fr> scaled(coeff.size(), Fr(0));
    Fr scale_power(1);
    for (size_t i = 0; i < coeff.size(); ++i) {
        Fr::mul(scaled[i], coeff[i], scale_power);
        Fr::mul(scale_power, scale_power, scale);
    }
    out.init_with_coeff(scaled);
    return out;
}

poly poly::operator+(const poly& p) const
{
    if (is_sparse() && p.is_sparse()) {
        ensure_sparse_terms();
        p.ensure_sparse_terms();
        poly r;
        r.set_sparse_terms_(sparse_add_sub(sparse_terms_, p.sparse_terms_, false));
        return r;
    }

    if (is_sparse() != p.is_sparse()) {
        const poly& sparse = is_sparse() ? *this : p;
        const poly& dense = is_sparse() ? p : *this;
        dense.ensure_coeff();
        sparse.ensure_sparse_terms();

        std::vector<Fr> out = dense.coeff;
        add_sparse_into_dense(out, sparse.sparse_terms_, false);

        poly r;
        r.set_dense_coefficients_(std::move(out));
        return r;
    }

    ensure_coeff();
    p.ensure_coeff();
    const std::vector<Fr>& a = coeff;
    const std::vector<Fr>& b = p.coeff;
    std::vector<Fr> out(std::max(a.size(), b.size()), Fr(0));
    for (size_t i = 0; i < out.size(); ++i) {
        Fr left = (i < a.size()) ? a[i] : Fr(0);
        Fr right = (i < b.size()) ? b[i] : Fr(0);
        Fr::add(out[i], left, right);
    }
    poly r;
    r.set_dense_coefficients_(std::move(out));
    return r;
}

poly poly::operator-(const poly& p) const
{
    if (is_sparse() && p.is_sparse()) {
        ensure_sparse_terms();
        p.ensure_sparse_terms();
        poly r;
        r.set_sparse_terms_(sparse_add_sub(sparse_terms_, p.sparse_terms_, true));
        return r;
    }

    if (is_sparse() != p.is_sparse()) {
        const poly& dense = is_sparse() ? p : *this;
        const poly& sparse = is_sparse() ? *this : p;
        dense.ensure_coeff();
        sparse.ensure_sparse_terms();

        std::vector<Fr> out = dense.coeff;
        if (is_sparse()) {
            for (Fr& c : out) {
                Fr::neg(c, c);
            }
        }
        add_sparse_into_dense(out, sparse.sparse_terms_, !is_sparse());

        poly r;
        r.set_dense_coefficients_(std::move(out));
        return r;
    }

    ensure_coeff();
    p.ensure_coeff();
    const std::vector<Fr>& a = coeff;
    const std::vector<Fr>& b = p.coeff;
    std::vector<Fr> out(std::max(a.size(), b.size()), Fr(0));
    for (size_t i = 0; i < out.size(); ++i) {
        Fr left = (i < a.size()) ? a[i] : Fr(0);
        Fr right = (i < b.size()) ? b[i] : Fr(0);
        Fr::sub(out[i], left, right);
    }
    poly r;
    r.set_dense_coefficients_(std::move(out));
    return r;
}

poly poly::operator*(const poly& p) const
{
    if (is_zero() || p.is_zero()) {
        poly r;
        r.init_zero();
        return r;
    }

    if (is_sparse() && p.is_sparse()) {
        ensure_sparse_terms();
        p.ensure_sparse_terms();
        if (!should_keep_sparse_product(sparse_terms_, p.sparse_terms_)) {
            std::vector<Fr> out = should_use_fft_mul(sparse_terms_.back().first + 1,
                                                     p.sparse_terms_.back().first + 1)
                ? dense_fft_mul(coefficients(), p.coefficients())
                : dense_naive_mul(coefficients(), p.coefficients());
            poly r;
            r.set_dense_coefficients_(std::move(out));
            return r;
        }
        poly r;
        r.set_sparse_terms_(sparse_mul(sparse_terms_, p.sparse_terms_));
        return r;
    }

    if (is_sparse() != p.is_sparse()) {
        const poly& sparse = is_sparse() ? *this : p;
        const poly& dense = is_sparse() ? p : *this;
        sparse.ensure_sparse_terms();
        dense.ensure_coeff();

        if (dense.coeff.empty() || sparse.sparse_terms_.empty()) {
            poly r;
            r.init_zero();
            return r;
        }

        const size_t sparse_cost = sparse.sparse_terms_.size() * dense.coeff.size();
        const size_t dense_span = sparse.sparse_terms_.back().first + 1;
        std::vector<Fr> out = (should_use_fft_mul(dense_span, dense.coeff.size()) &&
                               sparse_cost > 128)
            ? dense_fft_mul(sparse.coefficients(), dense.coeff)
            : sparse_dense_mul(sparse.sparse_terms_, dense.coeff);

        poly r;
        r.set_dense_coefficients_(std::move(out));
        return r;
    }

    ensure_coeff();
    p.ensure_coeff();
    std::vector<Fr> out = should_use_fft_mul(coeff.size(), p.coeff.size())
        ? dense_fft_mul(coeff, p.coeff)
        : dense_naive_mul(coeff, p.coeff);
    poly r;
    r.set_dense_coefficients_(std::move(out));
    return r;
}

poly poly::operator/(const poly& p) const
{
    if (!is_sparse() && p.is_sparse()) {
        ensure_coeff();
        p.ensure_sparse_terms();
        return div_dense_by_sparse_terms(coeff, p.sparse_terms_).first;
    }

    ensure_coeff();
    p.ensure_coeff();
    return div_dense_by_dense(coeff, p.coeff).first;
}

std::pair<poly, poly> div(const poly& a, const poly& b)
{
    if (!a.is_sparse() && b.is_sparse()) {
        a.ensure_coeff();
        b.ensure_sparse_terms();
        return div_dense_by_sparse_terms(a.coeff, b.sparse_terms_);
    }
    return div_dense_by_dense(a.coefficients(), b.coefficients());
}

poly poly::operator+(const Fr& c) const
{
    if (is_sparse()) {
        ensure_sparse_terms();
        SparseTerms out = sparse_terms_;
        accumulate_sparse_term(out, 0, c);
        poly r;
        r.set_sparse_terms_(std::move(out));
        return r;
    }

    ensure_coeff();
    std::vector<Fr> out = coeff;
    if (out.empty()) {
        if (!c.isZero()) {
            out.push_back(c);
        }
    } else {
        Fr::add(out[0], out[0], c);
    }
    poly r;
    r.set_dense_coefficients_(std::move(out));
    return r;
}

poly poly::operator-(const Fr& c) const
{
    Fr neg_c;
    Fr::neg(neg_c, c);
    return (*this) + neg_c;
}

poly poly::operator*(const Fr& c) const
{
    poly r;
    if (is_zero() || c.isZero()) {
        r.init_zero();
        return r;
    }

    if (is_sparse()) {
        SparseTerms out;
        ensure_sparse_terms();
        for (const auto& [exp, coeff_value] : sparse_terms_) {
            Fr prod;
            Fr::mul(prod, coeff_value, c);
            if (!prod.isZero()) {
                out.emplace_back(exp, prod);
            }
        }
        r.set_sparse_terms_(std::move(out));
        return r;
    }

    ensure_coeff();
    std::vector<Fr> out = coeff;
    for (Fr& x : out) {
        Fr::mul(x, x, c);
    }
    r.set_dense_coefficients_(std::move(out));
    return r;
}

poly poly::operator/(const Fr& c) const
{
    poly r;
    if (is_zero() || c.isZero()) {
        r.init_zero();
        return r;
    }

    Fr inv_c;
    Fr::inv(inv_c, c);
    return (*this) * inv_c;
}
