#include "plonk/lookup.hpp"

#include "algebra_utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

std::string compressed_value_key(const Fr& value)
{
    return value.getStr(16);
}

} // namespace

Fr compress_plonk_lookup_row(const Fr& left,
                             const Fr& right,
                             const Fr& output,
                             const Fr& compression_challenge)
{
    return left + compression_challenge * right +
           compression_challenge * compression_challenge * output;
}

plonkLookupPolynomials build_plonk_lookup_polynomials(const plonkProvingKey& proving_key,
                                                      const plonkWireAssignments& wires,
                                                      const Fr& compression_challenge)
{
    if (wires.a_evals.size() != proving_key.n || wires.b_evals.size() != proving_key.n ||
        wires.c_evals.size() != proving_key.n) {
        throw std::runtime_error("build_plonk_lookup_polynomials: invalid wire assignment shape");
    }

    plonkLookupPolynomials out;
    out.f_evals.assign(proving_key.n, Fr(0));
    out.table_evals.assign(proving_key.n, Fr(0));
    out.h1_evals.assign(proving_key.n, Fr(0));
    out.h2_evals.assign(proving_key.n, Fr(0));

    for (size_t row = 0; row < proving_key.n; ++row) {
        out.table_evals[row] = compress_plonk_lookup_row(proving_key.table_1_evals[row],
                                                         proving_key.table_2_evals[row],
                                                         proving_key.table_3_evals[row],
                                                         compression_challenge);
    }

    const Fr dummy = out.table_evals.empty() ? Fr(0) : out.table_evals.front();
    std::unordered_map<std::string, size_t> first_table_position;
    for (size_t row = 0; row < out.table_evals.size(); ++row) {
        const std::string key = compressed_value_key(out.table_evals[row]);
        if (first_table_position.count(key) == 0) {
            first_table_position.emplace(key, row);
        }
    }

    std::vector<std::pair<size_t, Fr>> sortable;
    sortable.reserve(2 * proving_key.n);

    for (size_t row = 0; row < proving_key.n; ++row) {
        if (!proving_key.qK_evals[row].isZero()) {
            out.f_evals[row] = compress_plonk_lookup_row(wires.a_evals[row],
                                                         wires.b_evals[row],
                                                         wires.c_evals[row],
                                                         compression_challenge);
            const std::string key = compressed_value_key(out.f_evals[row]);
            const auto it = first_table_position.find(key);
            if (it == first_table_position.end()) {
                throw std::runtime_error("build_plonk_lookup_polynomials: lookup witness value is not in the public table");
            }
            sortable.push_back({it->second, out.f_evals[row]});
        } else {
            out.f_evals[row] = dummy;
            sortable.push_back({0, out.f_evals[row]});
        }
    }

    for (size_t row = 0; row < proving_key.n; ++row) {
        sortable.push_back({row, out.table_evals[row]});
    }

    std::stable_sort(sortable.begin(),
                     sortable.end(),
                     [](const std::pair<size_t, Fr>& lhs, const std::pair<size_t, Fr>& rhs) {
                         return lhs.first < rhs.first;
                     });

    if (sortable.size() != 2 * proving_key.n) {
        throw std::runtime_error("build_plonk_lookup_polynomials: invalid sorted lookup shape");
    }
    for (size_t row = 0; row < proving_key.n; ++row) {
        out.h1_evals[row] = sortable[2 * row].second;
        out.h2_evals[row] = sortable[2 * row + 1].second;
    }

    out.f = poly_from_evals(out.f_evals);
    out.table = poly_from_evals(out.table_evals);
    out.h1 = poly_from_evals(out.h1_evals);
    out.h2 = poly_from_evals(out.h2_evals);
    return out;
}

std::vector<Fr> build_plonk_lookup_grand_product_evaluations(const plonkLookupPolynomials& lookup,
                                                             const Fr& omega,
                                                             const Fr& delta,
                                                             const Fr& epsilon)
{
    const size_t n = lookup.f_evals.size();
    if (n == 0 || lookup.table_evals.size() != n || lookup.h1_evals.size() != n ||
        lookup.h2_evals.size() != n) {
        throw std::runtime_error("build_plonk_lookup_grand_product_evaluations: invalid lookup shape");
    }

    std::vector<Fr> z(n, Fr(1));
    for (size_t row = 0; row + 1 < n; ++row) {
        const Fr numerator = (Fr(1) + delta) * (epsilon + lookup.f_evals[row]) *
                             (epsilon * (Fr(1) + delta) + lookup.table_evals[row] +
                              delta * lookup.table_evals[(row + 1) % n]);
        const Fr denominator =
            (epsilon * (Fr(1) + delta) + lookup.h1_evals[row] + delta * lookup.h2_evals[row]) *
            (epsilon * (Fr(1) + delta) + lookup.h2_evals[row] +
             delta * lookup.h1_evals[(row + 1) % n]);
        if (denominator.isZero()) {
            throw std::runtime_error("build_plonk_lookup_grand_product_evaluations: zero denominator");
        }
        Fr denominator_inv;
        Fr::inv(denominator_inv, denominator);
        z[row + 1] = z[row] * numerator * denominator_inv;
    }

    const Fr final_numerator = (Fr(1) + delta) * (epsilon + lookup.f_evals[n - 1]) *
                               (epsilon * (Fr(1) + delta) + lookup.table_evals[n - 1] +
                                delta * lookup.table_evals[0]);
    const Fr final_denominator =
        (epsilon * (Fr(1) + delta) + lookup.h1_evals[n - 1] + delta * lookup.h2_evals[n - 1]) *
        (epsilon * (Fr(1) + delta) + lookup.h2_evals[n - 1] + delta * lookup.h1_evals[0]);
    if (final_denominator.isZero()) {
        throw std::runtime_error("build_plonk_lookup_grand_product_evaluations: zero final denominator");
    }
    Fr final_denominator_inv;
    Fr::inv(final_denominator_inv, final_denominator);
    if (!(z[n - 1] * final_numerator * final_denominator_inv == Fr(1))) {
        throw std::runtime_error("build_plonk_lookup_grand_product_evaluations: lookup grand product does not close");
    }

    (void)omega;
    return z;
}
