#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bls12_381_wrapper.hpp"
#include "polynomial.hpp"

class ProofTranscript {
public:
    explicit ProofTranscript(std::string domain_separator = "Terrae transcript")
    {
        absorb_string("transcript.domain", domain_separator);
    }

    void absorb_bytes(std::string_view label, const void* data, size_t size)
    {
        const auto* bytes = static_cast<const uint8_t*>(data);
        append_frame(0x01, label, bytes, size);
    }

    void absorb_string(std::string_view label, std::string_view value)
    {
        absorb_bytes(label, value.data(), value.size());
    }

    void absorb_size(std::string_view label, size_t value)
    {
        const uint64_t widened = static_cast<uint64_t>(value);
        std::array<uint8_t, sizeof(widened)> bytes{};
        write_u64_be(bytes.data(), widened);
        absorb_bytes(label, bytes.data(), bytes.size());
    }

    void absorb_bool(std::string_view label, bool value)
    {
        const uint8_t byte = value ? 1 : 0;
        absorb_bytes(label, &byte, sizeof(byte));
    }

    std::array<uint8_t, 32> squeeze_hash(std::string_view label)
    {
        std::vector<uint8_t> input = state_;
        input.push_back(0x7f);
        append_u64(input, squeeze_count_);
        append_u64(input, static_cast<uint64_t>(label.size()));
        input.insert(input.end(), label.begin(), label.end());

        const std::array<uint8_t, 32> digest = sha256(input);
        append_frame(0x02, label, digest.data(), digest.size());
        ++squeeze_count_;
        return digest;
    }

    Fr squeeze_fr(std::string_view label)
    {
        const auto digest = squeeze_hash(label);
        Fr out;
        out.setHashOf(digest.data(), digest.size());
        return out;
    }

    Fr squeeze_nonzero_fr(std::string_view label)
    {
        for (size_t attempt = 0;; ++attempt) {
            std::string derived_label(label);
            if (attempt != 0) {
                derived_label += ".retry.";
                derived_label += std::to_string(attempt);
            }
            Fr out = squeeze_fr(derived_label);
            if (!out.isZero()) {
                return out;
            }
        }
    }

    const std::vector<uint8_t>& state() const noexcept
    {
        return state_;
    }

private:
    static constexpr std::array<uint32_t, 64> K = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u,
    };

    static uint32_t rotr(uint32_t x, uint32_t n)
    {
        return (x >> n) | (x << (32u - n));
    }

    static uint32_t read_u32_be(const uint8_t* bytes)
    {
        return (static_cast<uint32_t>(bytes[0]) << 24u) | (static_cast<uint32_t>(bytes[1]) << 16u) |
               (static_cast<uint32_t>(bytes[2]) << 8u) | static_cast<uint32_t>(bytes[3]);
    }

    static void write_u32_be(uint8_t* out, uint32_t value)
    {
        out[0] = static_cast<uint8_t>((value >> 24u) & 0xffu);
        out[1] = static_cast<uint8_t>((value >> 16u) & 0xffu);
        out[2] = static_cast<uint8_t>((value >> 8u) & 0xffu);
        out[3] = static_cast<uint8_t>(value & 0xffu);
    }

    static void write_u64_be(uint8_t* out, uint64_t value)
    {
        for (size_t i = 0; i < 8; ++i) {
            out[i] = static_cast<uint8_t>((value >> ((7 - i) * 8u)) & 0xffu);
        }
    }

    static void append_u64(std::vector<uint8_t>& out, uint64_t value)
    {
        const size_t offset = out.size();
        out.resize(offset + 8);
        write_u64_be(out.data() + offset, value);
    }

    static std::array<uint8_t, 32> sha256(const std::vector<uint8_t>& input)
    {
        std::vector<uint8_t> padded = input;
        const uint64_t bit_len = static_cast<uint64_t>(padded.size()) * 8u;
        padded.push_back(0x80u);
        while ((padded.size() % 64u) != 56u) {
            padded.push_back(0x00u);
        }
        const size_t len_offset = padded.size();
        padded.resize(len_offset + 8u);
        write_u64_be(padded.data() + len_offset, bit_len);

        uint32_t h0 = 0x6a09e667u;
        uint32_t h1 = 0xbb67ae85u;
        uint32_t h2 = 0x3c6ef372u;
        uint32_t h3 = 0xa54ff53au;
        uint32_t h4 = 0x510e527fu;
        uint32_t h5 = 0x9b05688cu;
        uint32_t h6 = 0x1f83d9abu;
        uint32_t h7 = 0x5be0cd19u;

        std::array<uint32_t, 64> w{};
        for (size_t offset = 0; offset < padded.size(); offset += 64u) {
            for (size_t i = 0; i < 16; ++i) {
                w[i] = read_u32_be(padded.data() + offset + i * 4u);
            }
            for (size_t i = 16; i < 64; ++i) {
                const uint32_t s0 = rotr(w[i - 15], 7u) ^ rotr(w[i - 15], 18u) ^ (w[i - 15] >> 3u);
                const uint32_t s1 = rotr(w[i - 2], 17u) ^ rotr(w[i - 2], 19u) ^ (w[i - 2] >> 10u);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }

            uint32_t a = h0;
            uint32_t b = h1;
            uint32_t c = h2;
            uint32_t d = h3;
            uint32_t e = h4;
            uint32_t f = h5;
            uint32_t g = h6;
            uint32_t h = h7;

            for (size_t i = 0; i < 64; ++i) {
                const uint32_t s1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
                const uint32_t ch = (e & f) ^ ((~e) & g);
                const uint32_t temp1 = h + s1 + ch + K[i] + w[i];
                const uint32_t s0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
                const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                const uint32_t temp2 = s0 + maj;

                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }

            h0 += a;
            h1 += b;
            h2 += c;
            h3 += d;
            h4 += e;
            h5 += f;
            h6 += g;
            h7 += h;
        }

        std::array<uint8_t, 32> digest{};
        write_u32_be(digest.data() + 0, h0);
        write_u32_be(digest.data() + 4, h1);
        write_u32_be(digest.data() + 8, h2);
        write_u32_be(digest.data() + 12, h3);
        write_u32_be(digest.data() + 16, h4);
        write_u32_be(digest.data() + 20, h5);
        write_u32_be(digest.data() + 24, h6);
        write_u32_be(digest.data() + 28, h7);
        return digest;
    }

    void append_frame(uint8_t tag, std::string_view label, const uint8_t* data, size_t size)
    {
        state_.push_back(tag);
        append_u64(state_, static_cast<uint64_t>(label.size()));
        state_.insert(state_.end(), label.begin(), label.end());
        append_u64(state_, static_cast<uint64_t>(size));
        state_.insert(state_.end(), data, data + size);
    }

    std::vector<uint8_t> state_;
    uint64_t squeeze_count_ = 0;
};

inline std::string transcript_representation(const Fr& value)
{
    return value.getStr(16);
}

inline std::string transcript_representation(const Group1& value)
{
    return value.value().getStr(16);
}

inline std::string transcript_representation(const Group2& value)
{
    return value.value().getStr(16);
}

inline std::string transcript_representation(const GroupT& value)
{
    return value.value().getStr(16);
}

inline std::string transcript_representation(const poly& value)
{
    const std::vector<Fr>& coeffs = value.coefficients();
    std::string out = "poly[";
    out += std::to_string(coeffs.size());
    out += "]:";
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += coeffs[i].getStr(16);
    }
    return out;
}
