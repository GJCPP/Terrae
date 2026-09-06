#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "transcript.hpp"

enum class ProofMode {
    Prove,
    Verify,
};

enum class ProofEncoding {
    Binary,
    LegacyText,
};

class SerializedProof {
public:
    std::vector<uint8_t> bytes;
};

inline std::string proof_encoding(const Fr& value)
{
    return value.serializeToHexStr();
}

inline Fr decode_fr(const std::string& encoded)
{
    Fr value;
    value.deserializeHexStr(encoded);
    return value;
}

inline std::string proof_encoding(const Group1& value)
{
    return value.value().serializeToHexStr();
}

inline Group1 decode_group1(const std::string& encoded)
{
    Group1 value;
    value.value().deserializeHexStr(encoded);
    return value;
}

inline std::string proof_encoding(const Group2& value)
{
    return value.value().serializeToHexStr();
}

inline Group2 decode_group2(const std::string& encoded)
{
    Group2 value;
    value.value().deserializeHexStr(encoded);
    return value;
}

inline std::string proof_encoding(bool value)
{
    return value ? "1" : "0";
}

inline bool decode_bool(const std::string& encoded)
{
    if (encoded == "1") {
        return true;
    }
    if (encoded == "0") {
        return false;
    }
    throw std::runtime_error("invalid boolean proof encoding");
}

inline constexpr uint8_t kBinaryProofMagic[] = {'T', 'P', 'B', '1'};

inline bool has_binary_proof_magic(const SerializedProof& proof)
{
    if (proof.bytes.size() < sizeof(kBinaryProofMagic)) {
        return false;
    }
    for (size_t i = 0; i < sizeof(kBinaryProofMagic); ++i) {
        if (proof.bytes[i] != kBinaryProofMagic[i]) {
            return false;
        }
    }
    return true;
}

class ProofSession {
public:
    explicit ProofSession(ProofMode mode,
                          SerializedProof proof = {},
                          std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>(),
                          ProofEncoding prover_encoding = ProofEncoding::Binary)
        : mode_(mode)
        , proof_(std::move(proof))
        , transcript_(std::move(transcript))
    {
        if (!transcript_) {
            throw std::runtime_error("ProofSession transcript must not be null");
        }
        if (mode_ == ProofMode::Prove) {
            binary_encoding_ = prover_encoding == ProofEncoding::Binary;
            if (binary_encoding_) {
                proof_.bytes.insert(proof_.bytes.end(), std::begin(kBinaryProofMagic), std::end(kBinaryProofMagic));
            }
        } else {
            binary_encoding_ = has_binary_proof_magic(proof_);
            if (binary_encoding_) {
                offset_ = sizeof(kBinaryProofMagic);
            }
        }
    }

    static std::shared_ptr<ProofSession> prover(
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>(),
        ProofEncoding encoding = ProofEncoding::Binary)
    {
        return std::make_shared<ProofSession>(
            ProofMode::Prove, SerializedProof{}, std::move(transcript), encoding);
    }

    static std::shared_ptr<ProofSession> verifier(const SerializedProof& proof,
                                                  std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        return std::make_shared<ProofSession>(ProofMode::Verify, proof, std::move(transcript));
    }

    // Metadata-only verifier session used by the exact soundness planner. It
    // follows verifier control flow but synthesizes deterministic proof values.
    static std::shared_ptr<ProofSession> counter(
        std::shared_ptr<ProofTranscript> transcript = std::make_shared<ProofTranscript>())
    {
        auto session = std::make_shared<ProofSession>(ProofMode::Verify, SerializedProof{}, std::move(transcript));
        session->count_only_ = true;
        return session;
    }

    bool count_only() const noexcept
    {
        return count_only_;
    }

    ProofMode mode() const noexcept
    {
        return mode_;
    }

    std::shared_ptr<ProofTranscript> shared_transcript() const
    {
        return transcript_;
    }

    ProofTranscript& transcript()
    {
        return *transcript_;
    }

    const ProofTranscript& transcript() const
    {
        return *transcript_;
    }

    SerializedProof take_proof() const
    {
        if (mode_ != ProofMode::Prove) {
            throw std::runtime_error("take_proof requires prover mode");
        }
        return proof_;
    }

    bool proof_fully_consumed() const noexcept
    {
        if (count_only_) {
            return true;
        }
        return offset_ == proof_.bytes.size();
    }

    Fr exchange_fr(std::string_view label, const std::function<Fr()>& prover_fn)
    {
        return exchange_value<Fr>(
            label,
            prover_fn,
            [](const Fr& value) { return proof_encoding(value); },
            [](const std::string& encoded) { return decode_fr(encoded); });
    }

    Group1 exchange_group1(std::string_view label, const std::function<Group1()>& prover_fn)
    {
        return exchange_value<Group1>(
            label,
            prover_fn,
            [](const Group1& value) { return proof_encoding(value); },
            [](const std::string& encoded) { return decode_group1(encoded); });
    }

    bool exchange_bool(std::string_view label, const std::function<bool()>& prover_fn)
    {
        return exchange_value<bool>(
            label,
            prover_fn,
            [](bool value) { return proof_encoding(value); },
            [](const std::string& encoded) { return decode_bool(encoded); });
    }

    Fr transport_fr(const std::function<Fr()>& prover_fn)
    {
        return transport_value<Fr>(
            prover_fn,
            [](const Fr& value) { return proof_encoding(value); },
            [](const std::string& encoded) { return decode_fr(encoded); });
    }

    Group1 transport_group1(const std::function<Group1()>& prover_fn)
    {
        return transport_value<Group1>(
            prover_fn,
            [](const Group1& value) { return proof_encoding(value); },
            [](const std::string& encoded) { return decode_group1(encoded); });
    }

    Group2 transport_group2(const std::function<Group2()>& prover_fn)
    {
        return transport_value<Group2>(
            prover_fn,
            [](const Group2& value) { return proof_encoding(value); },
            [](const std::string& encoded) { return decode_group2(encoded); });
    }

    std::vector<Group1> exchange_group1_vector(std::string_view label,
                                               size_t expected_size,
                                               const std::function<std::vector<Group1>()>& prover_fn)
    {
        std::vector<Group1> values;
        if (count_only_) {
            values.resize(expected_size);
            transcript_->absorb_size(std::string(label) + ".size", values.size());
            for (const Group1& value : values) {
                transcript_->absorb_string(std::string(label) + ".item", proof_encoding(value));
            }
            return values;
        }
        if (mode_ == ProofMode::Prove) {
            if (!prover_fn) {
                throw std::runtime_error("exchange_group1_vector missing prover callback");
            }
            values = prover_fn();
            write_size(values.size());
        } else {
            values.resize(read_size());
        }

        if (expected_size != 0 && values.size() != expected_size) {
            throw std::runtime_error("exchange_group1_vector size mismatch");
        }

        transcript_->absorb_size(std::string(label) + ".size", values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const std::string item_label = std::string(label) + ".item";
            if (mode_ == ProofMode::Prove) {
                const std::string encoded = proof_encoding(values[i]);
                write_proof_value(values[i], encoded);
                transcript_->absorb_string(item_label, encoded);
            } else if (binary_encoding_) {
                const Group1 value = read_proof_value<Group1>([](const std::string& encoded) {
                    return decode_group1(encoded);
                });
                const std::string encoded = proof_encoding(value);
                transcript_->absorb_string(item_label, encoded);
                values[i] = value;
            } else {
                const std::string encoded = read_string();
                transcript_->absorb_string(item_label, encoded);
                values[i] = decode_group1(encoded);
            }
        }
        return values;
    }

    std::vector<Group1> transport_group1_vector(size_t expected_size,
                                                const std::function<std::vector<Group1>()>& prover_fn)
    {
        std::vector<Group1> values;
        if (count_only_) {
            values.resize(expected_size);
            return values;
        }
        if (mode_ == ProofMode::Prove) {
            if (!prover_fn) {
                throw std::runtime_error("transport_group1_vector missing prover callback");
            }
            values = prover_fn();
            write_size(values.size());
        } else {
            values.resize(read_size());
        }

        if (expected_size != 0 && values.size() != expected_size) {
            throw std::runtime_error("transport_group1_vector size mismatch");
        }

        for (size_t i = 0; i < values.size(); ++i) {
            if (mode_ == ProofMode::Prove) {
                write_proof_value(values[i], proof_encoding(values[i]));
            } else {
                values[i] = read_proof_value<Group1>([](const std::string& encoded) {
                    return decode_group1(encoded);
                });
            }
        }
        return values;
    }

private:
    template <class T, class EncodeFn, class DecodeFn>
    T exchange_value(std::string_view label,
                     const std::function<T()>& prover_fn,
                     EncodeFn&& encode,
                     DecodeFn&& decode)
    {
        if (count_only_) {
            T value{};
            if constexpr (std::is_same_v<T, Fr>) {
                value = Fr(1);
            } else if constexpr (std::is_same_v<T, bool>) {
                value = true;
            }
            transcript_->absorb_string(label, encode(value));
            return value;
        }
        if (mode_ == ProofMode::Prove) {
            if (!prover_fn) {
                throw std::runtime_error("exchange_value missing prover callback");
            }
            const T value = prover_fn();
            const std::string encoded = encode(value);
            write_proof_value(value, encoded);
            transcript_->absorb_string(label, encoded);
            return value;
        }

        if (!binary_encoding_) {
            const std::string encoded = read_string();
            transcript_->absorb_string(label, encoded);
            return decode(encoded);
        }

        const T value = read_proof_value<T>(std::forward<DecodeFn>(decode));
        const std::string encoded = encode(value);
        transcript_->absorb_string(label, encoded);
        return value;
    }

    template <class T, class EncodeFn, class DecodeFn>
    T transport_value(const std::function<T()>& prover_fn, EncodeFn&& encode, DecodeFn&& decode)
    {
        if (count_only_) {
            T value{};
            if constexpr (std::is_same_v<T, Fr>) {
                value = Fr(1);
            } else if constexpr (std::is_same_v<T, bool>) {
                value = true;
            }
            return value;
        }
        if (mode_ == ProofMode::Prove) {
            if (!prover_fn) {
                throw std::runtime_error("transport_value missing prover callback");
            }
            const T value = prover_fn();
            write_proof_value(value, encode(value));
            return value;
        }

        return read_proof_value<T>(std::forward<DecodeFn>(decode));
    }

    template <class T>
    void write_proof_value(const T& value, const std::string& encoded)
    {
        if (!binary_encoding_) {
            write_string(encoded);
            return;
        }

        if constexpr (std::is_same_v<T, Fr>) {
            write_mcl_value(value, Fr::getByteSize(), "Fr");
        } else if constexpr (std::is_same_v<T, Group1>) {
            write_mcl_value(value.value(), G1::getSerializedByteSize(), "Group1");
        } else if constexpr (std::is_same_v<T, Group2>) {
            write_mcl_value(value.value(), G2::getSerializedByteSize(), "Group2");
        } else if constexpr (std::is_same_v<T, bool>) {
            proof_.bytes.push_back(value ? 1 : 0);
        } else {
            static_assert(!sizeof(T), "unsupported proof value type");
        }
    }

    template <class T, class DecodeFn>
    T read_proof_value(DecodeFn&& decode)
    {
        if (!binary_encoding_) {
            return decode(read_string());
        }

        if constexpr (std::is_same_v<T, Fr>) {
            Fr value;
            read_mcl_value(value, Fr::getByteSize(), "Fr");
            return value;
        } else if constexpr (std::is_same_v<T, Group1>) {
            G1 raw;
            read_mcl_value(raw, G1::getSerializedByteSize(), "Group1");
            return Group1(raw);
        } else if constexpr (std::is_same_v<T, Group2>) {
            G2 raw;
            read_mcl_value(raw, G2::getSerializedByteSize(), "Group2");
            return Group2(raw);
        } else if constexpr (std::is_same_v<T, bool>) {
            if (offset_ >= proof_.bytes.size()) {
                throw std::runtime_error("proof underflow while reading bool");
            }
            const uint8_t value = proof_.bytes[offset_++];
            if (value > 1) {
                throw std::runtime_error("invalid boolean proof encoding");
            }
            return value != 0;
        } else {
            static_assert(!sizeof(T), "unsupported proof value type");
        }
    }

    template <class T>
    void write_mcl_value(const T& value, size_t encoded_size, const char* label)
    {
        const size_t offset = proof_.bytes.size();
        proof_.bytes.resize(offset + encoded_size);
        const size_t written = value.serialize(proof_.bytes.data() + offset, encoded_size, mcl::IoSerialize);
        if (written != encoded_size) {
            throw std::runtime_error(std::string("failed to serialize proof ") + label);
        }
    }

    template <class T>
    void read_mcl_value(T& value, size_t encoded_size, const char* label)
    {
        if (offset_ + encoded_size > proof_.bytes.size()) {
            throw std::runtime_error(std::string("proof underflow while reading ") + label);
        }
        const size_t read = value.deserialize(proof_.bytes.data() + offset_, encoded_size, mcl::IoSerialize);
        if (read != encoded_size) {
            throw std::runtime_error(std::string("failed to deserialize proof ") + label);
        }
        offset_ += encoded_size;
    }

    void write_size(size_t value)
    {
        const uint64_t widened = static_cast<uint64_t>(value);
        for (size_t i = 0; i < 8; ++i) {
            proof_.bytes.push_back(static_cast<uint8_t>((widened >> ((7 - i) * 8u)) & 0xffu));
        }
    }

    size_t read_size()
    {
        if (offset_ + 8 > proof_.bytes.size()) {
            throw std::runtime_error("proof underflow while reading size");
        }
        uint64_t value = 0;
        for (size_t i = 0; i < 8; ++i) {
            value = (value << 8u) | proof_.bytes[offset_ + i];
        }
        offset_ += 8;
        return static_cast<size_t>(value);
    }

    void write_string(const std::string& value)
    {
        write_size(value.size());
        proof_.bytes.insert(proof_.bytes.end(), value.begin(), value.end());
    }

    std::string read_string()
    {
        const size_t size = read_size();
        if (offset_ + size > proof_.bytes.size()) {
            throw std::runtime_error("proof underflow while reading string");
        }
        std::string out(proof_.bytes.begin() + static_cast<std::ptrdiff_t>(offset_),
                        proof_.bytes.begin() + static_cast<std::ptrdiff_t>(offset_ + size));
        offset_ += size;
        return out;
    }

    ProofMode mode_;
    bool count_only_ = false;
    SerializedProof proof_;
    size_t offset_ = 0;
    bool binary_encoding_ = false;
    std::shared_ptr<ProofTranscript> transcript_;
};
