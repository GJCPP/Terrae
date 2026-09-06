#include "plonk/bundle.hpp"

#include "plonk/prover.hpp"
#include "plonk/transcript.hpp"
#include "plonk/verifier.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint8_t kPlonkBinaryBundleMagic[] = {'P', 'L', 'B', 'B'};

class plonkBundleWriter {
public:
    plonkBundleWriter()
    {
        proof_.bytes.insert(proof_.bytes.end(),
                            std::begin(kPlonkBinaryBundleMagic),
                            std::end(kPlonkBinaryBundleMagic));
    }

    SerializedProof take() const
    {
        return proof_;
    }

    void write_size(size_t value)
    {
        const uint64_t widened = static_cast<uint64_t>(value);
        if (static_cast<size_t>(widened) != value) {
            throw std::runtime_error("serialize_plonk_proof_bundle: size does not fit into uint64");
        }
        for (size_t i = 0; i < 8; ++i) {
            proof_.bytes.push_back(static_cast<uint8_t>((widened >> ((7 - i) * 8u)) & 0xffu));
        }
    }

    void write_array(const std::array<uint8_t, 32>& value)
    {
        proof_.bytes.insert(proof_.bytes.end(), value.begin(), value.end());
    }

    void write_bytes(const std::vector<uint8_t>& value)
    {
        write_size(value.size());
        proof_.bytes.insert(proof_.bytes.end(), value.begin(), value.end());
    }

private:
    SerializedProof proof_;
};

class plonkBundleReader {
public:
    explicit plonkBundleReader(const SerializedProof& proof)
        : bytes_(proof.bytes)
    {
        read_magic();
    }

    bool fully_consumed() const noexcept
    {
        return offset_ == bytes_.size();
    }

    size_t read_size()
    {
        if (offset_ > bytes_.size() || 8 > bytes_.size() - offset_) {
            throw std::runtime_error("deserialize_plonk_proof_bundle: bundle underflow while reading size");
        }

        uint64_t value = 0;
        for (size_t i = 0; i < 8; ++i) {
            value = (value << 8u) | bytes_[offset_ + i];
        }
        offset_ += 8;
        if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("deserialize_plonk_proof_bundle: size does not fit into size_t");
        }
        return static_cast<size_t>(value);
    }

    std::array<uint8_t, 32> read_array()
    {
        if (offset_ > bytes_.size() || 32 > bytes_.size() - offset_) {
            throw std::runtime_error("deserialize_plonk_proof_bundle: bundle underflow while reading digest");
        }
        std::array<uint8_t, 32> out{};
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = bytes_[offset_ + i];
        }
        offset_ += out.size();
        return out;
    }

    std::vector<uint8_t> read_bytes()
    {
        const size_t size = read_size();
        if (offset_ > bytes_.size() || size > bytes_.size() - offset_) {
            throw std::runtime_error("deserialize_plonk_proof_bundle: bundle underflow while reading proof bytes");
        }
        std::vector<uint8_t> out(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                                 bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + size));
        offset_ += size;
        return out;
    }

private:
    void read_magic()
    {
        if (bytes_.size() < sizeof(kPlonkBinaryBundleMagic)) {
            throw std::runtime_error("deserialize_plonk_proof_bundle: bundle underflow while reading header");
        }
        for (size_t i = 0; i < sizeof(kPlonkBinaryBundleMagic); ++i) {
            if (bytes_[i] != kPlonkBinaryBundleMagic[i]) {
                throw std::runtime_error("deserialize_plonk_proof_bundle: invalid bundle header");
            }
        }
        offset_ = sizeof(kPlonkBinaryBundleMagic);
    }

    const std::vector<uint8_t>& bytes_;
    size_t offset_ = 0;
};

} // namespace

std::array<uint8_t, 32> plonk_verification_key_digest(const plonkVerificationKey& verification_key)
{
    ProofTranscript transcript("PLONK verification key");
    absorb_plonk_verification_key(transcript, verification_key);
    return transcript.squeeze_hash("plonk.vk.digest");
}

plonkProofBundle make_plonk_proof_bundle(const plonkVerificationKey& verification_key,
                                         const std::vector<Fr>& public_inputs,
                                         const plonkProof& proof)
{
    plonkProofBundle bundle;
    bundle.proof = serialize_plonk_proof(proof);
    bundle.public_input_count = public_inputs.size();
    bundle.verification_key_digest = plonk_verification_key_digest(verification_key);
    return bundle;
}

SerializedProof serialize_plonk_proof_bundle(const plonkProofBundle& bundle)
{
    plonkBundleWriter writer;
    writer.write_size(bundle.public_input_count);
    writer.write_array(bundle.verification_key_digest);
    writer.write_bytes(bundle.proof.bytes);
    return writer.take();
}

plonkProofBundle deserialize_plonk_proof_bundle(const SerializedProof& serialized)
{
    plonkBundleReader reader(serialized);

    plonkProofBundle bundle;
    bundle.public_input_count = reader.read_size();
    bundle.verification_key_digest = reader.read_array();
    bundle.proof.bytes = reader.read_bytes();

    if (!reader.fully_consumed()) {
        throw std::runtime_error("deserialize_plonk_proof_bundle: trailing bundle bytes");
    }
    return bundle;
}

plonkProofBundle prove_plonk_bundle(const plonkProvingKey& proving_key,
                                    const plonkWitness& witness,
                                    const std::vector<Fr>& public_inputs)
{
    plonkProofBundle bundle;
    bundle.proof = prove_plonk_serialized(proving_key, witness, public_inputs);
    bundle.public_input_count = public_inputs.size();
    bundle.verification_key_digest = plonk_verification_key_digest(derive_plonk_verification_key(proving_key));
    return bundle;
}

bool verify_plonk_bundle(const plonkVerificationKey& verification_key,
                         const std::vector<Fr>& public_inputs,
                         const plonkProofBundle& bundle)
{
    if (bundle.public_input_count != public_inputs.size()) {
        return false;
    }
    if (bundle.verification_key_digest != plonk_verification_key_digest(verification_key)) {
        return false;
    }
    return verify_plonk_serialized(verification_key, public_inputs, bundle.proof);
}

bool verify_plonk_serialized_bundle(const plonkVerificationKey& verification_key,
                                    const std::vector<Fr>& public_inputs,
                                    const SerializedProof& serialized)
{
    try {
        return verify_plonk_bundle(verification_key,
                                   public_inputs,
                                   deserialize_plonk_proof_bundle(serialized));
    } catch (const std::exception&) {
        return false;
    }
}
