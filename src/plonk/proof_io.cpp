#include "plonk/proof_io.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint8_t kPlonkBinaryProofMagic[] = {'P', 'L', 'B', '1'};

class plonkProofWriter {
public:
    plonkProofWriter()
    {
        proof_.bytes.insert(proof_.bytes.end(),
                            std::begin(kPlonkBinaryProofMagic),
                            std::end(kPlonkBinaryProofMagic));
    }

    SerializedProof take() const
    {
        return proof_;
    }

    void write_fr(const Fr& value)
    {
        write_mcl_value(value, Fr::getByteSize(), "Fr");
    }

    void write_group1(const Group1& value)
    {
        write_mcl_value(value.value(), G1::getSerializedByteSize(), "Group1");
    }

    void write_commitment(const plonkPolynomialCommitment& commitment)
    {
        write_group1(commitment.value);
    }

private:
    template <class T>
    void write_mcl_value(const T& value, size_t encoded_size, const char* label)
    {
        const size_t offset = proof_.bytes.size();
        proof_.bytes.resize(offset + encoded_size);
        const size_t written =
            value.serialize(proof_.bytes.data() + offset, encoded_size, mcl::IoSerialize);
        if (written != encoded_size) {
            throw std::runtime_error(std::string("serialize_plonk_proof: failed to serialize ") + label);
        }
    }

    SerializedProof proof_;
};

class plonkProofReader {
public:
    explicit plonkProofReader(const SerializedProof& proof)
        : bytes_(proof.bytes)
    {
        read_magic();
    }

    bool fully_consumed() const noexcept
    {
        return offset_ == bytes_.size();
    }

    Fr read_fr()
    {
        Fr value;
        read_mcl_value(value, Fr::getByteSize(), "Fr");
        return value;
    }

    Group1 read_group1()
    {
        G1 value;
        read_mcl_value(value, G1::getSerializedByteSize(), "Group1");
        return Group1(value);
    }

    plonkPolynomialCommitment read_commitment()
    {
        return plonkPolynomialCommitment{read_group1()};
    }

private:
    void read_magic()
    {
        if (bytes_.size() < sizeof(kPlonkBinaryProofMagic)) {
            throw std::runtime_error("deserialize_plonk_proof: proof underflow while reading header");
        }
        for (size_t i = 0; i < sizeof(kPlonkBinaryProofMagic); ++i) {
            if (bytes_[i] != kPlonkBinaryProofMagic[i]) {
                throw std::runtime_error("deserialize_plonk_proof: invalid proof header");
            }
        }
        offset_ = sizeof(kPlonkBinaryProofMagic);
    }

    template <class T>
    void read_mcl_value(T& value, size_t encoded_size, const char* label)
    {
        if (offset_ > bytes_.size() || encoded_size > bytes_.size() - offset_) {
            throw std::runtime_error(std::string("deserialize_plonk_proof: proof underflow while reading ") + label);
        }
        const size_t read =
            value.deserialize(bytes_.data() + offset_, encoded_size, mcl::IoSerialize);
        if (read != encoded_size) {
            throw std::runtime_error(std::string("deserialize_plonk_proof: failed to deserialize ") + label);
        }
        offset_ += encoded_size;
    }

    const std::vector<uint8_t>& bytes_;
    size_t offset_ = 0;
};

} // namespace

SerializedProof serialize_plonk_proof(const plonkProof& proof)
{
    plonkProofWriter writer;

    writer.write_commitment(proof.a_commitment);
    writer.write_commitment(proof.b_commitment);
    writer.write_commitment(proof.c_commitment);
    writer.write_commitment(proof.f_commitment);
    writer.write_commitment(proof.h1_commitment);
    writer.write_commitment(proof.h2_commitment);
    writer.write_commitment(proof.z_commitment);
    writer.write_commitment(proof.z_lookup_commitment);
    writer.write_commitment(proof.t_lo_commitment);
    writer.write_commitment(proof.t_mid_commitment);
    writer.write_commitment(proof.t_hi_commitment);

    writer.write_fr(proof.a_eval);
    writer.write_fr(proof.b_eval);
    writer.write_fr(proof.c_eval);
    writer.write_fr(proof.sigma1_eval);
    writer.write_fr(proof.sigma2_eval);
    writer.write_fr(proof.f_eval);
    writer.write_fr(proof.lookup_table_eval);
    writer.write_fr(proof.lookup_table_omega_eval);
    writer.write_fr(proof.z_omega_eval);
    writer.write_fr(proof.z_lookup_omega_eval);
    writer.write_fr(proof.h1_omega_eval);
    writer.write_fr(proof.h2_eval);
    writer.write_fr(proof.public_input_eval);
    writer.write_fr(proof.t_eval);
    writer.write_fr(proof.r_eval);

    writer.write_commitment(proof.w_zeta_commitment);
    writer.write_commitment(proof.w_zeta_omega_commitment);
    return writer.take();
}

plonkProof deserialize_plonk_proof(const SerializedProof& serialized)
{
    plonkProofReader reader(serialized);

    plonkProof proof;
    proof.a_commitment = reader.read_commitment();
    proof.b_commitment = reader.read_commitment();
    proof.c_commitment = reader.read_commitment();
    proof.f_commitment = reader.read_commitment();
    proof.h1_commitment = reader.read_commitment();
    proof.h2_commitment = reader.read_commitment();
    proof.z_commitment = reader.read_commitment();
    proof.z_lookup_commitment = reader.read_commitment();
    proof.t_lo_commitment = reader.read_commitment();
    proof.t_mid_commitment = reader.read_commitment();
    proof.t_hi_commitment = reader.read_commitment();

    proof.a_eval = reader.read_fr();
    proof.b_eval = reader.read_fr();
    proof.c_eval = reader.read_fr();
    proof.sigma1_eval = reader.read_fr();
    proof.sigma2_eval = reader.read_fr();
    proof.f_eval = reader.read_fr();
    proof.lookup_table_eval = reader.read_fr();
    proof.lookup_table_omega_eval = reader.read_fr();
    proof.z_omega_eval = reader.read_fr();
    proof.z_lookup_omega_eval = reader.read_fr();
    proof.h1_omega_eval = reader.read_fr();
    proof.h2_eval = reader.read_fr();
    proof.public_input_eval = reader.read_fr();
    proof.t_eval = reader.read_fr();
    proof.r_eval = reader.read_fr();

    proof.w_zeta_commitment = reader.read_commitment();
    proof.w_zeta_omega_commitment = reader.read_commitment();

    if (!reader.fully_consumed()) {
        throw std::runtime_error("deserialize_plonk_proof: trailing proof bytes");
    }
    return proof;
}
