#include <cstdio>
#include <elf.h>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <fstream>

constexpr size_t MMIO_BASE      = 0x1000'0000;
constexpr size_t PROGRAM_BASE   = 0x8000'0000;
constexpr size_t MEMORY_SIZE    = 1024 * 1024 * 128;

enum class Opcode : uint8_t {
    LUI         = 0b011'0111,
    AUIPC       = 0b001'0111,
    JAL         = 0b110'1111,
    JALR        = 0b110'0111,
    BRANCH      = 0b110'0011,
    LOAD        = 0b000'0011,
    STORE       = 0b010'0011,
    IMM         = 0b001'0011,
    OP          = 0b011'0011,
    FENCE       = 0b000'1111,
    SYSTEM      = 0b111'0011,
    ILLEGAL0    = 0b000'0000,
    ILLEGAL1    = 0b111'1111,
};

namespace riscv {
inline uint8_t rd(uint32_t inst) { return (inst >> 7) & 0x1F; }
inline uint8_t rs1(uint32_t inst) { return (inst >> 15) & 0x1F; }
inline uint8_t rs2(uint32_t inst) { return (inst >> 20) & 0x1F; }
inline uint8_t funct3(uint32_t inst) { return (inst >> 12) & 0x07; }
inline uint8_t funct7(uint32_t inst) { return (inst >> 25) & 0x7F; }
inline Opcode  opcode(uint32_t inst) { return static_cast<Opcode>(inst & 0x7F); }

// https://github.com/NJU-ProjectN/nemu/blob/e13af789353868b40f7cede54fb26f182272ed14/include/macro.h#L86-L88
#define BITMASK(bits) ((1ull << (bits)) - 1)
#define BITS(hi, lo) (((inst) >> (lo)) & BITMASK((hi) - (lo) + 1)) // similar to x[hi:lo] in verilog
#define SEXT(x, len) ({ struct { int64_t n : len; } __x = { .n = x }; (uint64_t)__x.n; })
//? how to verify the correctness of static_cast<int64_t>BITS(hi, lo) ?
#define EXTRACT(hi, lo, shift) (SEXT(static_cast<int64_t>BITS(hi, lo), (hi - lo + 1)) << shift)

template<typename T> inline constexpr T immI(uint32_t inst) { return EXTRACT(31, 20,  0); }
template<typename T> inline constexpr T immU(uint32_t inst) { return EXTRACT(31, 12, 12); }
template<typename T> inline constexpr T immS(uint32_t inst) { return EXTRACT(31, 25, 5) | EXTRACT(11, 7, 0); }
template<typename T> inline constexpr T immB(uint32_t inst) {
    return EXTRACT(31, 31, 12) | EXTRACT( 7,  7, 11) | EXTRACT(30, 25,  5) | EXTRACT(11,  8, 1);
}
template<typename T> inline constexpr T immJ(uint32_t inst) {
    return EXTRACT(31, 31, 20) | EXTRACT(19, 12, 12) | EXTRACT(21, 21, 11) | EXTRACT(30, 21, 1);
}
}

template <typename T> class CPU;
template <typename T> class Memory;
template <typename T> class Instruction;
template <typename T> class Decoder;

template <typename T> struct ArchType { using type = T; };

template<typename T>
class Memory {
public:
    Memory() : mem(MEMORY_SIZE, 0) {}

    T read(T addr, size_t len) const {
        if ((addr < PROGRAM_BASE) || (addr + len - PROGRAM_BASE) > MEMORY_SIZE) {
            throw std::out_of_range("Memory read out of bounds");
        }

        switch (len) {
            case 1: return mem[addr - PROGRAM_BASE];
            case 2: return *reinterpret_cast<const uint16_t *>(&mem[addr - PROGRAM_BASE]);
            case 4: return *reinterpret_cast<const uint32_t *>(&mem[addr - PROGRAM_BASE]);
            case 8: return *reinterpret_cast<const uint64_t *>(&mem[addr - PROGRAM_BASE]);
            default:
                throw std::invalid_argument("Invalid read length");
        }
        return 0;
    }

    void write(T addr, size_t len, T value) {
        if ((addr < PROGRAM_BASE) || (addr + len - PROGRAM_BASE) > MEMORY_SIZE) {
            throw std::out_of_range("Memory write out of bounds");
        }

        switch (len) {
            case 1: mem[addr - PROGRAM_BASE] = value & 0xFF; break;
            case 2: *reinterpret_cast<uint16_t *>(&mem[addr - PROGRAM_BASE]) = value & 0xFFFF; break;
            case 4: *reinterpret_cast<uint32_t *>(&mem[addr - PROGRAM_BASE]) = value & 0xFFFF'FFFF; break;
            case 8: *reinterpret_cast<uint64_t *>(&mem[addr - PROGRAM_BASE]) = value; break;
            default:
                throw std::invalid_argument("Invalid write length");
        }
    }

    void load(const std::vector<uint8_t> &data, T addr) {
        if ((addr < PROGRAM_BASE) || (addr + data.size() - PROGRAM_BASE) > MEMORY_SIZE) {
            throw std::out_of_range("Memory write out of bounds");
        }
        std::copy(data.begin(), data.end(), mem.begin() + addr - PROGRAM_BASE);
    }

    void fill(T addr, size_t size, uint8_t value = 0) {
        if ((addr < PROGRAM_BASE) || (addr + size - PROGRAM_BASE) > MEMORY_SIZE) {
            throw std::out_of_range("Memory fill out of bounds");
        }
        std::fill(mem.begin() + addr - PROGRAM_BASE, mem.begin() + addr - PROGRAM_BASE + size, value);
    }

private:
    std::vector<uint8_t> mem;
};

template <typename T>
class CPU {
public:
    CPU(T pc, Memory<T> &memory) : pc(pc), memory(memory) {
        regs[0] = 0;
        regs[2] = PROGRAM_BASE + MEMORY_SIZE;
    }

    void load(const char *filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file");
        }

        using ELF_Ehdr = typename std::conditional<std::is_same_v<T, uint32_t>, Elf32_Ehdr, Elf64_Ehdr>::type;
        using ELF_Phdr = typename std::conditional<std::is_same_v<T, uint32_t>, Elf32_Phdr, Elf64_Phdr>::type;

        ELF_Ehdr ehdr;
        ELF_Phdr phdr;

        file.seekg(0, std::ios::beg);

        if (file.read(reinterpret_cast<char *>(&ehdr), sizeof(ehdr)).gcount() != sizeof(ehdr)) {
            throw std::runtime_error("Failed to read ELF header");
        }

        if (ehdr.e_machine != EM_RISCV) {
            throw std::runtime_error("Not a RISC-V ELF file");
        }

        if (*reinterpret_cast<uint32_t *>(ehdr.e_ident) != 0x464C457F) {
            throw std::runtime_error("Invalid ELF magic number");
        }

        for (int i = 0; i < ehdr.e_phnum; ++i) {
            file.seekg(ehdr.e_phoff + i * sizeof(phdr), std::ios::beg);
            if (file.read(reinterpret_cast<char *>(&phdr), sizeof(phdr)).gcount() != sizeof(phdr)) {
                throw std::runtime_error("Failed to read program header");
            }

            if (phdr.p_type == PT_LOAD) {
                if (phdr.p_flags > 0) {
                    file.seekg(phdr.p_offset, std::ios::beg);
                    std::vector<uint8_t> buffer(phdr.p_filesz);
                    if (file.read(reinterpret_cast<char *>(buffer.data()), phdr.p_filesz).gcount() != phdr.p_filesz) {
                        throw std::runtime_error("Failed to read program segment");
                    }
                    memory.load(buffer, phdr.p_vaddr);
                }

                if (phdr.p_memsz > phdr.p_filesz) {
                    memory.fill(phdr.p_vaddr + phdr.p_filesz, phdr.p_memsz - phdr.p_filesz);
                }
            }
        }
        this->pc = ehdr.e_entry;
    }

    void run() {
        while (true) {
            uint32_t inst = memory.read(pc, 4);
            auto instruction = Decoder<T>::Decode(inst);
            printf("PC: 0x%llx, Instruction: 0x%08x\n", (unsigned long long)pc, inst);
            if (!instruction) {
                fprintf(stderr, "Unknown instruction at PC: 0x%llx\n", (unsigned long long)(pc - 4));
                break;
            }

            pc = instruction->execute(*this);
            regs[0] = 0;
        }
    }

public:
    T pc;
    std::array<T, 32> regs;
    Memory<T> &memory;
};

template <typename T>
class Instruction {
public:
    Instruction(uint32_t inst) : instruction(inst), opcode(riscv::opcode(inst)) {}
    virtual ~Instruction() = default;

    virtual T execute(CPU<T> &cpu) = 0;

protected:
    Opcode opcode;
    uint32_t instruction;
};

template <typename T>
class RTypeInstruction : public Instruction<T> {
public:
    RTypeInstruction(uint32_t inst) : Instruction<T>(inst) {
        rd      = riscv::rd(inst);
        rs1     = riscv::rs1(inst);
        rs2     = riscv::rs2(inst);
        funct3  = riscv::funct3(inst);
        funct7  = riscv::funct7(inst);
    }

    T execute(CPU<T> &cpu) override {
        switch (funct3) {
            case 0b000: // ADD or SUB
                cpu.regs[rd] = cpu.regs[rs1] + (funct7 ? -cpu.regs[rs2] : cpu.regs[rs2]);
                break;
            case 0b001: // SLL
                if constexpr (std::is_same_v<T, uint32_t>) {
                    cpu.regs[rd] = cpu.regs[rs1] << (cpu.regs[rs2] & 0x1F);
                }
                else {
                    cpu.regs[rd] = cpu.regs[rs1] << (cpu.regs[rs2] & 0x3F);
                }
                break;
            case 0b010: // SLT
                cpu.regs[rd] = static_cast<typename std::make_signed<T>::type>(cpu.regs[rs1]) <
                            static_cast<typename std::make_signed<T>::type>(cpu.regs[rs2]);
                break;
            case 0b011: // SLTU
                cpu.regs[rd] = cpu.regs[rs1] < cpu.regs[rs2];
                break;
            case 0b100: // XOR
                cpu.regs[rd] = cpu.regs[rs1] ^ cpu.regs[rs2];
                break;
            case 0b101: // SRL or SRA
                uint8_t shift;
                if constexpr (std::is_same_v<T, uint32_t>) {
                    shift = cpu.regs[rs2] & 0x1F;
                } else {
                    shift = cpu.regs[rs2] & 0x3F;
                }

                if (funct7 == 0b0000000) { // SRL
                    cpu.regs[rd] = cpu.regs[rs1] >> shift;
                } else { // SRA
                    cpu.regs[rd] = static_cast<typename std::make_signed<T>::type>(cpu.regs[rs1]) >> shift;
                }
                break;
            case 0b110: // OR
                cpu.regs[rd] = cpu.regs[rs1] | cpu.regs[rs2];
                break;
            case 0b111: // AND
                cpu.regs[rd] = cpu.regs[rs1] & cpu.regs[rs2];
                break;
            default:
                printf("Unknown funct3 0x%08x in instruction 0x%08x\n", funct3, this->instruction);
                break;
        }
        return cpu.pc + 4;
    }

private:
    uint8_t rd, funct3, rs1, rs2, funct7;
};

template <typename T>
class ITypeInstruction : public Instruction<T> {
public:
    ITypeInstruction(uint32_t inst) : Instruction<T>(inst) {
        rd      = riscv::rd(inst);
        rs1     = riscv::rs1(inst);
        imm     = riscv::immI<T>(inst);
        funct3  = riscv::funct3(inst);
    }

    T execute(CPU<T> &cpu) override {
        if (this->opcode == Opcode::JALR) {
            cpu.regs[rd] = cpu.pc + 4;
            return (cpu.regs[rs1] + imm) & ~1;
        } else if (this->opcode == Opcode::LOAD) {
            switch (funct3) {
                case 0b000: // LB
                    cpu.regs[rd] = static_cast<int8_t>(cpu.memory.read(cpu.regs[rs1] + imm, 1));
                    break;
                case 0b001: // LH
                    cpu.regs[rd] = static_cast<int16_t>(cpu.memory.read(cpu.regs[rs1] + imm, 2));
                    break;
                case 0b010: // LW
                    cpu.regs[rd] = static_cast<int32_t>(cpu.memory.read(cpu.regs[rs1] + imm, 4));
                    break;
                case 0b100: // LBU
                    cpu.regs[rd] = cpu.memory.read(cpu.regs[rs1] + imm, 1);
                    break;
                case 0b101: // LHU
                    cpu.regs[rd] = cpu.memory.read(cpu.regs[rs1] + imm, 2);
                    break;
                default:
                    printf("Unknown funct3 0x%08x in instruction 0x%08x\n", funct3, this->instruction);
                    break;
            }
        } else if (this->opcode == Opcode::IMM) {
            switch (funct3) {
                case 0b000: // ADDI
                    cpu.regs[rd] = cpu.regs[rs1] + imm;
                    break;
                case 0b010: // SLLI
                    if constexpr (std::is_same_v<T, uint32_t>) {
                        cpu.regs[rd] = cpu.regs[rs1] << (imm & 0x1F);
                    } else {
                        cpu.regs[rd] = cpu.regs[rs1] << (imm & 0x3F);
                    }
                    break;
                case 0b011: // SLLIU
                    cpu.regs[rd] = cpu.regs[rs1] < imm;
                    break;
                case 0b100: // XORI
                    cpu.regs[rd] = cpu.regs[rs1] ^ imm;
                    break;
                case 0b110: // ORI
                    cpu.regs[rd] = cpu.regs[rs1] | imm;
                    break;
                case 0b111: // ANDI
                    cpu.regs[rd] = cpu.regs[rs1] & imm;
                    break;
            }
        } else {
            printf("Unknown instruction 0x%08x\n", this->instruction);
        }
        return cpu.pc + 4;
    }

private:
    uint8_t rs1, rd, funct3;
    std::make_signed_t<T> imm;
};

template <typename T>
class STypeInstruction : public Instruction<T> {
public:
    STypeInstruction(uint32_t inst) : Instruction<T>(inst) {
        rs1    = riscv::rs1(inst);
        rs2    = riscv::rs2(inst);
        imm    = riscv::immS<T>(inst);
        funct3 = riscv::funct3(inst);
    }

    T execute(CPU<T> &cpu) override {
        if (this->opcode == Opcode::STORE) {
            switch (funct3) {
                case 0b000: // SB
                    cpu.memory.write(cpu.regs[rs1] + imm, 1, cpu.regs[rs2] & 0xFF);
                    break;
                case 0b001: // SH
                    cpu.memory.write(cpu.regs[rs1] + imm, 2, cpu.regs[rs2] & 0xFFFF);
                    break;
                case 0b010: // SW
                    cpu.memory.write(cpu.regs[rs1] + imm, 4, cpu.regs[rs2]);
                    break;
                default:
                    printf("Unknown funct3 0x%08x in instruction 0x%08x\n", funct3, this->instruction);
                    break;
            }
        } else {
            printf("Unknown opcode 0x%08x in instruction 0x%08x\n", static_cast<uint8_t>(this->opcode), this->instruction);
        }
        return cpu.pc + 4;
    }

private:
    uint8_t rs1, rs2, funct3;
    std::make_signed_t<T> imm;
};

template <typename T>
class BTypeInstruction : public Instruction<T> {
public:
    BTypeInstruction(uint32_t inst) : Instruction<T>(inst) {
        rs1     = riscv::rs1(inst);
        rs2     = riscv::rs2(inst);
        imm     = riscv::immB<T>(inst);
        funct3  = riscv::funct3(inst);
    }

    T execute(CPU<T> &cpu) override {
        switch (funct3) {
            case 0b000: // BEQ
                if (cpu.regs[rs1] == cpu.regs[rs2]) {
                    return cpu.pc + imm;
                }
                break;
            case 0b001: // BNE
                if (cpu.regs[rs1] != cpu.regs[rs2]) {
                    return cpu.pc + imm;
                }
                break;
            case 0b100: // BLT
                if (cpu.regs[rs1] < cpu.regs[rs2]) {
                    return cpu.pc + imm;
                }
                break;
            case 0b101: // BGE
                if (static_cast<typename std::make_signed_t<T>>(cpu.regs[rs1]) >= static_cast<typename std::make_signed_t<T>>(cpu.regs[rs2])) {
                    return cpu.pc + imm;
                }
                break;
            case 0b110: // BLTU
                if (static_cast<T>(cpu.regs[rs1]) < static_cast<T>(cpu.regs[rs2])) {
                    return cpu.pc + imm;
                }
                break;
            case 0b111: // BGEU
                if (static_cast<T>(cpu.regs[rs1]) >= static_cast<T>(cpu.regs[rs2])) {
                    return cpu.pc + imm;
                }
                break;
            default:
                printf("Unknown funct3 0x%08x in instruction 0x%08x\n", funct3, this->instruction);
                break;
        }
        return cpu.pc + 4;
    }

private:
    uint8_t funct3, rs1, rs2;
    std::make_signed_t<T> imm;
};

template <typename T>
class UTypeInstruction : public Instruction<T> {
public:
    UTypeInstruction(uint32_t inst) : Instruction<T>(inst) {
        rd  = riscv::rd(inst);
        imm = riscv::immU<T>(inst);
    }

    T execute(CPU<T> &cpu) override {
        if (this->opcode == Opcode::LUI) {
            cpu.regs[rd] = imm;
        } else if (this->opcode == Opcode::AUIPC) {
            cpu.regs[rd] = cpu.pc + imm;
        } else {
            printf("Unknown instruction 0x%08x\n", this->instruction);
        }
        return cpu.pc + 4;
    }

private:
    uint8_t rd;
    std::make_signed_t<T> imm;
};

template <typename T>
class JTypeInstruction : public Instruction<T> {
public:
    JTypeInstruction(uint32_t inst) : Instruction<T>(inst) {
        rd = riscv::rd(inst);
        imm = riscv::immJ<T>(inst);
    }

    T execute(CPU<T> &cpu) override {
        if (this->opcode == Opcode::JAL) {
            cpu.regs[rd] = cpu.pc + 4;
            return cpu.pc + imm;
        } else {
            printf("Unknown opcode 0x%08x in instruction 0x%08x\n", static_cast<uint8_t>(this->opcode), this->instruction);
        }
        return cpu.pc + 4;
    }

private:
    uint8_t rd;
    std::make_signed_t<T> imm;
};

template <typename T>
class Decoder {
public:
    static std::unique_ptr<Instruction<T>> Decode(uint32_t inst) {
        switch (riscv::opcode(inst)) {
            case Opcode::LUI:
            case Opcode::AUIPC:
                return std::make_unique<UTypeInstruction<T>>(inst);
            case Opcode::LOAD:
            case Opcode::JALR:
            case Opcode::IMM:
                return std::make_unique<ITypeInstruction<T>>(inst);
            case Opcode::JAL:
                return std::make_unique<JTypeInstruction<T>>(inst);
            case Opcode::BRANCH:
                return std::make_unique<BTypeInstruction<T>>(inst);
            case Opcode::STORE:
                return std::make_unique<STypeInstruction<T>>(inst);
            case Opcode::OP:
                return std::make_unique<RTypeInstruction<T>>(inst);
            default: {
                printf("Unknown Opcode 0x%08x\n", static_cast<uint8_t>(riscv::opcode(inst)));
                return nullptr;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <ELF file>\n", argv[0]);
        return 1;
    }

    try {
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) {
            fprintf(stderr, "Failed to open file: %s\n", argv[1]);
            return 1;
        }

        file.seekg(0, std::ios::beg);

        std::array<char, EI_NIDENT> buffer;
        if (file.read(reinterpret_cast<char *>(buffer.data()), EI_NIDENT).gcount() != EI_NIDENT) {
            fprintf(stderr, "Failed to read ELF header\n");
            return 1;
        }

        file.close();

        auto CPUArch = [&](auto arch) {
            Memory<typename decltype(arch)::type> memory;
            CPU<typename decltype(arch)::type> cpu(0, memory);
            cpu.load(argv[1]);
            cpu.run();
        };

        if (buffer[EI_CLASS] == ELFCLASS32) {
            CPUArch(ArchType<uint32_t>());
        } else if (buffer[EI_CLASS] == ELFCLASS64) {
            CPUArch(ArchType<uint64_t>());
        } else {
            fprintf(stderr, "Unsupported ELF class: %d\n", buffer[EI_CLASS]);
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}