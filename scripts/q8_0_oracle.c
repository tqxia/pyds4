/* q8_0_oracle.c — minimal Q8_0 dequant oracle for M4 parity tests.
 *
 * Reads Q8_0-packed bytes from stdin (or argv[1]), dequants in-order, writes
 * fp32 little-endian bytes to stdout (or argv[2]). The dequant formula is
 * lifted verbatim from ds4.c::dot_q8_0_row_2 (line 3083):
 *
 *     output[block * 32 + j] = f16_to_f32(scale_bits) * (float)int8_value[j]
 *
 * We don't link against ds4 to avoid drift; instead we re-implement the same
 * IEEE-754 binary16 -> binary32 conversion that ds4 uses on the non-NEON
 * path (ds4.c lines 1535-1567). NumPy's fp16 cast is the same algorithm.
 *
 * Build:  cc -O2 -std=c11 -o q8_0_oracle q8_0_oracle.c
 * Usage:  ./q8_0_oracle <input.bin> <output.bin> <n_blocks>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* subnormal -> normalize */
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        /* Inf / NaN */
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin> <n_blocks>\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    long n_blocks = strtol(argv[3], NULL, 10);
    if (n_blocks <= 0) {
        fprintf(stderr, "n_blocks must be > 0\n");
        return 2;
    }

    FILE *fin = fopen(in_path, "rb");
    if (!fin) { perror(in_path); return 1; }
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { perror(out_path); fclose(fin); return 1; }

    uint8_t block[34];
    float vals[32];
    for (long b = 0; b < n_blocks; b++) {
        if (fread(block, 1, 34, fin) != 34) {
            fprintf(stderr, "short read at block %ld\n", b);
            fclose(fin); fclose(fout);
            return 1;
        }
        uint16_t scale_bits;
        memcpy(&scale_bits, block, 2);
        float d = f16_to_f32(scale_bits);
        const int8_t *q = (const int8_t *)(block + 2);
        for (int j = 0; j < 32; j++) vals[j] = d * (float)q[j];
        if (fwrite(vals, sizeof(float), 32, fout) != 32) {
            perror("write");
            fclose(fin); fclose(fout);
            return 1;
        }
    }
    fclose(fin); fclose(fout);
    return 0;
}
