/* q2_k_oracle.c — minimal Q2_K dequant oracle for M6 parity tests.
 *
 * Reads Q2_K-packed bytes from argv[1], dequants in-order, writes fp32
 * little-endian bytes to argv[2]. The dequant formula and bit-packing are
 * lifted verbatim from ds4.c:
 *
 *   - f16_to_f32                        (line 1535)
 *   - struct block_q2_K                 (line 139)
 *   - scalar dot product layout         (ds4_vec_dot_q2_K_q8_K, line 1888 onward,
 *                                        non-NEON path)
 *
 * Per block layout (84 bytes):
 *   [0..16)   uint8 scales[16]   — 4-bit (sub-scale | sub-min << 4)
 *   [16..80)  uint8 qs[64]       — 2-bit packed quants
 *   [80..82)  fp16 d             — super-scale
 *   [82..84)  fp16 dmin          — super-min
 *
 * Per element (sub-block s in 0..15, position i in 0..15):
 *   k                = s / 8
 *   second_in_pair   = s & 1
 *   pair_idx         = (s >> 1) & 3
 *   shift            = 2 * pair_idx
 *   byte             = qs[32*k + 16*second_in_pair + i]
 *   q                = (byte >> shift) & 0x3
 *   out[s*16 + i]    = d * sc * q - dmin * mn
 *
 * We hoist `d_sc = d * sc` and `dmin_mn = dmin * mn` once per sub-block so the
 * per-element compute order is identical to the NumPy implementation
 * (`(d_sc) * q - (dmin_mn)`).
 *
 * Build:  cc -O2 -std=c11 -o q2_k_oracle q2_k_oracle.c
 * Usage:  ./q2_k_oracle <input.bin> <output.bin> <n_blocks>
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
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
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

    uint8_t block[84];
    float vals[256];

    for (long b = 0; b < n_blocks; b++) {
        if (fread(block, 1, 84, fin) != 84) {
            fprintf(stderr, "short read at block %ld\n", b);
            fclose(fin); fclose(fout);
            return 1;
        }

        const uint8_t *scales = block;          /* 16 bytes */
        const uint8_t *qs     = block + 16;     /* 64 bytes */

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    block + 80, 2);
        memcpy(&dmin_bits, block + 82, 2);
        const float d    = f16_to_f32(d_bits);
        const float dmin = f16_to_f32(dmin_bits);

        for (int s = 0; s < 16; s++) {
            const int sc = scales[s] & 0x0f;
            const int mn = scales[s] >> 4;
            const float d_sc    = d    * (float)sc;
            const float dmin_mn = dmin * (float)mn;

            const int k              = s / 8;
            const int second_in_pair = s & 1;
            const int pair_idx       = (s >> 1) & 3;
            const int shift          = 2 * pair_idx;
            const uint8_t *q2_base   = qs + 32 * k + 16 * second_in_pair;

            float *out = vals + s * 16;
            for (int i = 0; i < 16; i++) {
                const int q = (q2_base[i] >> shift) & 0x3;
                out[i] = d_sc * (float)q - dmin_mn;
            }
        }

        if (fwrite(vals, sizeof(float), 256, fout) != 256) {
            perror("write");
            fclose(fin); fclose(fout);
            return 1;
        }
    }
    fclose(fin); fclose(fout);
    return 0;
}
