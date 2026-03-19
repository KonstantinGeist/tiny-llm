#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int byte_to_unicode(unsigned char b) {
    if (b >= 33 && b <= 126) return b;
    if (b >= 161 && b <= 172) return b;
    if (b >= 174) return b;
    int n = 0;
    for (int i = 0; i < 256; i++) {
        int is_printable = (i >= 33 && i <= 126) ||
                           (i >= 161 && i <= 172) ||
                           (i >= 174 && i <= 255);
        if (!is_printable) {
            if (i == (int)b) return 256 + n;
            n++;
        }
    }
    return b; // unreachable
}

static int encode_utf8(uint32_t cp, char *buf) {
    if (cp < 0x80)        { buf[0] = (char)cp; return 1; }
    if (cp < 0x800)       { buf[0] = (char)(0xC0 | (cp >> 6));
                            buf[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000)     { buf[0] = (char)(0xE0 | (cp >> 12));
                            buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            buf[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
                            buf[0] = (char)(0xF0 | (cp >> 18));
                            buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                            buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            buf[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

static int decode_utf8(const unsigned char *s, uint32_t *cp) {
    if (s[0] < 0x80)                          { *cp = s[0]; return 1; }
    if ((s[0] & 0xE0) == 0xC0)                { *cp = ((s[0] & 0x1F) << 6)  | (s[1] & 0x3F); return 2; }
    if ((s[0] & 0xF0) == 0xE0)                { *cp = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3; }
                                                *cp = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4;
}

static int16_t s_u2b[512];
static int    s_u2b_ready = 0;

static void build_u2b(void) {
    for (int i = 0; i < 512; i++) s_u2b[i] = -1;
    for (int b = 0; b < 256; b++) {
        int cp = byte_to_unicode((unsigned char)b);
        if (cp < 512) s_u2b[cp] = (int16_t)b;
    }
    s_u2b_ready = 1;
}

static const char *SPECIAL[] = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", NULL
};

void tok_init(Tokenizer *t,
              char **vocab, int vocab_size,
              char **merges, int n_merges,
              int bos_id, int eos_id) {
    t->vocab      = vocab;
    t->vocab_size = vocab_size;
    t->bos_id     = bos_id;
    t->eos_id     = eos_id;

    t->vocab_map = hm_new();
    for (int i = 0; i < vocab_size; i++)
        hm_put(&t->vocab_map, vocab[i], i);

    t->merge_map = hm_new();
    for (int i = 0; i < n_merges; i++)
        hm_put(&t->merge_map, merges[i], i);

    if (!s_u2b_ready) build_u2b();
}

int tok_encode(const Tokenizer *t, const char *text, int *out) {
    size_t raw_len = strlen(text);
    char *norm = malloc(raw_len * 4 + 1);
    int nlen = 0;
    for (size_t i = 0; i < raw_len; i++) {
        unsigned char c = (unsigned char)text[i];
        int cp = byte_to_unicode(c);
        nlen += encode_utf8((uint32_t)cp, norm + nlen);
    }
    norm[nlen] = '\0';

    int n = 0, len = nlen, pos = 0;
    char buf[256];

    /* Greedy longest-match initial segmentation */
    // Первоначальная сегментация, жадное сопоставление с наиболее длинным токеном.
    while (pos < len) {
        // Сначала специальные токены
        int special_match = -1, special_len = 0;
        for (int s = 0; SPECIAL[s]; s++) {
            int sl = (int)strlen(SPECIAL[s]);
            if (pos + sl <= len && memcmp(norm + pos, SPECIAL[s], sl) == 0) {
                int id = hm_get(&t->vocab_map, SPECIAL[s]);
                if (id >= 0) { special_match = id; special_len = sl; break; }
            }
        }
        if (special_match >= 0) {
            out[n++] = special_match;
            pos += special_len;
            continue;
        }

        // Жадное сопоставление по самому длинному токену
        int max_len = len - pos;
        for (int s = 0; SPECIAL[s]; s++) {
            int sl = (int)strlen(SPECIAL[s]);
            for (int off = 1; off < max_len; off++) {
                if (off + sl <= len - pos &&
                        memcmp(norm + pos + off, SPECIAL[s], sl) == 0) {
                    if (off < max_len) max_len = off;
                    break;
                }
            }
        }
        int try = max_len < 64 ? max_len : 64;
        int match = -1, mlen = 0;
        for (int l = try; l >= 1; l--) {
            memcpy(buf, norm + pos, l); buf[l] = '\0';
            int id = hm_get(&t->vocab_map, buf);
            if (id >= 0) { match = id; mlen = l; break; }
        }
        if (match < 0) { pos++; continue; }
        out[n++] = match;
        pos += mlen;
    }

    free(norm);

    char pair[256];
    int changed = 1;
    while (changed) {
        changed = 0;
        int best_rank = INT32_MAX, best_i = -1, prev = -1;
        for (int i = 0; i < n; i++) {
            if (out[i] < 0) continue;
            if (prev >= 0) {
                snprintf(pair, sizeof(pair), "%s %s",
                         t->vocab[prev], t->vocab[out[i]]);
                int rank = hm_get(&t->merge_map, pair);
                if (rank >= 0 && rank < best_rank) {
                    best_rank = rank; best_i = i;
                }
            }
            prev = out[i];
        }
        if (best_i < 0) break;
        int left = best_i - 1;
        while (left >= 0 && out[left] < 0) left--;
        snprintf(pair, sizeof(pair), "%s%s",
                 t->vocab[out[left]], t->vocab[out[best_i]]);
        int merged = hm_get(&t->vocab_map, pair);
        if (merged < 0) break;
        out[left] = merged; out[best_i] = -1;
        changed = 1;
    }

    int j = 0;
    for (int i = 0; i < n; i++) if (out[i] >= 0) out[j++] = out[i];
    return j;
}

const char *tok_decode(const Tokenizer *t, int token_id) {
    static char buf[512];
    const char *piece = t->vocab[token_id];

    const unsigned char *p = (const unsigned char *)piece;
    int out = 0;
    while (*p && out < (int)sizeof(buf) - 1) {
        uint32_t cp;
        int consumed = decode_utf8(p, &cp);
        p += consumed;
        if (cp < 512 && s_u2b[cp] >= 0) {
            buf[out++] = (char)(uint8_t)s_u2b[cp];
        } else {
            int enc = encode_utf8(cp, buf + out);
            out += enc;
        }
    }
    buf[out] = '\0';
    return buf;
}

void tok_free(Tokenizer *t) {
    free(t->vocab_map.entries);
    free(t->merge_map.entries);
}
